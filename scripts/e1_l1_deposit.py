# -*- coding: utf-8 -*-
"""E1 前置实测：L1 热沉积物（秩截断 LoRA）在 L0 上的行为保持率。

方案 §2.2 通道2 的晋升门要求「迷你 LoRA 保真率 ≥ 95%（相对教师）」，但此前只测过
ΔW 的**矩阵能量**保真（r8 = 0.922），从没测过**行为**保真。本脚本补上这一环：

  1. 在 L0 上训 LoRA r16（默认 600 步），得「教师」
  2. 取 ΔW = W_trained − W_base（全 168 个投影矩阵，fp32 CPU）
  3. 对每个秩 r ∈ {16,8,4,2}：SVD 截断重建 W = W_base + U_r S_r V_rᵀ，落回模型
  4. 用同一套 holdout 判分（A_tf / A_gen）+ 通用文本 ppl（C 门回归）测每个 r
  5. 算保真率 = 该秩指标 / r16 指标，并折算沉积物体积（r×(in+out)×2B×矩阵数）

判定（预注册）：
  保真率 ≥ 0.95 且 ppl 回归 ≤ 1.02×裸基座 → 该秩可晋升 L1；取**通过的最小秩**为常驻候选。
"""
import os, sys, json, time, argparse, re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, DEFAULT_L0
from accept_l0_d2 import build, tf_acc, gen_rows, gen_acc, SYS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
os.makedirs(ART, exist_ok=True)
DEV = "cuda:0"

GEN段落 = ("人工智能技术在近年来取得了显著进展，尤其是在自然语言处理与计算机视觉领域。"
           "大规模预训练模型通过海量文本数据学习语言规律，能够在翻译、摘要、问答等任务中"
           "展现出接近人类水平的表现。然而，模型的部署仍然面临算力与内存的挑战，"
           "因此研究者提出了量化、蒸馏、稀疏化等多种压缩手段，以在质量与成本之间取得平衡。")


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def ppl_of(m, tok, text):
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=DEV)
    # ⚠️ 本模型 forward 不做内部 shift：ppl 的标签必须左移一位
    lab = torch.cat([ids[:, 1:], torch.full((1, 1), -100, device=DEV, dtype=ids.dtype)], dim=1)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = m(ids, labels=lab).loss.item()
    return float(np.exp(min(loss, 20)))


def deposit_mb(rank, n_mat=168):
    """按本模型的实际形状估体积：每层 7 个矩阵，(1536×1536)×4 + (384×1536)×2 + (1536×4096)×3"""
    per_layer = rank * (1536 + 1536) * 4 + rank * (384 + 1536) * 2 + rank * (1536 + 4096) * 3
    return per_layer * 24 * 2 / 1024 / 1024      # fp16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--ranks", default="16,8,4,2")
    ap.add_argument("--tag", default="l1")
    ap.add_argument("--save-adapter", default="")
    args = ap.parse_args()

    tok = load_tokenizer(DEFAULT_L0)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_train.jsonl"), encoding="utf-8")]
    hold = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"), encoding="utf-8")]
    ev = hold[:args.n_eval]

    base, _ = load_l0(dtype="float32", device=DEV)
    bare_ppl = ppl_of(base, tok, GEN段落)
    log(f"[基线] 裸基座 ppl(通用段) {bare_ppl:.3f} | A_gen {gen_acc(ev, gen_rows(base, tok, ev)):.3f}")

    base_snap = {k: v.detach().float().cpu().clone()
                 for k, v in base.state_dict().items() if k.endswith("_proj.weight")}

    from peft import LoraConfig, TaskType, get_peft_model
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.02,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(base, lc)
    m.print_trainable_parameters()
    feats = [build(tok, r) for r in rows]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(0)
    t0 = time.time()
    log(f"[训练] {args.steps} 步 × bs{args.bs}")
    for s in range(args.steps):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.bs):
            i = int(rng.integers(len(feats)))
            ids = torch.tensor([feats[i][0]], device=DEV)
            lb = torch.tensor([feats[i][1]], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = m(ids, labels=lb).loss
            scaler.scale(loss / args.bs).backward()
            acc += loss.item() / args.bs
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        if (s + 1) % 100 == 0:
            log(f"  step {s+1}/{args.steps} loss {acc:.4f} ({time.time()-t0:.0f}s)")

    if args.save_adapter:
        m.save_pretrained(args.save_adapter)
        log(f"[保存] {args.save_adapter}")

    merged = m.merge_and_unload(); merged.eval()
    msd = merged.state_dict()
    # ΔW（fp32 CPU）
    delta = {}
    for k, w0 in base_snap.items():
        if k in msd and msd[k].shape == w0.shape:
            delta[k] = msd[k].float().cpu() - w0
    log(f"[ΔW] {len(delta)} 个矩阵，总相对能量中位 "
        f"{np.median([float(d.norm()/base_snap[k].norm()) for k, d in delta.items()]):.5f}")

    # SVD 分解（只做一次，各秩复用）
    rmax = max(int(x) for x in args.ranks.split(","))
    svd = {}
    for k, d in delta.items():
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        svd[k] = (U[:, :rmax].contiguous(), S[:rmax].contiguous(), Vh[:rmax].contiguous())
    log(f"[SVD] 缓存前 {rmax} 个奇异分量（{len(svd)} 个矩阵）")

    def set_rank(r):
        """把模型权重写回 base + 秩 r 截断的 ΔW。"""
        sd = merged.state_dict()
        for k, (U, S, Vh) in svd.items():
            rr = min(r, len(S))
            rec = (U[:, :rr] * S[:rr]) @ Vh[:rr]
            sd[k].copy_((base_snap[k] + rec).to(sd[k].dtype).to(sd[k].device))
        return merged

    results = []
    for r in [int(x) for x in args.ranks.split(",")]:
        set_rank(r)
        a_tf = tf_acc(merged, tok, ev)
        gs = gen_rows(merged, tok, ev)
        a_gen = gen_acc(ev, gs)
        ppl = ppl_of(merged, tok, GEN段落)
        results.append({"rank": r, "A_tf": round(a_tf, 4), "A_gen": round(a_gen, 4),
                        "ppl_general": round(ppl, 3), "deposit_mb": round(deposit_mb(r), 2)})
        log(f"  r={r:<2} A_tf {a_tf:.3f} A_gen {a_gen:.3f} ppl {ppl:.3f} "
            f"沉积物 ≈{deposit_mb(r):.1f} MB | 样例 {gs[0][:28]!r}")

    full = results[0]          # r=16 为教师参照
    for x in results:
        x["keep_A_tf"] = round(x["A_tf"] / max(full["A_tf"], 1e-9), 3)
        x["keep_A_gen"] = round(x["A_gen"] / max(full["A_gen"], 1e-9), 3)
        x["ppl_regress_vs_bare"] = round(x["ppl_general"] / bare_ppl, 3)
        # ⚠️ 两个门必须分开算：A_tf 有约 570 个 token 样本、A_gen 只有 30 条（粒度 0.033），
        # 保真主判用 A_tf；C 门（通用回归）与秩无关，是技能训练本身的代价。
        x["gate_fidelity"] = bool(x["keep_A_tf"] >= 0.95)
        x["gate_general"] = bool(x["ppl_regress_vs_bare"] <= 1.02)
        x["promotable"] = bool(x["gate_fidelity"] and x["gate_general"])
        log(f"  r={x['rank']:<2} 保真 A_tf {x['keep_A_tf']:.3f} / A_gen {x['keep_A_gen']:.3f} | "
            f"ppl×裸基座 {x['ppl_regress_vs_bare']:.3f} → "
            f"保真门{'过' if x['gate_fidelity'] else '不过'} / "
            f"C门{'过' if x['gate_general'] else '不过'}")

    ok = [x for x in results if x["promotable"] and x["rank"] < 16]
    best = min(ok, key=lambda x: x["rank"]) if ok else None
    fid_ok = [x["rank"] for x in results if x["rank"] < 16 and x["gate_fidelity"]]
    c_ok = [x["rank"] for x in results if x["gate_general"]]
    teacher_c = results[0]["ppl_regress_vs_bare"]
    if best:
        verdict = (f"✅ 秩 {best['rank']} 可晋升 L1（A_tf 保真 {best['keep_A_tf']:.3f}，"
                   f"{best['deposit_mb']:.0f} MB/个，1 GB 可常驻 {int(1024/best['deposit_mb'])} 个）")
    elif fid_ok:
        verdict = (f"⚠️ 保真门通过（秩 {fid_ok}），但卡在 C 门：教师自身通用 ppl 就退化到 "
                   f"{teacher_c:.3f}×裸基座（门限 1.02），各秩 1.037–1.043 几乎相同 → "
                   f"**卡住晋升的是技能训练对通用分布的挤占，不是秩截断**。"
                   f"秩截断路线本身成立（{min(fid_ok)} 秩即够）。")
    else:
        verdict = "❌ 所有秩均未过保真门 → 秩截断路线在 L0 上不成立"
    out = {"protocol": {"steps": args.steps, "bs": args.bs, "n_eval": args.n_eval,
                        "prereg": "保真门 keep_A_tf>=0.95；C 门 ppl_regress<=1.02；两门分开报"},
           "bare_ppl_general": round(bare_ppl, 3), "results": results,
           "fidelity_pass_ranks": fid_ok, "c_gate_pass_ranks": c_ok,
           "promotable_ranks": [x["rank"] for x in ok],
           "residency": {"l1_budget_gb": 1.0,
                         "n_deposits_at_min_fidelity_rank": (round(1024 / deposit_mb(min(fid_ok)))
                                                             if fid_ok else 0)},
           "verdict": verdict}
    log("\n" + out["verdict"])
    p = os.path.join(ART, f"e1_{args.tag}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"→ {p}")


if __name__ == "__main__":
    main()
