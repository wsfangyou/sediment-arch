# -*- coding: utf-8 -*-
"""E2-replay：技能训练混采通用语料，能否救回 C 门（通用回归）？

背景〔实测，2026-10-31〕：纯 Sedu 技能训练 600 步后，
  技能能力 A_tf 0.727，但通用 ppl 56.92 → 76.17（1.338×，门限 1.02）。
本实验唯一变量 = 每个训练步以 20% 概率抽一条通用中文样本（来自 corpus/tokens.bin，
纯 LM 目标，不做 -100 掩码），其余同 E1（LoRA r16 / bs4 / 600 步 / 同 lr）。

预注册判定：
  · C 门救援：ppl_regress ≤ 1.02 → ✅ 完全救回；≤1.15 → ⚠️ 部分救回；否则 ❌
  · 技能代价：A_tf 相对纯技能基线（0.727）的保持率 ≥0.90 → 可接受，<0.80 → 混采拖累太大
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, DEFAULT_L0
from pretrain_l0 import TokenStream
from accept_l0_d2 import build, tf_acc, gen_rows, gen_acc, SYS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
DEV = "cuda:0"


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def val_ppl(m, stream, n=24):
    m.eval(); rng = np.random.default_rng(7); ls = []
    for _ in range(n):
        x, y = stream.batch(4, train=False, rng=rng)
        x, y = x.to(DEV), y.to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ls.append(m(x, y).loss.item())
    import math
    vl = float(np.mean(ls))
    return math.exp(min(vl, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--replay", type=float, default=0.2, help="通用语料混采比例")
    ap.add_argument("--gen-seq", type=int, default=256, help="通用样本序列长度")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--tag", default="replay20")
    ap.add_argument("--marks", default="", help="中途评测点（逗号分隔），只测 A_tf，不跑生成")
    ap.add_argument("--init-adapter", default="", help="从已有 LoRA 继续训（续训/加步数）")
    ap.add_argument("--fast-baseline", action="store_true", help="跳过裸基座的生成评测（省 3 分钟）")
    ap.add_argument("--save-adapter", default="")
    args = ap.parse_args()

    tok = load_tokenizer(DEFAULT_L0)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_train.jsonl"), encoding="utf-8")]
    hold = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"), encoding="utf-8")]
    ev = hold[:args.n_eval]
    stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), args.gen_seq)
    # ⚠️ C 门 ppl 必须与 e1_cgate 同口径（seq 1024），否则与 1.338× 不可比
    vstream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)

    base, _ = load_l0(dtype="float32", device=DEV)
    ppl_bare = val_ppl(base, vstream)
    bare_tf = tf_acc(base, tok, ev)
    if args.fast_baseline:
        bare_gen = 0.0                          # 已在多轮实验中实测为 0.000
        log(f"[基线] 裸基座 通用 ppl {ppl_bare:.2f} | A_gen 0.000（沿用已知值，跳过生成）"
            f" | A_tf {bare_tf:.3f}")
    else:
        bare_gen = gen_acc(ev, gen_rows(base, tok, ev))
        log(f"[基线] 裸基座 通用 ppl {ppl_bare:.2f} | A_gen {bare_gen:.3f} | A_tf {bare_tf:.3f}")

    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.02,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])
    if args.init_adapter:
        # is_trainable=True 必须给：默认 False 会把 LoRA 参数冻住 → optimizer 空参数表
        m = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        log(f"[续训] 载入 adapter {args.init_adapter}，继续训 {args.steps} 步")
    else:
        m = get_peft_model(base, lc)
    m.print_trainable_parameters()
    feats = [build(tok, r) for r in rows]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    marks = sorted(int(x) for x in args.marks.split(",") if x.strip().isdigit())
    tf_curve = []
    rng = np.random.default_rng(0)
    t0 = time.time(); n_replay = 0
    log(f"[训练] {args.steps} 步 × bs{args.bs}，通用混采比例 {args.replay:.0%}（seq {args.gen_seq}）")
    for s in range(args.steps):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.bs):
            if rng.random() < args.replay:
                x, y = stream.batch(1, train=True, rng=rng)
                x, y = x.to(DEV), y.to(DEV)          # 纯 LM：y 已是右移一位
                n_replay += 1
            else:
                i = int(rng.integers(len(feats)))
                x = torch.tensor([feats[i][0]], device=DEV)
                y = torch.tensor([feats[i][1]], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = m(x, labels=y).loss
            scaler.scale(loss / args.bs).backward()
            acc += loss.item() / args.bs
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        if (s + 1) % 100 == 0:
            log(f"  step {s+1}/{args.steps} loss {acc:.4f} 已混采 {n_replay} 条 ({time.time()-t0:.0f}s)")
        if (s + 1) in marks:
            m.eval()
            a_now = tf_acc(m, tok, ev)
            tf_curve.append({"step": s + 1, "A_tf": round(a_now, 4)})
            log(f"  ★ step {s+1}: holdout A_tf {a_now:.3f}（中途点，仅 teacher-forcing）")
            m.train()

    if args.save_adapter:
        m.save_pretrained(args.save_adapter)
    m.eval()
    merged = m.merge_and_unload(); merged.eval()
    a_tf = tf_acc(merged, tok, ev)
    gs = gen_rows(merged, tok, ev)
    a_gen = gen_acc(ev, gs)
    ppl_skill = val_ppl(merged, vstream)
    reg = ppl_skill / ppl_bare
    keep = a_tf / 0.727            # 与 E1 纯技能 600 步基线比
    log(f"\n[结果] A_tf {a_tf:.3f}（纯技能基线 0.727，保持 {keep:.3f}）"
        f" | A_gen {a_gen:.3f} | 通用 ppl {ppl_skill:.2f} = {reg:.4f}×裸基座")
    for r, g in list(zip(ev, gs))[:3]:
        log(f"   「{r['q']}」期望 {r['a']!r} → {g!r}")
    if reg <= 1.02:
        verdict = f"✅ C 门完全救回（{reg:.3f}×，门限 1.02）"
    elif reg <= 1.15:
        verdict = f"⚠️ C 门部分救回（{reg:.3f}×，较纯技能的 1.338× 明显改善，仍不过 1.02 门）"
    else:
        verdict = f"❌ C 门未救回（{reg:.3f}×）"
    log("判定: " + verdict)

    out = {"protocol": {"steps": args.steps, "bs": args.bs, "lr": args.lr,
                        "replay_ratio": args.replay, "gen_seq": args.gen_seq,
                        "n_eval": args.n_eval, "replay_samples": n_replay,
                        "prereg": "C 门 ppl_regress<=1.02 全救回 / <=1.15 部分救回；技能保持率>=0.90 可接受"},
           "bare": {"ppl": round(ppl_bare, 2), "A_gen": round(bare_gen, 3)},
           "tf_curve": tf_curve,
           "after": {"A_tf": round(a_tf, 4), "A_gen": round(a_gen, 4),
                     "ppl_general": round(ppl_skill, 2)},
           "ppl_regress_vs_bare": round(reg, 4),
           "skill_retention_vs_pure": round(keep, 4),
           "baseline_pure_skill": {"A_tf": 0.727, "ppl_regress_vs_bare": 1.338,
                                   "source": "e1_l1.json（同 600 步 / bs4 / 无混采）"},
           "verdict": verdict}
    p = os.path.join(ART, f"e2_{args.tag}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"→ {p}")


if __name__ == "__main__":
    main()
