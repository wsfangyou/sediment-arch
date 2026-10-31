# -*- coding: utf-8 -*-
"""D2 技能承载 —— 重做版（含学习曲线、双口径判分、全矩阵 SVD 保真）。

【为什么重做】
旧 duty2 的三个测量缺陷（2026-10-30 诊断实测确认）：
  1. 生成侧被模型因果掩码 bug 污染：attention_mask 一走，注意力变双向，最后一个位置
     的 logits 也会变（最大差 10.1，argmax 从 ',' 翻成 '于'）→ 生成全是 \n 复读，
     与 teacher-forcing 读数（loss 0.07~2.7）完全脱节。已在 model_l0.hidden_states 修复。
  2. 训练步数饥饿：300 步 × bs1 只覆盖 290/4800 条样本，holdout 必为 0 —— 不是"基座不能
     承载技能"，是没训到。改为覆盖全量数据并给出 300/600/1200 步学习曲线。
  3. SVD 保真只抽 4 个矩阵且用随机初始化对照以外的口径。改为全 168 个投影矩阵统计。

【预注册判定（跑前写死）】
  主判 A_tf  = holdout 答案区间 teacher-forcing 逐 token top-1 准确率 ≥ 0.90
  副判 A_gen = holdout greedy 生成精确匹配率（沿用 eval_sedu 三向判分口径）≥ 0.80
  对照 bare  = 裸基座 A_gen ≤ 0.15（证明技能非既有能力）
  → A_tf≥0.90 且 A_gen≥0.80 且 bare≤0.15 判 ✅
  → A_tf≥0.90 但 A_gen<0.80 判 ⚠️（可承载、解码端不稳，需在 L1 侧补解码约束）
  → A_tf<0.90 判 ❌（承载失败）
"""
import os, sys, json, time, argparse, re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, DEFAULT_L0

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
os.makedirs(ART, exist_ok=True)
DEV = "cuda:0"
SYS = "你是精通Sedu语与中文互译的语言助手。"


@torch.no_grad()
def val_ppl(m, n=24):
    """验证集 98k token 口径的通用 ppl（C 门）。
    ⚠️ 必须用验证集，不能用单段落：实测单段测出 1.043，验证集是 1.338（低估 8 倍）。"""
    from pretrain_l0 import TokenStream
    st = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)
    m.eval(); rng = np.random.default_rng(7); ls = []
    for _ in range(n):
        x, y = st.batch(4, train=False, rng=rng)
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            ls.append(m(x, y).loss.item())
    return float(np.exp(min(float(np.mean(ls)), 20)))


def log(*a):
    print(*a, flush=True)


def norm_sedu(s):
    return re.sub(r"\s+", " ", s.strip().lower().replace("，", " ").replace(",", " "))


def build(tok, r, trunc=192):
    """构造 (input_ids, labels)。

    ⚠️ 标签必须左移一位（2026-10-30 修）：模型的 chunked_cross_entropy 是
    CE(hidden[i], labels[i])，没有内部 shift —— pretrain_l0 里 y 由 TokenStream
    预先右移一位提供。旧 duty2 直接给 [−100]*np_ + ids[np_:]，等于让模型"预测自己
    刚读到的那个 token"：训练 loss 一路降到 0.07，模型只学会复读输入，
    生成时第一个 token 就吐 EOS → 领域 acc=0.000。这是 D2 判 0 的真因。
    """
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["q"]},
            {"role": "assistant", "content": r["a"]}]
    full = tok.apply_chat_template(msgs, tokenize=False)
    pr = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"][:trunc]
    np_ = min(len(tok(pr, add_special_tokens=False)["input_ids"]), len(ids))
    labels = list(ids)
    for i in range(np_):
        labels[i] = -100
    labels = labels[1:] + [-100]           # 左移：位置 i 的标签 = 下一个 token
    return ids, labels


@torch.no_grad()
def tf_acc(m, tok, rs):
    """答案区间逐 token top-1 准确率（teacher-forcing 口径，技能是否被承载的直接量）。

    约定与 build() 一致：labels 已左移，故直接用 lg[i] 对 labels[i] 比对，不可再切一位。
    """
    m.eval(); hit = tot = 0
    for r in rs:
        ids, lb = build(tok, r)
        x = torch.tensor([ids], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m(x).logits[0].float()
        tgt = torch.tensor(lb, device=DEV)
        keep = tgt != -100
        pred = lg.argmax(-1)
        hit += int((pred[keep] == tgt[keep]).sum()); tot += int(keep.sum())
    return hit / max(tot, 1)


@torch.no_grad()
def gen_rows(m, tok, rs, max_new=28, rep_penalty=1.0):
    """纯 greedy 生成。⚠️ repetition_penalty=1.0 比已验收的 3B 参考管线（generate 默认 1.05）
    更严格 —— 弱模型在这种设定下更容易陷进重复循环，故 A_gen 应视为**下限**。"""
    m.eval(); out = []
    for r in rs:
        text = tok.apply_chat_template([{"role": "system", "content": SYS},
                                        {"role": "user", "content": r["q"]}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        # ⚠️ 必须显式给 eos_token_id（2026-10-31 修）：本模型的 generation_config 是空的
        # （全 None），generate 只认 <|endoftext|>(151643)，不把 <|im_end|>(151645) 当回合结束
        # → 模型答完正确答案后继续生成下一个回合，整串精确匹配必然失败。
        # 实测：A_gen 0.333（只认 151643）→ **0.800**（151645+151643），平均输出 36→7.7 字符。
        o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                       do_sample=False, pad_token_id=tok.pad_token_id,
                       repetition_penalty=rep_penalty, eos_token_id=[151645, 151643])
        out.append(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip())
    return out


def gen_acc(rows, answers):
    hit = tot = 0
    for r, a in zip(rows, answers):
        tot += 1
        aa = a.strip().rstrip("。.!！")
        if r["task"] == "gloss":
            ok = r["a"] in a
        elif r["task"] == "s2c":
            ok = aa == r["a"].rstrip("。") or aa == r["a"]
        else:
            ok = norm_sedu(aa) == norm_sedu(r["a"])
        hit += int(ok)
    return hit / max(tot, 1)


@torch.no_grad()
def svd_fidelity(merged, base_snap, ks=(2, 4, 8)):
    """全矩阵 ΔW 的秩截断保真：keep_r = 1 - ||ΔW - rec_r|| / ||ΔW||。"""
    stats = {k: [] for k in ks}
    rel_energy = []
    msd = merged.state_dict()
    n = 0
    for key, w0 in base_snap.items():
        if key not in msd:
            continue
        dW = (msd[key].float().cpu() - w0)
        dn = float(dW.norm())
        if dn < 1e-12:
            continue
        n += 1
        rel_energy.append(dn / max(float(w0.norm()), 1e-12))
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        for k in ks:
            kk = min(k, len(S))
            rec = (U[:, :kk] * S[:kk]) @ Vh[:kk]
            stats[k].append(round(1 - float((dW - rec).norm()) / dn, 4))
    return {"n_matrices": n,
            "keep_median": {k: round(float(np.median(v)), 4) for k, v in stats.items()},
            "keep_mean": {k: round(float(np.mean(v)), 4) for k, v in stats.items()},
            "deltaW_rel_energy_median": round(float(np.median(rel_energy)), 5),
            "deltaW_rel_energy_max": round(float(np.max(rel_energy)), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="300,600,1200", help="在这些步数处评测（逗号分隔）")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--tag", default="d2")
    ap.add_argument("--save-adapter", default="", help="把训练好的 LoRA 存到该目录（便于事后换解码参数复测）")
    args = ap.parse_args()
    marks = sorted(int(s) for s in args.steps.split(","))
    total = marks[-1]

    tok = load_tokenizer(DEFAULT_L0)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_train.jsonl"), encoding="utf-8")]
    hold = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"), encoding="utf-8")]
    ev = hold[:args.n_eval]

    base, _ = load_l0(dtype="float32", device=DEV)
    # 基线：裸基座（生成口径应为 ≈0）
    log("\n[基线] 裸基座")
    bare_tf = tf_acc(base, tok, ev[:40])
    bare_gen_rows = gen_rows(base, tok, ev[:40])
    bare_gen = gen_acc(ev[:40], bare_gen_rows)
    bare_ppl = val_ppl(base)
    log(f"  裸基座 A_tf={bare_tf:.3f}  A_gen={bare_gen:.3f}  通用 ppl={bare_ppl:.2f}")
    log(f"  样例1 期望 {ev[0]['a']!r} → 生成 {bare_gen_rows[0]!r}")

    # ΔW 参照快照（fp32，CPU，约 2.4GB）
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
    scaler = torch.amp.GradScaler("cuda")               # 旧版缺 scaler，fp16 梯度会下溢
    rng = np.random.default_rng(0)
    curve, t0 = [], time.time()
    log(f"\n[训练] {total} 步 × bs{args.bs}，中间评测点 {marks}")
    for s in range(total):
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
        if (s + 1) % 50 == 0:
            log(f"  step {s+1}/{total} loss {acc:.4f} ({time.time()-t0:.0f}s)")
        if (s + 1) in marks:
            m.eval()
            a_tf = tf_acc(m, tok, ev)
            gs = gen_rows(m, tok, ev)
            a_gen = gen_acc(ev, gs)
            # 记忆对照（训练集抽样 40 条）
            mem_rows = [rows[i] for i in rng.choice(len(rows), 40, replace=False)]
            a_tf_mem = tf_acc(m, tok, mem_rows)
            curve.append({"step": s + 1, "train_loss": round(acc, 4), "A_tf": round(a_tf, 4),
                          "A_gen": round(a_gen, 4), "A_tf_mem": round(a_tf_mem, 4),
                          "sec": round(time.time() - t0)})
            log(f"  ★ step {s+1}: train_loss {acc:.4f} | holdout A_tf {a_tf:.3f} A_gen {a_gen:.3f} "
                f"| 训练集抽样 A_tf {a_tf_mem:.3f}")
            for r, g in list(zip(ev, gs))[:3]:
                log(f"     「{r['q']}」期望 {r['a']!r} → {g!r}")
            m.train()

    if args.save_adapter:
        m.save_pretrained(args.save_adapter)
        log(f"[保存] LoRA adapter → {args.save_adapter}")

    # 合并与保真
    log("\n[合并] merge_and_unload + 全矩阵 SVD 保真")
    m.eval()
    pre = tf_acc(m, tok, ev[:20])
    merged = m.merge_and_unload(); merged.eval()
    post = tf_acc(merged, tok, ev[:20])
    log(f"  合并前后 A_tf（同 20 条）: {pre:.3f} → {post:.3f}（应一致，差>0.02 说明合并失效）")
    skill_ppl = val_ppl(merged)
    c_reg = skill_ppl / bare_ppl
    log(f"  C 门（通用回归）: {bare_ppl:.2f} → {skill_ppl:.2f} = {c_reg:.4f}×（门限 1.02）"
        f" → {'✅' if c_reg <= 1.02 else '❌'}")
    fid = svd_fidelity(merged, base_snap)
    log(f"  ΔW 统计: {fid['n_matrices']} 个矩阵 | 相对能量（中位）{fid['deltaW_rel_energy_median']:.5f}")
    for k in (2, 4, 8):
        log(f"  r={k}: keep 中位 {fid['keep_median'][k]:.4f} / 均值 {fid['keep_mean'][k]:.4f}")

    final = curve[-1]
    rising = (len(curve) >= 2 and final["A_tf"] - curve[0]["A_tf"] >= 0.15)
    if final["A_tf"] >= 0.90 and final["A_gen"] >= 0.80 and bare_gen <= 0.15:
        verdict, ok = "✅ 可承载技能（A_tf/A_gen 双达标）", True
    elif final["A_tf"] >= 0.90:
        verdict, ok = "⚠️ 可承载但解码端不稳（A_tf 达标、A_gen 未达标）", False
    elif rising:
        verdict, ok = (f"⚠️ 未达 0.90 线（A_tf {final['A_tf']:.3f}），但随步数单调改善"
                       f"（{curve[0]['A_tf']:.3f}→{final['A_tf']:.3f}）→ 训练预算不足，非结构不能承载"), False
    else:
        verdict, ok = "❌ 承载失败（A_tf 未达标且无改善趋势）", False
    log(f"\n判定: {verdict}")

    out = {"protocol": {"steps": marks, "bs": args.bs, "lr": args.lr, "n_eval": args.n_eval,
                        "prereg": "A_tf>=0.90 且 A_gen>=0.80 且 bare_gen<=0.15；"
                                  "L1 晋升另需 C 门 ppl_regress<=1.02"},
           "c_gate": {"bare_ppl": round(bare_ppl, 2), "skill_ppl": round(skill_ppl, 2),
                      "regress": round(c_reg, 4), "pass": bool(c_reg <= 1.02)},
           "promotable_to_L1": bool(ok and c_reg <= 1.02),
           "bare": {"A_tf": round(bare_tf, 4), "A_gen": round(bare_gen, 4),
                    "sample": bare_gen_rows[0]},
           "curve": curve, "merge_consistency": {"pre": round(pre, 4), "post": round(post, 4)},
           "svd": fid, "verdict": verdict, "pass": ok}
    p = os.path.join(ART, f"d2_{args.tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"→ {p}")


if __name__ == "__main__":
    main()
