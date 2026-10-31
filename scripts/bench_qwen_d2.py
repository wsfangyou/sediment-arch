# -*- coding: utf-8 -*-
"""把项目的 D2 判分协议原样搬到任意 HF 因果模型上（用于评估"换 4B 级 Qwen 基座"）。

与 L0 版唯一的差别：
  · L0 的 SedimentBase.forward **不做 labels 内部 shift**，所以要传已右移的 labels；
  · HF 模型（Qwen）**自带内部 shift**，所以传未右移、只在 prompt 区打 -100 的 labels。
判分代码（A_tf / A_gen / C 门 ppl）直接复用 accept_l0_d2，保证与 L0 的验收口径完全一致。

预注册判定与 L0 相同：
  A_tf >= 0.90 且 A_gen >= 0.80 且 bare A_gen <= 0.15 → ✅ 可承载技能
  另报 C 门（通用 ppl 回归）：严格 1.02 / 观测 1.05
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accept_l0_d2 import gen_acc, gen_rows, tf_acc, SYS      # 同一套判分（除 ppl 外）
from loader import load_tokenizer
from pretrain_l0 import TokenStream

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
DEV = "cuda:0"


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def val_ppl_hf(m, n=24):
    """C 门 ppl（98k token 验证集口径）。
    ⚠️ HF 模型内部会 shift，所以 labels 必须传**未右移**的 x（不能像 L0 那样传 TokenStream 的 y）。"""
    st = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)
    m.eval()
    rng = np.random.default_rng(7)
    ls = []
    for _ in range(n):
        x, _ = st.batch(4, train=False, rng=rng)
        x = x.to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            ls.append(m(input_ids=x, labels=x).loss.item())
    return float(np.exp(min(float(np.mean(ls)), 20)))


def build_hf(tok, r, trunc=192):
    """HF 口径：labels 不右移，只在 prompt 区打 -100（模型内部会 shift）。"""
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["q"]},
            {"role": "assistant", "content": r["a"]}]
    full = tok.apply_chat_template(msgs, tokenize=False)
    pr = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"][:trunc]
    np_ = min(len(tok(pr, add_special_tokens=False)["input_ids"]), len(ids))
    return ids, [-100] * np_ + ids[np_:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--replay", type=float, default=0.35)
    ap.add_argument("--gen-seq", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--save-adapter", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(DEV).eval()
    log(f"[{args.tag}] 已加载 {args.model}，参数 {sum(p.numel() for p in base.parameters())/1e9:.3f}B")

    tr = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_train.jsonl"), encoding="utf-8")]
    ev = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"), encoding="utf-8")][:args.n_eval]

    # ---- 裸基座基线（技能不是既有能力）----
    bare_tf = tf_acc(base, tok, ev)
    bare_gen = gen_acc(ev, gen_rows(base, tok, ev))
    bare_ppl = val_ppl_hf(base)
    log(f"  裸基座 A_tf={bare_tf:.3f} A_gen={bare_gen:.3f} 通用 ppl={bare_ppl:.2f}")

    from peft import LoraConfig, TaskType, get_peft_model
    m = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.02,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    m.config.use_cache = False
    # 梯度检查点：4B/3B 在 22GB 卡上不开会 OOM（bs8 实测撞 20.2 GiB）
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m.train()
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    log(f"  可训练 {n_tr/1e6:.2f}M ({n_tr/sum(p.numel() for p in m.parameters())*100:.3f}%)")

    stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), args.gen_seq)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    pk = tok.pad_token_id or tok.eos_token_id
    t0 = time.time()
    n_replay = 0
    for s in range(args.steps):
        if rng.random() < args.replay:                        # 混采通用中文（纯 LM 目标）
            # ⚠️ TokenStream 的 y 已右移（L0 口径）；HF 模型内部会再 shift，所以这里必须传未右移的 x
            x, _ = stream.batch(args.bs, train=True, rng=rng)
            ids = x.to(DEV)
            labels = ids.clone()
            n_replay += 1
        else:
            rows = [tr[int(rng.integers(len(tr)))] for _ in range(args.bs)]
            pairs = [build_hf(tok, r) for r in rows]
            L = max(len(p[0]) for p in pairs)
            ids = torch.full((len(pairs), L), pk, dtype=torch.long, device=DEV)
            labels = torch.full((len(pairs), L), -100, dtype=torch.long, device=DEV)
            for i, (a, b) in enumerate(pairs):
                ids[i, :len(a)] = torch.tensor(a, device=DEV)
                labels[i, :len(b)] = torch.tensor(b, device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = m(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (s + 1) % 100 == 0:
            log(f"  step {s+1}/{args.steps} loss {loss.item():.4f} "
                f"已混采 {n_replay} ({time.time()-t0:.0f}s)")

    m.eval()
    if args.save_adapter:
        m.save_pretrained(args.save_adapter)
        log(f"  adapter → {args.save_adapter}")

    a_tf = tf_acc(m, tok, ev)
    gs = gen_rows(m, tok, ev)
    a_gen = gen_acc(ev, gs)
    ppl = val_ppl_hf(m)
    reg = ppl / bare_ppl
    log(f"  训练后 A_tf={a_tf:.3f} A_gen={a_gen:.3f} 通用 ppl={ppl:.2f} = {reg:.4f}×裸基座")

    ok = bool(a_tf >= 0.90 and a_gen >= 0.80 and bare_gen <= 0.15)
    cg = bool(reg <= 1.02)
    verdict = ("✅ D2 全过（A_tf/A_gen 双达标）" if ok else
               ("⚠️ A_tf 过线但 A_gen 未达" if a_tf >= 0.90 else "❌ A_tf 未过线"))
    cg_txt = ("✅ 严格线 1.02" if cg else
              ("🟡 观测线 1.05" if reg <= 1.05 else "❌ 超观测线"))
    log(f"  【判定】{verdict} | C 门 {reg:.4f}× → {cg_txt}")
    for r, g in list(zip(ev, gs))[:5]:
        log(f"    期望 {r['a']!r} → 得到 {g!r}")

    out = {"protocol": {"model": args.model, "steps": args.steps, "bs": args.bs, "lr": args.lr,
                        "replay": args.replay, "gen_seq": args.gen_seq, "n_eval": args.n_eval,
                        "eos_fix": "eos_token_id=[151645,151643]",
                        "prereg": "A_tf>=0.90 且 A_gen>=0.80 且 bare<=0.15；C 门严格 1.02 / 观测 1.05"},
           "bare": {"A_tf": round(bare_tf, 4), "A_gen": round(bare_gen, 4), "ppl": round(bare_ppl, 2)},
           "after": {"A_tf": round(a_tf, 4), "A_gen": round(a_gen, 4)},
           "c_gate": {"ppl": round(ppl, 2), "regress": round(reg, 4),
                      "pass_strict": cg, "pass_observed": bool(reg <= 1.05)},
           "verdict": verdict, "trainable_m": round(n_tr / 1e6, 2),
           "samples": [{"q": r["q"], "expect": r["a"], "gen": g} for r, g in zip(ev[:20], gs[:20])]}
    p = os.path.join(ART, f"{args.tag}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"  → {p}")


if __name__ == "__main__":
    main()
