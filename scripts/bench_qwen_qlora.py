# -*- coding: utf-8 -*-
"""B 线 B1：4bit（NF4）底座上直接训练 LoRA —— QLoRA 路径能不能用？

已经验过"fp16 训练 → 4bit 推理"不掉点；这里验"4bit 训练"本身：
  · 显存多少（决定能不能腾出更多 batch / 更长上下文）
  · 训练 400 步后 A_tf / A_gen 掉多少（对标 fp16 训练的 0.994 / 0.960）
  · 速度多少（bnb 反量化有额外开销）

判据：A_tf ≥0.90 且 A_gen ≥0.80 → QLoRA 可作 B 线常态训练路径。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_qwen_d2 import build_hf, val_ppl_hf
from qwen4b_quant_gap import gen_strip, THINK

DEV = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
MODEL = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B"
SYS = "你是精通Sedu语与中文互译的语言助手。"


def log(*a):
    print(*a, flush=True)


def rows_of(path, n=None):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]
    return rows[:n] if n else rows


@torch.no_grad()
def tf_acc_hf(m, tok, ev):
    m.eval()
    hit = tot = 0
    for r in ev:
        ids, lb = build_hf(tok, r)
        x = torch.tensor([ids], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m(input_ids=x).logits[0].float()
        tgt = torch.tensor(lb[1:], device=DEV)
        keep = tgt != -100
        hit += int((lg[:-1].argmax(-1)[keep] == tgt[keep]).sum())
        tot += int(keep.sum())
    return hit / max(tot, 1)


def gen_acc(rows, answers):
    import re

    def norm(s):
        return re.sub(r"\s+", " ", s.strip().lower().replace("，", " ").replace(",", " "))
    hit = 0
    for r, a in zip(rows, answers):
        aa = a.strip().rstrip("。.!！")
        if r["task"] == "gloss":
            ok = r["a"] in a
        elif r["task"] == "s2c":
            ok = aa == r["a"].rstrip("。") or aa == r["a"]
        else:
            ok = norm(aa) == norm(r["a"])
        hit += int(ok)
    return hit / max(len(rows), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--replay", type=float, default=0.35)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--tag", default="qlora")
    ap.add_argument("--save-adapter", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from pretrain_l0 import TokenStream

    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                quantization_config=bnb, device_map={"": 0}).eval()
    ev = rows_of("data/sedu_holdout.jsonl", args.n_eval)
    tr = rows_of("data/sedu_train.jsonl")
    bare_ppl = val_ppl_hf(base)
    log(f"[{args.tag}] 4bit 载入，权重 {torch.cuda.memory_allocated()/2**30:.2f} GiB"
        f"，裸基座 ppl {bare_ppl:.2f}，裸 A_tf {tf_acc_hf(base, tok, ev):.3f}")

    m = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    m = get_peft_model(m, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.02,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    m.config.use_cache = False
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 256)
    rng = np.random.default_rng(0)
    pk = tok.pad_token_id or tok.eos_token_id

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for s in range(args.steps):
        if rng.random() < args.replay:
            x, _ = stream.batch(args.bs, train=True, rng=rng)
            ids = x.to(DEV)
            labels = ids.clone()
        else:
            pairs = [build_hf(tok, tr[int(rng.integers(len(tr)))]) for _ in range(args.bs)]
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
            log(f"  step {s+1}/{args.steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
    peak = torch.cuda.max_memory_allocated() / 2 ** 30

    m.eval()
    a_tf = tf_acc_hf(m, tok, ev)
    a_gen = gen_acc(ev, gen_strip(m, tok, ev))
    ppl = val_ppl_hf(m)
    log(f"[{args.tag}] 训练后 A_tf {a_tf:.3f} | A_gen {a_gen:.3f} | ppl {ppl:.2f} ({ppl/bare_ppl:.3f}×)"
        f" | 峰值显存 {peak:.2f} GiB | 总耗时 {time.time()-t0:.0f}s")
    ok = a_tf >= 0.90 and a_gen >= 0.80
    log(f"[判定] {'✅ QLoRA 可作常态训练路径' if ok else '⚠️ QLoRA 掉点，建议 fp16 训练 + 4bit 推理'}")
    if args.save_adapter:
        m.save_pretrained(os.path.join(ROOT, args.save_adapter))

    out = {"tag": args.tag, "protocol": {"steps": args.steps, "bs": args.bs, "lr": args.lr,
                                         "replay": args.replay, "n_eval": args.n_eval,
                                         "train_precision": "nf4 + lora fp16"},
           "bare_ppl": round(bare_ppl, 2),
           "after": {"A_tf": round(a_tf, 4), "A_gen": round(a_gen, 4), "ppl": round(ppl, 2)},
           "c_gate_regress": round(ppl / bare_ppl, 4),
           "peak_gib": round(peak, 2), "minutes": round((time.time() - t0) / 60, 1),
           "pass": bool(ok),
           "fp16_reference": {"A_tf": 0.9939, "A_gen": 0.96, "peak_gib": 16.8, "minutes": 23}}
    p = os.path.join(ART, f"{args.tag}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("→", p)


if __name__ == "__main__":
    main()
