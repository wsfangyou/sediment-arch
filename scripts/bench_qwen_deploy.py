# -*- coding: utf-8 -*-
"""Qwen3-4B 的「部署侧」账：D1 的显存预算 + D4 的解码速度。

D1 的 6GB 档要求 L0 权重 + KV@4096 + 工作区 ≤ 6GB，并给 L1 热沉积带留 0.8–1.2GB。
本脚本实测 fp16 与 NF4 两种部署形态的真实占用，以及带 KV cache 的解码速度（D4）。
"""
import json
import os
import sys
import time

import torch

DEV = "cuda:0"
MODEL = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B"
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "l0_acceptance")


def gb(x):
    return round(x / 2 ** 30, 3)


def measure(tag, quant=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 quantization_config=bnb, device_map={"": 0})
    else:
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV)
    m.eval()
    load_s = time.time() - t0
    w = gb(torch.cuda.memory_allocated())
    print(f"[{tag}] 加载 {load_s:.1f}s，权重常驻 {w} GiB", flush=True)

    # ---- 4096 上下文前向（峰值 = 权重 + KV + logits + 工作区）----
    ids = torch.randint(0, 151936, (1, 4096), device=DEV)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        m(input_ids=ids)
    peak4k = gb(torch.cuda.max_memory_allocated())
    kv_per_tok = 2 * 36 * 8 * 128 * (1 if quant == "nf4" else 2)   # 量化下 KV 仍是 fp16 → 统一 2
    kv_per_tok = 2 * 36 * 8 * 128 * 2
    print(f"[{tag}] @ctx4096 峰值 {peak4k} GiB（其中 KV 理论上 {kv_per_tok*4096/2**30:.3f} GiB）", flush=True)
    del ids
    torch.cuda.empty_cache()

    # ---- 解码速度（D4）：带 KV cache 逐 token ----
    p = tok("请用一句话解释什么是沉积架构。", return_tensors="pt").input_ids.to(DEV)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        o = m.generate(p, attention_mask=torch.ones_like(p), max_new_tokens=32,
                       do_sample=False, pad_token_id=tok.pad_token_id,
                       eos_token_id=[151645, 151643])
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_new = o.shape[1] - p.shape[1]
    print(f"[{tag}] 解码 {n_new} tok / {dt:.2f}s = {n_new/dt:.1f} tok/s "
          f"({dt/n_new*1000:.0f} ms/token)", flush=True)
    out = {"tag": tag, "load_s": round(load_s, 1), "weights_gib": w, "peak_ctx4096_gib": peak4k,
           "decode_tok_per_s": round(n_new / dt, 1), "decode_ms_per_tok": round(dt / n_new * 1000, 1)}
    del m
    torch.cuda.empty_cache()
    return out


def main():
    res = {}
    res["fp16"] = measure("fp16")
    res["nf4"] = measure("nf4", quant="nf4")
    # 6GB 档核算：权重 + KV@4096 + 工作区（用实测峰值减去权重里的 logits 部分近似）
    for k, v in res.items():
        budget = 6.0 - v["peak_ctx4096_gib"]
        v["l1_headroom_in_6gb"] = round(budget, 2)
        print(f"[6GB 档] {k}: 6.0 − {v['peak_ctx4096_gib']} = 余 {budget:.2f} GB 给 L1（要求 0.8–1.2）",
              flush=True)
    p = os.path.join(ART, "qwen4b_deploy.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
