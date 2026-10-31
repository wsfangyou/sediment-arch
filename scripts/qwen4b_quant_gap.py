# -*- coding: utf-8 -*-
"""补最后一块：把 fp16 上训出来的技能 adapter 挂到 **NF4 4bit 底座**上，D2 掉多少？

背景：D2 的 A_tf 0.994 / A_gen 0.960 是在 fp16 权重上量的，但 6GB 档部署必须 4bit。
两者之间隔着一次量化，这是 B 线（Qwen3-4B 当 L0）唯一没验的落差。

三个口径一起报：
  fp16+adapter   ← 已测（0.994 / 0.960 / ppl 14.42）
  nf4+adapter    ← 本脚本
  nf4 裸基座     ← 本脚本（看技能是否还在、通用 ppl 退化多少）
"""
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accept_l0_d2 import gen_acc, tf_acc, SYS
from loader import load_tokenizer
from pretrain_l0 import TokenStream

DEV = "cuda:0"
MODEL = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ADAPTER = os.path.join(ROOT, "l0_acceptance", "adapter_qwen4b_sedu")
THINK = re.compile(r"<think>.*?</think>", re.S)


def gen_strip(m, tok, rs, max_new=48):
    outs = []
    for r in rs:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["q"]}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        with torch.no_grad():
            o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id,
                           repetition_penalty=1.0, eos_token_id=[151645, 151643])
        outs.append(THINK.sub("", tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)).strip())
    return outs


@torch.no_grad()
def val_ppl(m, n=24):
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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                quantization_config=bnb, device_map={"": 0}).eval()
    ev = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"),
                                      encoding="utf-8")][:50]

    out = {}
    out["nf4_bare"] = {"A_tf": round(tf_acc(base, tok, ev), 4),
                       "A_gen": round(gen_acc(ev, gen_strip(base, tok, ev)), 4),
                       "ppl": round(val_ppl(base), 2)}
    print("[nf4 裸基座]", out["nf4_bare"], flush=True)

    m = PeftModel.from_pretrained(base, ADAPTER).eval()
    out["nf4_adapter"] = {"A_tf": round(tf_acc(m, tok, ev), 4),
                          "A_gen": round(gen_acc(ev, gen_strip(m, tok, ev)), 4),
                          "ppl": round(val_ppl(m), 2)}
    print("[nf4 + adapter]", out["nf4_adapter"], flush=True)

    out["fp16_adapter_ref"] = {"A_tf": 0.9939, "A_gen": 0.96, "ppl": 14.42,
                               "source": "qwen4b_d2.json / qwen4b_d2_agen_fix.json"}
    o = out["nf4_adapter"]
    ref = out["fp16_adapter_ref"]
    out["gap"] = {"A_tf": round(o["A_tf"] - ref["A_tf"], 4),
                  "A_gen": round(o["A_gen"] - ref["A_gen"], 4),
                  "ppl_x": round(o["ppl"] / ref["ppl"], 3)}
    print(f"[落差] A_tf {out['gap']['A_tf']:+.4f} | A_gen {out['gap']['A_gen']:+.4f} "
          f"| ppl {out['gap']['ppl_x']:.3f}×", flush=True)
    out["verdict"] = ("✅ 量化后 D2 仍过线（A_tf>=0.90 且 A_gen>=0.80）"
                      if o["A_tf"] >= 0.90 and o["A_gen"] >= 0.80 else
                      "⚠️ 量化后掉出预注册线，需要 QLoRA 训练或更高位宽")
    print("[判定]", out["verdict"], flush=True)
    p = os.path.join(ROOT, "l0_acceptance", "qwen4b_quant_gap.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
