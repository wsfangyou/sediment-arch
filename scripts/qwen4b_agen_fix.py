# -*- coding: utf-8 -*-
"""D2 的 A_gen 复评：Qwen3 系列是「思考型」模型，chat template 会在答案前插入  thinking 块。

直接做整串 exact match 会把"答对了但在前面写了思考块"判成错——这与项目记录里的 B4
（generate 不认 <|im_end|>）是同一类**判分口径问题**，不是能力问题。

本脚本对同一个 adapter 用三种口径复评 A_gen：
  raw          原始输出做匹配（= 上一轮 0.320 的口径）
  strip        剥掉  thinking…<｜end▁of▁thinking｜> 后再匹配
  no_think     用 enable_thinking=False 的提示词生成
"""
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accept_l0_d2 import gen_acc, SYS
from loader import load_tokenizer

DEV = "cuda:0"
MODEL = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B"
ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "l0_acceptance", "adapter_qwen4b_sedu")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

THINK = re.compile(r"<think>.*?</think>", re.S)


def gen(rs, tok, m, thinking):
    outs = []
    for r in rs:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["q"]}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        with torch.no_grad():
            o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=48,
                           do_sample=False, pad_token_id=tok.pad_token_id,
                           repetition_penalty=1.0, eos_token_id=[151645, 151643])
        outs.append(tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    m = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload().eval()

    ev = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"), encoding="utf-8")][:50]

    a_raw = gen(ev, tok, m, True)
    a_strip = [THINK.sub("", g).strip() for g in a_raw]
    a_nothink = gen(ev, tok, m, False)

    res = {
        "raw": round(gen_acc(ev, a_raw), 4),
        "strip_think": round(gen_acc(ev, a_strip), 4),
        "no_think_prompt": round(gen_acc(ev, a_nothink), 4),
    }
    print("[A_gen 三口径]", res, flush=True)
    print(f"  平均输出长度: raw {sum(map(len,a_raw))/len(a_raw):.1f} → strip {sum(map(len,a_strip))/len(a_strip):.1f} "
          f"→ no_think {sum(map(len,a_nothink))/len(a_nothink):.1f} 字符", flush=True)
    for r, g1, g2, g3 in list(zip(ev, a_raw, a_strip, a_nothink))[:6]:
        print(f"  期望 {r['a']!r}\n    raw={g1[:60]!r}\n    strip={g2[:40]!r} | no_think={g3[:40]!r}",
              flush=True)

    p = os.path.join(ROOT, "l0_acceptance", "qwen4b_d2_agen_fix.json")
    json.dump({"adapter": ADAPTER, "results": res,
               "note": "同一 adapter、同一 50 条 holdout，只改判分/提示口径"},
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
