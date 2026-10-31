# -*- coding: utf-8 -*-
"""补两个口径：
  cot   四步题放开到 400 token（上一轮 96 token 是截断，不是答错）
  quick 明确要求"只回字母、禁止解释"，看没有显式推理时掉多少
"""
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_reasoning_probe import make_items, LETTERS, DEV, ART

MODEL = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B-Instruct-2507"
SYS_COT = "你在做单项选择题。请一步步算清楚，最后一行写「答案是：X」。"
SYS_QUICK = "你在做单项选择题。只输出一个字母（A/B/C/D），不要写任何过程。"


def run(m, tok, items, sysmsg, max_new):
    hit = 0
    nofmt = 0
    detail = []
    for it in items:
        body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["opts"]))
        user = f"{it['q']}\n{body}"
        try:
            text = tok.apply_chat_template([{"role": "system", "content": sysmsg},
                                            {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template([{"role": "system", "content": sysmsg},
                                            {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        with torch.no_grad():
            o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id,
                           eos_token_id=[151645, 151643])
        out = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
        mm = re.search(r"(?:答案|answer)[^A-Da-d]{0,6}([ABCD])", out)
        if mm:
            pred = LETTERS.index(mm.group(1))
        else:
            c = re.findall(r"\b([ABCD])\b", out) or re.findall(r"[ABCD]", out)
            pred = LETTERS.index(c[-1]) if c else -1
        if pred < 0:
            nofmt += 1
        hit += int(pred == it["ans"])
        detail.append({"cat": it["cat"], "expect": LETTERS[it["ans"]],
                       "pred": LETTERS[pred] if pred >= 0 else "NONE", "out": out[-90:]})
    return round(hit / len(items), 4), round(nofmt / len(items), 4), detail


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    items = make_items(per_cat=20)
    hard = [x for x in items if x["cat"] == "arith4"][:12]
    easy = [x for x in items if x["cat"] == "arith2"][:12]

    res = {}
    acc, nofmt, det = run(m, tok, hard, SYS_COT, 400)
    res["arith4_cot_400tok"] = {"acc": acc, "no_format": nofmt,
                                "samples": [d["out"][-60:] for d in det[:2]]}
    print(f"[四步·CoT·400tok] {acc:.3f} 格式失败 {nofmt:.2f}", flush=True)

    acc2, nofmt2, _ = run(m, tok, hard, SYS_QUICK, 8)
    res["arith4_quick"] = {"acc": acc2, "no_format": nofmt2}
    print(f"[四步·直接答] {acc2:.3f} 格式失败 {nofmt2:.2f}", flush=True)

    acc3, nofmt3, _ = run(m, tok, easy, SYS_QUICK, 8)
    res["arith2_quick"] = {"acc": acc3, "no_format": nofmt3}
    print(f"[两步·直接答] {acc3:.3f} 格式失败 {nofmt3:.2f}", flush=True)

    res["protocol"] = {"model": MODEL, "n_hard": len(hard), "n_easy": len(easy),
                       "note": "对比有无显式推理（CoT）的差距；400 token 是为避免长度截断"}
    p = os.path.join(ART, "reasoning_probe_extra.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
