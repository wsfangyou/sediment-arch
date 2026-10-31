# -*- coding: utf-8 -*-
"""A_gen 的失败类型拆分 —— 决定"约束解码"值不值得做。

对 adapter_sedu_r16（600 步纯技能）的 30 条 holdout 生成结果分类：
  PASS   精确命中
  EOS    生成即 EOS（空串）
  LOOP   出现换行/同一片段重复 ≥3 次（格式类错误）
  SUBST  格式正常但选错词/句子（语义类错误）
并给出「若在第一个换行处截断」的准确率 —— 这是格式约束类修复的**上界代理**。
"""
import os, sys, json, re
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, DEFAULT_L0, register
from accept_l0_d2 import build, gen_rows, gen_acc, SYS

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEV = "cuda:0"
tok = load_tokenizer(DEFAULT_L0)
hold = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"), encoding="utf-8")][:30]

register()
base, _ = load_l0(dtype="float16", device=DEV)
from peft import PeftModel
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "adapter_sedu_r16"
TAG = sys.argv[2] if len(sys.argv) > 2 else "e3_errorsplit"
m = PeftModel.from_pretrained(base, os.path.join(ROOT, "l0_acceptance", ADAPTER))
m = m.merge_and_unload(); m.eval()

gs = gen_rows(m, tok, hold, max_new=28)

def is_loop(s):
    if "\n" in s.strip():
        return True
    # 同 4-gram 重复 3 次
    for n in (2, 4, 6):
        t = [s[i:i+n] for i in range(0, max(1, len(s)-n), n)]
        if len(t) >= 3 and len(set(t[-3:])) == 1 and len(t[-1]) == n:
            return True
    return False

rows, cnt = [], {"PASS": 0, "EOS": 0, "LOOP": 0, "SUBST": 0}
for r, g in zip(hold, gs):
    exact = (r["a"] in g) if r["task"] == "gloss" else (g.strip().rstrip("。") == r["a"])
    if exact: k = "PASS"
    elif g.strip() == "": k = "EOS"
    elif is_loop(g): k = "LOOP"
    else: k = "SUBST"
    first_line = g.split("\n")[0].strip().rstrip("。")
    cut_ok = (r["a"] in first_line) if r["task"] == "gloss" else (first_line == r["a"])
    cnt[k] += 1
    rows.append({"q": r["q"], "expect": r["a"], "gen": g[:60], "type": k, "cut_first_line_ok": bool(cut_ok)})

n = len(rows)
print(f"[分类] 共 {n} 条")
for k, v in cnt.items():
    print(f"  {k:6s} {v:3d}  {v/n:.1%}")
acc_raw = gen_acc(hold, gs)
acc_cut = sum(r["cut_first_line_ok"] for r in rows) / n
print(f"\nA_gen（原始，纯 greedy）      = {acc_raw:.3f}")
print(f"A_gen（只在第一个换行处截断） = {acc_cut:.3f}  ← 格式约束类修复的上界代理")
print(f"格式类错误占比 = {(cnt['LOOP']+cnt['EOS'])/n:.1%}，语义类（选错词）= {cnt['SUBST']/n:.1%}")
print("\n样例（每类最多 2 条）:")
seen = {}
for r in rows:
    if seen.get(r["type"], 0) >= 2: continue
    seen[r["type"]] = seen.get(r["type"], 0) + 1
    print(f"  [{r['type']}] {r['q']} | 期望 {r['expect']!r} | 得到 {r['gen']!r}")

out = {"n": n, "counts": cnt, "A_gen_raw": round(acc_raw, 4),
       "A_gen_if_cut_first_line": round(acc_cut, 4),
       "format_error_share": round((cnt["LOOP"]+cnt["EOS"])/n, 4),
       "semantic_error_share": round(cnt["SUBST"]/n, 4),
       "adapter": ADAPTER,
       "samples": rows}
p = os.path.join(ROOT, "l0_acceptance", f"{TAG}.json")
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("→", p)
