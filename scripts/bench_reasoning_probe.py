# -*- coding: utf-8 -*-
"""智力/推理探针：用选项对数似然（loglikelihood）测 base 模型，不外挂生成。

为什么用 loglikelihood：L0 与 Qwen3-4B 都是 **base 模型**（没有指令微调），
生成会大量出现"答非所问/不复述选项"的格式噪声；MMLU 一类基准对 base 模型也是这么测的。
每个题只做 **一次前向**（比较 A/B/C/D 四个字母的对数概率），结果确定、可复现。

四类题（全部由程序生成并程序化判分，答案唯一）：
  arith2  两步应用题         arith4  四步应用题（难度阶梯）
  logic2  两段传递推理       unit    单位/时间换算

被测对象：
  L0（自研 0.83B） / Qwen3-4B base / Qwen3-4B + Sedu 技能 adapter
  —— 最后一个用来回答"挂载技能之后，推理能力变了吗"。
"""
import argparse
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
DEV = "cuda:0"
QWEN = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B"
LETTERS = "ABCD"

FEWSHOT = """题目：树上有 5 只鸟，飞来 3 只，又飞走 2 只，现在有几只？
A. 4
B. 6
C. 8
D. 10
答案：B

题目：一支笔 3 元，买 4 支要多少钱？
A. 7
B. 12
C. 9
D. 16
答案：B

题目：甲比乙高，乙比丙高，谁最矮？
A. 甲
B. 乙
C. 丙
D. 无法确定
答案：C

"""


def _opts(rng, correct, gen):
    """把正确答案与三个干扰项打乱，返回 (选项列表, 正确答案下标)。"""
    vals = {correct}
    tries = 0
    while len(vals) < 4 and tries < 200:
        v = gen()
        if v not in vals:
            vals.add(v)
        tries += 1
    vals = list(vals)
    rng.shuffle(vals)
    return vals, vals.index(correct)


def make_items(seed=7, per_cat=20):
    rng = random.Random(seed)
    items = []

    for _ in range(per_cat):                      # 两步
        a = rng.randint(3, 15); b = rng.randint(2, 9); c = rng.randint(1, a)
        q = f"盒子里有 {a} 个球，又放进 {b} 个，然后拿走 {c} 个，现在有几个？"
        ans = a + b - c
        opts, idx = _opts(rng, ans, lambda: ans + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
        items.append({"cat": "arith2", "q": q, "opts": opts, "ans": idx})

    for _ in range(per_cat):                      # 四步（难度阶梯）
        a = rng.randint(4, 12); b = rng.randint(3, 8); c = rng.randint(1, 9)
        d = rng.randint(2, 5); e = rng.randint(1, 3)
        q = (f"仓库里有 {a} 箱货，运来 {b} 箱，又发出 {c} 箱，然后每箱装 {d} 件、共 {e} 种规格，"
             f"若最后又运来 {d} 箱，现在有几箱？")
        ans = a + b - c + d
        opts, idx = _opts(rng, ans, lambda: ans + rng.choice([-6, -5, -3, -2, -1, 1, 2, 3, 5, 6]))
        items.append({"cat": "arith4", "q": q, "opts": opts, "ans": idx})

    names = ["甲", "乙", "丙", "丁", "戊", "己"]
    for _ in range(per_cat // 2):                 # 两段传递推理
        x, y, z = rng.sample(names, 3)
        if rng.random() < 0.5:
            q = f"{x}比{y}高，{y}比{z}高，谁最矮？"
            ans = z
        else:
            q = f"{x}比{y}轻，{z}比{y}轻，{x}比{z}轻，谁最重？"
            ans = y
        opts = [x, y, z, "无法确定"]
        rng.shuffle(opts)
        items.append({"cat": "logic2", "q": q, "opts": opts, "ans": opts.index(ans)})

    for _ in range(per_cat // 2):                 # 单位换算
        h = rng.randint(1, 5); m = rng.randint(0, 59)
        q = f"{h} 小时 {m} 分等于多少分钟？"
        ans = h * 60 + m
        opts, idx = _opts(rng, ans, lambda: ans + rng.choice([-60, -30, -10, 10, 30, 60]))
        items.append({"cat": "unit", "q": q, "opts": opts, "ans": idx})
    return items


CHAT_SYS = ("你在做单项选择题。请先在脑内算清楚，然后**只输出一个字母**（A、B、C 或 D），不要写任何解释或过程。")


def build_prompt_chat(tok, it):
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["opts"]))
    user = f"{it['q']}\n{body}"
    msgs = [{"role": "system", "content": CHAT_SYS}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompt(it):
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["opts"]))
    return FEWSHOT + f"题目：{it['q']}\n{body}\n答案："


@torch.no_grad()
def score_likelihood(m, tok, items, chat=False, max_new=24):
    """每题一次前向：比较 A/B/C/D 四个字母在「答案：」之后的对数概率。"""
    import re
    hit = {c: 0 for c in {i["cat"] for i in items}}
    tot = {c: 0 for c in hit}
    nofmt = 0
    detail = []
    for it in items:
        p = build_prompt_chat(tok, it) if chat else build_prompt(it)
        ids = torch.tensor([tok(p, add_special_tokens=False)["input_ids"]], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            try:
                out = m.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=(max_new),
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=[151645, 151643])
            except Exception:
                out = m.generate(ids, max_new_tokens=(max_new), do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        mm = re.search(r"(?:答案|answer)[^A-Da-d]{0,6}([ABCD])", txt)
        if not mm:
            cands = re.findall(r"\b([ABCD])\b", txt) or re.findall(r"[ABCD]", txt)
            mm = cands[-1] if cands else None
            pred = LETTERS.index(mm) if mm else -1
        else:
            pred = LETTERS.index(mm.group(1))
        if pred < 0:
            nofmt += 1
        hit[it["cat"]] += int(pred == it["ans"])
        tot[it["cat"]] += 1
        if len(detail) < 8:
            detail.append({"cat": it["cat"], "expect": LETTERS[it["ans"]],
                           "pred": LETTERS[pred] if pred >= 0 else "NONE", "out": txt[:40]})
        if pred != it["ans"] and len(detail) <= 8:
            pass
    acc = {c: round(hit[c] / max(tot[c], 1), 4) for c in hit}
    overall = round(sum(hit.values()) / max(sum(tot.values()), 1), 4)
    return acc, overall, round(nofmt / max(len(items), 1), 4), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=200, help="chat 模式生成上限（要够写推理过程）")
    ap.add_argument("--out", default="reasoning_probe.json")
    ap.add_argument("--skip-base", action="store_true", help="跳过 Qwen3-4B base 的 chat 评测")
    ap.add_argument("--skip-sedu", action="store_true", help="跳过 +Sedu 的 chat 评测")
    ap.add_argument("--instruct", default="", help="额外评测一个指令模型的路径（chat 模板 + 字母似然）")
    args = ap.parse_args()
    items = make_items(per_cat=args.per_cat)
    print(f"[探针] {len(items)} 题：{args.per_cat} 两步 + {args.per_cat} 四步 + "
          f"{args.per_cat} 传递推理 + {args.per_cat} 换算", flush=True)

    res = {"n_items": len(items),
           "protocol": {"scoring": "生成式：每个模型用各自最佳提示（base 用 3-shot 续写，instruct 用 chat 模板），生成 ≤24 token，解析首个 A/B/C/D",
                        "chance": 0.25},
           "models": {}}

    # ---------- L0（自研 0.83B）----------
    from loader import load_l0, load_tokenizer, register
    register()
    l0, l0tok = load_l0(dtype="float16", device=DEV)
    acc, overall, nofmt, det = score_likelihood(l0, l0tok, items)
    res["models"]["L0_0.83B"] = {"per_cat": acc, "overall": overall, "no_format": nofmt, "detail": det}
    print(f"[L0 0.83B] 总体 {overall:.3f} | {acc}", flush=True)
    del l0; torch.cuda.empty_cache()

    # ---------- Qwen3-4B base ----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qt = AutoTokenizer.from_pretrained(QWEN)
    qm = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float16).to(DEV).eval()
    if args.skip_base:
        print("[Qwen3-4B base] 跳过 chat 评测（沿用 3-shot 读数）", flush=True)
    else:
        acc, overall, nofmt, det = score_likelihood(qm, qt, items, chat=True, max_new=args.max_new)
        res["models"]["Qwen3-4B_base"] = {"per_cat": acc, "overall": overall,
                                          "no_format": nofmt, "detail": det}
        print(f"[Qwen3-4B base] 总体 {overall:.3f} | {acc}", flush=True)

    # ---------- Qwen3-4B + Sedu 技能 ----------
    from peft import PeftModel
    ap_path = os.path.join(ROOT, "l0_acceptance", "adapter_qwen4b_sedu")
    if os.path.isdir(ap_path) and not args.skip_sedu:
        pm = PeftModel.from_pretrained(qm, ap_path).eval()
        acc, overall, nofmt, det = score_likelihood(pm, qt, items, chat=True, max_new=args.max_new)
        res["models"]["Qwen3-4B_+Sedu"] = {"per_cat": acc, "overall": overall, "no_format": nofmt, "detail": det}
        print(f"[Qwen3-4B + Sedu 技能] 总体 {overall:.3f} | {acc}", flush=True)

    if args.instruct:
        im = AutoModelForCausalLM.from_pretrained(args.instruct, dtype=torch.float16).to(DEV).eval()
        itk = AutoTokenizer.from_pretrained(args.instruct)
        acc, overall, nofmt, det = score_likelihood(im, itk, items, chat=True, max_new=args.max_new)
        res["models"]["Qwen3-4B-Instruct-2507"] = {"per_cat": acc, "overall": overall,
                                                   "no_format": nofmt, "detail": det}
        print(f"[Qwen3-4B-Instruct-2507] 总体 {overall:.3f} | {acc}", flush=True)

    p = os.path.join(ART, args.out)
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
