# -*- coding: utf-8 -*-
"""Step 4 —— 保真门评测 + 领域验收 + 通用回归（方案 E1d 的真实模型版）。

变体（同一评测管线）：
  bare        基座裸模型（域基线，应≈随机）
  teacher     教师技能（r=16，域上限参照）
  svd_r8/4/2  SVD 秩截断蒸馏产物
  kd_r8       行为重蒸学生（同数据低秩重训）

指标：
  领域：holdout 600 条三向任务（s2c 精确匹配 / c2s 规范化匹配 / gloss 包含匹配）
  回归：① 通用中文段落困惑度（技能不应伤害语言分布）
        ② 12 项域外指令的期望子串通过率（常识/算术/翻译/代码/指令跟随）

预注册验收线（跑前写在脚本里，杜绝事后挑数）：
  A. 域基线：bare ≤ 0.15（证明技能非骨干既有能力）
  B. 保真门：某蒸馏变体 ≥ 0.95 × teacher
  C. 回归线：蒸馏变体 ppl ≤ 1.02×teacher 且 域外通过 ≥ 0.9×bare
  D. 交付判定：过 B+C 中最紧凑者为“蒸馏好的”L1 常驻候选
"""
import os, re, sys, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct"
DEVICE = "cuda:0"
SYS = "你是精通Sedu语与中文互译的语言助手。"
MAX_NEW = 28

GEN段落 = ("人工智能技术在近年来取得了显著进展，尤其是在自然语言处理与计算机视觉领域。"
           "大规模预训练模型通过海量文本数据学习语言规律，能够在翻译、摘要、问答等任务中"
           "展现出接近人类水平的表现。然而，模型的部署仍然面临算力与内存的挑战，"
           "因此研究者提出了量化、蒸馏、稀疏化等多种压缩手段，以在质量与成本之间取得平衡。")

GEN_TASKS = [
    ("计算 17 加 26 的结果，只输出数字。", "43"),
    ("中国的首都是哪里？", "北京"),
    ("把“hello world”翻译成中文。", "世界"),
    ("水在标准大气压下多少摄氏度沸腾？", "100"),
    ("写一个 Python 函数返回列表最大值，写出 def 关键字。", "def"),
    ("“月落乌啼霜满天”的下一句是什么？", "江枫"),
    ("一年有多少个月？", "12"),
    ("光年是什么单位？回答包含“距离”或“长度”。", "距离"),
    ("请把下面句子改成否定：他喜欢咖啡。", "不"),
    ("“苹果”用英文怎么说？", "apple"),
    ("地球围绕什么天体公转？", "太阳"),
    ("简述 2×3 等于几，只输出数字。", "6"),
]

def load_model(variant):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    if variant != "bare":
        path = {"teacher": "teacher/adapter", "kd_r8": "distilled/adapter_kd_r8",
                "svd_r8": "distilled/adapter_svd_r8", "svd_r4": "distilled/adapter_svd_r4",
                "svd_r2": "distilled/adapter_svd_r2"}[variant]
        model = PeftModel.from_pretrained(model, os.path.join(ROOT, path))
        model = model.merge_and_unload()
    model.to(DEVICE); model.eval()
    return tok, model

def gen_batch(tok, model, prompts, bs=16):
    """批量 greedy 生成，返回与 prompts 等长的答案字符串列表。"""
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in prompts]
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", add_special_tokens=False, padding=True)
        ids = enc.input_ids.to(DEVICE); am = enc.attention_mask.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=am, max_new_tokens=MAX_NEW,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(ids.shape[0]):
            new = out[j][ids.shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True).strip())
    return outs

def gen_answer(tok, model, prompt):
    return gen_batch(tok, model, [prompt])[0]

def norm_sedu(s):
    return re.sub(r"\s+", " ", s.strip().lower().replace("，", " ").replace(",", " "))

def eval_domain(tok, model, rows):
    answers = gen_batch(tok, model, [r["q"] for r in rows], bs=16)
    hit = {"s2c": 0, "c2s": 0, "gloss": 0}
    tot = {"s2c": 0, "c2s": 0, "gloss": 0}
    for r, ans in zip(rows, answers):
        tot[r["task"]] += 1
        a = ans.strip().rstrip("。.!！")
        if r["task"] == "gloss":
            if r["a"] in ans: hit[r["task"]] += 1
        elif r["task"] == "s2c":
            if a == r["a"].rstrip("。") or a == r["a"]: hit[r["task"]] += 1
        else:
            if norm_sedu(a) == norm_sedu(r["a"]): hit[r["task"]] += 1
    acc = {k: hit[k] / max(tot[k], 1) for k in hit}
    overall = sum(hit.values()) / max(sum(tot.values()), 1)
    return acc, overall

def eval_ppl(tok, model, text):
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=DEVICE)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    return float(torch.exp(torch.tensor(loss)))

def eval_general(tok, model):
    hits = 0
    for q, expect in GEN_TASKS:
        ans = gen_answer(tok, model, q)
        if expect.lower() in ans.lower():
            hits += 1
    return hits / len(GEN_TASKS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="bare,teacher,svd_r8,svd_r4,svd_r2,kd_r8")
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "sedu_holdout.jsonl"), encoding="utf-8")]
    res = {}
    for v in args.variants.split(","):
        tok, model = load_model(v)
        acc, overall = eval_domain(tok, model, rows)
        ppl = eval_ppl(tok, model, GEN段落)
        gen = eval_general(tok, model) if v != "kd_r8" or True else None
        res[v] = {"domain": {k: round(x, 4) for k, x in acc.items()},
                  "domain_overall": round(overall, 4), "ppl_general": round(ppl, 4),
                  "general_pass": round(gen, 4)}
        print(v, json.dumps(res[v], ensure_ascii=False), flush=True)
        del model; torch.cuda.empty_cache()
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---- 预注册验收判定 ----
    if "bare" in res and "teacher" in res:
        t = res["teacher"]["domain_overall"]
        checks = {"A 域基线(bare≤0.15)": res["bare"]["domain_overall"] <= 0.15,
                  "教师显著高于裸基线": t > 3 * max(res["bare"]["domain_overall"], 0.05)}
        for v in [x for x in res if x not in ("bare", "teacher")]:
            d = res[v]["domain_overall"]
            checks[f"B {v} 保真≥95%×teacher"] = d >= 0.95 * t
            checks[f"C {v} 回归(ppl≤1.02×teacher)"] = res[v]["ppl_general"] <= 1.02 * res["teacher"]["ppl_general"]
            checks[f"C {v} 域外≥0.9×bare"] = res[v]["general_pass"] >= 0.9 * res["bare"]["general_pass"]
        print(json.dumps({"verdicts": {k: bool(x) for k, x in checks.items()}}, ensure_ascii=False, indent=2))
        with open(os.path.join(ROOT, "eval", "verdicts.json"), "w", encoding="utf-8") as f:
            json.dump({k: bool(x) for k, x in checks.items()}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
