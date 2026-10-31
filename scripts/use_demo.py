# -*- coding: utf-8 -*-
"""使用示例冒烟测试：基座 + 蒸馏常驻候选 svd_r2，现场验证挂载与推理。"""
import json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct"
ADAPTER = r"D:\Qwen-works\Qwen-work-1\蒸馏实验\distilled\adapter_svd_r2"
SYS = "你是精通Sedu语与中文互译的语言助手。"   # 与技能训练一致（建议保留）

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(model, ADAPTER)   # 挂载，不重写基座
model.eval()

def ask(q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

lex = json.load(open(r"D:\Qwen-works\Qwen-work-1\蒸馏实验\data\sedu_lexicon.json", encoding="utf-8"))
inv = {v: k for k, v in lex.items()}          # 中文词 -> Sedu 词
nouns = [k for k in lex if k not in {"ve", "ne", "nu", "xe", "le", "ke"}]
r = random.Random(1)
w1, w2, adj = r.sample(nouns, 3)
sedu1 = f"{w1} {inv.get('和','ke')} {w2}"                    # “X和Y”
cn1 = f"{lex[w1]}和{lex[w2]}"
sedu2 = f"{w1} ve xe {inv['大']} ne"                          # “X是很大的”（用词典码）
tests = [
    f"把这句Sedu语翻译成中文：{sedu1}",
    f"把这句中文翻译成Sedu语：{lex[w1]}是很{lex[inv['大']]}的",
    f"Sedu语词“{w2}”用中文怎么说？",
]
for q in tests:
    print("问:", q)
    print("答:", ask(q))
    print()
print("GPU 占用:", round(torch.cuda.memory_allocated() / 2**30, 2), "GiB")
