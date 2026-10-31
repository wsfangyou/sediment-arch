# -*- coding: utf-8 -*-
"""Step 1 —— 合成技能“Sedu 语（密语）翻译”数据。

选这个技能的理由：
· 基座（Qwen2.5-3B-Instruct）必然没学过——词表由本脚本程序化发明，
  于是“骨干裸答 ≈ 随机”成为硬基线，保真门测的就是真实技能保持度。
· 任务完全可程序化验证（密→中 与 中→密 双向逐词映射），评分无主观性。
· 规则由固定 seed 决定，可复现；语法位（是/的/不/很/了/和）与全部词
  一起构成 92 项词典。

产出（均在 data/ 目录）：
  sedu_lexicon.json   Sedu 词 -> 中文 词典
  sedu_train.jsonl    4800 条训练样本（密→中 / 中→密 / 词义查询，三向轮换）
  sedu_holdout.jsonl  600 条验收样本（句子与训练集不重叠）
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)
rng = random.Random(20261029)

SYLL = ["ka", "lu", "mi", "to", "re", "sa", "ni", "bo", "de", "fu",
        "ga", "ho", "ji", "ko", "la", "mo", "nu", "pi", "qu", "ra"]
CN_NOUN = [
    "太阳", "月亮", "星星", "天空", "云朵", "雨水", "河流", "海洋", "山丘", "森林",
    "花朵", "果实", "种子", "叶子", "树木", "草地", "沙子", "石头", "火焰", "冰雪",
    "风", "雷", "电", "雾", "彩虹", "早晨", "黄昏", "夜晚", "白天", "春天",
    "夏天", "秋天", "冬天", "鸟", "鱼", "马", "牛", "羊", "猫", "狗",
    "人", "手", "脚", "眼睛", "耳朵", "嘴巴", "心脏", "头发", "面包", "牛奶",
]
CN_V = ["吃饭", "喝水", "睡觉", "走路", "跑步", "跳跃", "说话", "倾听", "观看", "思考", "学习", "工作"]
CN_ADJ = ["大", "小", "热", "冷", "快", "慢", "新", "旧", "好", "坏", "美丽", "明亮"]
assert len(CN_NOUN) == 50

# ---- 发明词：名词 2 音节，动词/形容词/语法词各分配唯一短码 ----
def make_words(n, forbid):
    out, seen = [], set()
    while len(out) < n:
        w = rng.choice(SYLL) + rng.choice(SYLL)
        if w not in forbid and w not in seen:
            seen.add(w); out.append(w)
    return out

nouns_sedu = make_words(50, set())
taken = set(nouns_sedu)
verbs_sedu = make_words(12, taken); taken |= set(verbs_sedu)
adjs_sedu = make_words(12, taken); taken |= set(adjs_sedu)
gram = {"是": "ve", "的": "ne", "不": "nu", "很": "xe", "了": "le", "和": "ke"}
assert not (taken & set(gram.values()))

sedu2cn = {w: c for w, c in zip(nouns_sedu, CN_NOUN)}
sedu2cn.update({w: c for w, c in zip(verbs_sedu, CN_V)})
sedu2cn.update({w: c for w, c in zip(adjs_sedu, CN_ADJ)})
sedu2cn.update({code: cn for cn, code in gram.items()})
cn2sedu = {v: k for k, v in sedu2cn.items()}

TEMPLATES = [  # (中文模板, Sedu模板, 槽位序列)
    ("{n1}和{n2}", "{n1} ke {n2}", ("n1", "n2")),
    ("{n1}是很{adj}的", "{n1} ve xe {adj} ne", ("n1", "adj")),
    ("{n1}不{v}了", "{n1} nu {v} le", ("n1", "v")),
    ("{n1}的{n2}很{adj}", "{n1} ne {n2} xe {adj}", ("n1", "n2", "adj")),
    ("{n1}{v}{n2}", "{n1} {v} {n2}", ("n1", "v", "n2")),
    ("{n1}和{n2}都不{v}", "{n1} ke {n2} ke nu {v}", ("n1", "n2", "v")),
]
POOLS = {"n1": CN_NOUN, "n2": CN_NOUN, "v": CN_V, "adj": CN_ADJ}

def gen_pair(i):
    tpl_cn, tpl_se, slots = TEMPLATES[i % len(TEMPLATES)]
    vals = {s: rng.choice(POOLS[s]) for s in slots}
    cn, se = tpl_cn, tpl_se
    for s in slots:
        cn = cn.replace("{" + s + "}", vals[s])
        se = se.replace("{" + s + "}", cn2sedu[vals[s]])
    return cn, se

def records(n, offset=0):
    out = []
    for j in range(n):
        i = offset + j
        cn, se = gen_pair(i)
        mode = i % 3
        if mode == 0:
            out.append({"id": f"r{i}", "task": "s2c", "q": f"把这句Sedu语翻译成中文：{se}", "a": cn})
        elif mode == 1:
            out.append({"id": f"r{i}", "task": "c2s", "q": f"把这句中文翻译成Sedu语：{cn}", "a": se})
        else:
            w = se.split()[0]
            out.append({"id": f"r{i}", "task": "gloss",
                        "q": f"Sedu语词“{w}”用中文怎么说？", "a": sedu2cn[w]})
    return out

train = records(4800)
holdout = records(600, offset=4800)

with open(os.path.join(DATA, "sedu_lexicon.json"), "w", encoding="utf-8") as f:
    json.dump(sedu2cn, f, ensure_ascii=False, indent=1)
for name, rs in (("sedu_train.jsonl", train), ("sedu_holdout.jsonl", holdout)):
    with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print("train", len(train), "holdout", len(holdout), "lexicon", len(sedu2cn))
print("holdout 任务分布:", dict(Counter(r["task"] for r in holdout)))
for r in train[:3]:
    print("例:", r["q"], "=>", r["a"])
