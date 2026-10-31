# -*- coding: utf-8 -*-
"""为 E1 多技能闭环生成两个额外技能（Nuri / Tavi），格式与 Sedu 完全一致。

为什么能复用判分：判分器 `gen_acc` 只认 task ∈ {gloss, s2c, c2s} 与 q/a 字符串，
所以只要保持同样的三种问句模板 + 同样可程序化比对的答案，就能用同一套 A_tf / A_gen 口径。

每个技能一个自造语：80 个词 + 3 个虚词（和/的/是）+ 三种句法。
技能之间**词表、音系、虚词全不相同**，所以"互相干扰"是可测的。
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")

ZH_NOUN = ["马", "叶子", "石头", "河流", "月亮", "火", "水", "鸟", "鱼", "山", "云", "雪",
           "花", "树", "门", "窗", "书", "笔", "茶", "米", "盐", "布", "船", "车", "路",
           "桥", "塔", "钟", "灯", "刀", "绳", "网", "瓶", "碗", "席", "鼓", "笛", "镜",
           "伞", "鞋", "帽", "衣", "线", "针", "纸", "墨", "砚", "算盘", "炉", "井"]
ZH_ADJ = ["红", "蓝", "白", "黑", "大", "小", "冷", "热", "轻", "重", "甜", "苦",
          "快", "慢", "亮", "暗", "软", "硬", "新", "旧", "深", "浅", "圆", "方",
          "香", "臭", "干", "湿", "宽", "窄"]

PHONO = {
    "nuri": {"ons": ["m", "n", "l", "r", "t", "k", "s", "p"], "vow": ["a", "e", "i", "o", "u"],
             "cod": ["", "n", "l"], "name": "Nuri", "part": ["me", "ti", "so"]},
    "tavi": {"ons": ["v", "z", "h", "d", "g", "b", "f", "j"], "vow": ["a", "ae", "ei", "ou", "y"],
             "cod": ["", "k", "t"], "name": "Tavi", "part": ["gwa", "dzo", "phy"]},
}


def make_lexicon(skill, n_words, rng):
    p = PHONO[skill]
    used, words = set(), []
    while len(words) < n_words:
        w = (rng.choice(p["ons"]) + rng.choice(p["vow"]) + rng.choice(p["cod"]))
        if len(w) < 2 or w in used:
            continue
        used.add(w)
        words.append(w)
    return words


def build(skill, n_train, n_hold, seed):
    rng = random.Random(seed)
    p = PHONO[skill]
    lex = make_lexicon(skill, len(ZH_NOUN) + len(ZH_ADJ), rng)
    zh2cw = dict(zip(ZH_NOUN + ZH_ADJ, lex))
    nouns = ZH_NOUN
    adjs = ZH_ADJ
    p_and, p_poss, p_is = p["part"]

    def sent(kind):
        if kind == 0:
            a, b = rng.choice(nouns), rng.choice(nouns)
            return f"{a}和{b}", f"{zh2cw[a]} {p_and} {zh2cw[b]}"
        if kind == 1:
            a, b = rng.choice(nouns), rng.choice(nouns)
            return f"{a}的{b}", f"{zh2cw[a]} {p_poss} {zh2cw[b]}"
        a, x = rng.choice(nouns), rng.choice(adjs)
        return f"{a}是{x}的", f"{zh2cw[a]} {p_is} {zh2cw[x]}"

    def one(i):
        t = rng.random()
        if t < 0.2:                                        # gloss
            w = rng.choice(ZH_NOUN + ZH_ADJ)
            return {"id": f"{skill}{i}", "task": "gloss",
                    "q": f"{p['name']}语词“{zh2cw[w]}”用中文怎么说？", "a": w}
        kind = rng.randrange(3)
        zh, cw = sent(kind)
        if t < 0.6:                                        # s2c
            return {"id": f"{skill}{i}", "task": "s2c",
                    "q": f"把这句{p['name']}语翻译成中文：{cw}", "a": zh}
        return {"id": f"{skill}{i}", "task": "c2s",         # c2s
                "q": f"把这句中文翻译成{p['name']}语：{zh}", "a": cw}

    tr = [one(i) for i in range(n_train)]
    ho = [one(i) for i in range(n_hold)]
    return tr, ho


def main():
    os.makedirs(DATA, exist_ok=True)
    for skill, seed in (("nuri", 11), ("tavi", 23)):
        tr, ho = build(skill, 2400, 300, seed)
        for name, rows in ((f"{skill}_train.jsonl", tr), (f"{skill}_holdout.jsonl", ho)):
            with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{skill}] 训练 {len(tr)} / holdout {len(ho)}  → data/{skill}_*.jsonl")
        print("   样例:", json.dumps(tr[:2], ensure_ascii=False))


if __name__ == "__main__":
    main()
