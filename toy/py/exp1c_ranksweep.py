# -*- coding: utf-8 -*-
"""实验 1c —— 热沉积的"秩-预算-保真"相边界（复用 v2 技能缓存，纯 SVD+评测）。
exp1 实测：rank8 截断保真 0.75–0.85 < 门 0.95（被正确拒绝）。
本消融扫 rank∈{6,8,10,12,14,16}：r=16 是 LoRA 原秩（数学上无损）→ 门应放行。
输出部署决策表：L1 预算容得下完整产物→无损常驻；
预算不足→截断不过门→自动退回按需加载形态。"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import next_token_acc, make_skill_chains, svd_truncate, DEV, ART_DIR
from skills import get_base_model, ensure_lora_deltas, Merger

deltas = ensure_lora_deltas()
model = get_base_model().to(DEV)
merger = Merger(model)
chains = make_skill_chains(8)
active = [2, 3, 4, 5]

acc_full = {}
for i in active:
    merger.merge([deltas[i]])
    acc_full[i] = next_token_acc(model, chains[i], rng=np.random.default_rng(7))
merger.restore()

table = []
for rank in [6, 8, 10, 12, 14, 16]:
    keeps = []
    for i in active:
        d_r, _ = svd_truncate(deltas[i].to(DEV), rank)
        merger.merge([d_r])
        a = next_token_acc(model, chains[i], rng=np.random.default_rng(7))
        merger.restore()
        keeps.append(a / acc_full[i])
    table.append({
        "rank": rank,
        "keep_min": round(float(min(keeps)), 3),
        "keep_mean": round(float(np.mean(keeps)), 3),
        "pass_gate_0.95": bool(min(keeps) >= 0.95),
        "mb_4skills_fp16": round(4 * deltas[active[0]].numel() * 2 * rank / 16 / 1024 / 1024, 2),
    })

out = {"acc_full": {str(k): round(v, 3) for k, v in acc_full.items()}, "rank_sweep": table}
out["decision"] = ("若仅 r=原秩 过门 → 热沉积产物必须按完整 LoRA 记账（截断不可晋升），"
                   "L1 预算决定常驻技能个数；若低秩也过门 → 截断路线成立。以本表为准。")
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(ART_DIR, "exp1c_ranksweep.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
