# -*- coding: utf-8 -*-
"""实验 1（真实版，方案 §7 实验1）——沉积闭环：
  基座只学过链 0–1；链 2–7 = 6 个真实训练的 LoRA 技能。
  200 轮混合负载（Zipf 抽 4 个活跃技能）：
   · 每轮评测真实 next-token acc（骨干裸 / 完整 LoRA / 蒸馏 LoRA）
   · score 超阈值 → SVD 低秩蒸馏（r=4）→ 晋升门（真实保真 ≥95%）
     → 合并常驻（多技能联合合并，真实测量相互干扰 = 报告 R4 的直接证据）
   · L1 预算：以蒸馏产物真实字节计（toy 尺度按参数×2B 折算）
  指标：各形态 acc、蒸馏保真、联合合并干扰、加载税曲线（物理用 hw 模型）。
诚实边界：toy 尺度（d=96）的绝对 MB 无意义，相对关系与机制动力学有效。
"""
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import make_skill_chains, sample_batch, next_token_acc, svd_truncate, DEV, D_MODEL, VOCAB, ART_DIR
from skills import get_base_model, ensure_lora_deltas, Merger, SKILL_CHAINS
from hw import H, load_latency_ms, prefill_ms

t0 = time.time()
deltas = ensure_lora_deltas()
model = get_base_model().to(DEV)
merger = Merger(model)
chains = make_skill_chains(8)
print(f"[加载] 技能库就绪 {time.time()-t0:.0f}s")

R_FULL = 8          # 蒸馏秩
active = [2, 3, 4, 5]            # Zipf 负载只触达这 4 个
sleepers = [6, 7]                # 沉睡 → 基因化候选（本实验仅统计）

acc_bare = {i: next_token_acc(model, chains[i], rng=np.random.default_rng(7)) for i in SKILL_CHAINS}
acc_full = {}
for i in SKILL_CHAINS:
    merger.merge([deltas[i]])
    acc_full[i] = next_token_acc(model, chains[i], rng=np.random.default_rng(7))
merger.restore()

# 蒸馏 + 真实保真
distill, keep_real = {}, {}
for i in active:
    d_r, fro_rel = svd_truncate(deltas[i].to(DEV), R_FULL)
    distill[i] = d_r
    merger.merge([d_r])
    a = next_token_acc(model, chains[i], rng=np.random.default_rng(7))
    merger.restore()
    keep_real[i] = a / acc_full[i]
    print(f"[蒸馏 s{i}] rank{R_FULL} acc {a:.3f} vs full {acc_full[i]:.3f} 保真 {keep_real[i]:.3f} (Fro残留 {fro_rel:.2f})")

# 联合合并干扰（R4 真实测量）
interf = {}
merger.merge([distill[i] for i in active])
for i in active:
    a = next_token_acc(model, chains[i], rng=np.random.default_rng(7))
    interf[i] = a / acc_full[i]
merger.restore()
print(f"[联合] 4 蒸馏技能同时合并后各自保真: {interf}")

# ---- 200 轮闭环：L1 集合动态演化（晋升由真实保真门控） ----
rng = np.random.default_rng(42)
L1 = set()
l1_bytes = lambda: sum(distill[i].numel() * 2 for i in L1) if L1 else 0
L1_BUDGET = int(sum(distill[i].numel() * 2 for i in active[:3]))  # 预算=3个蒸馏产物 → 强制竞争
scores = {i: 0.0 for i in SKILL_CHAINS}
tax, accs, promo = [], [], []
for r in range(200):
    s = active[min(int(np.power(rng.random(), 1.7) * len(active)), len(active) - 1)]
    # 加载税（物理模型）：L1 常驻=0；否则冷加载
    tax.append(30 + max(0.0 if s in L1 else load_latency_ms(300 * (1 + 1.5 * rng.random())) - prefill_ms([32, 64, 128, 256, 512, 1024][int(rng.random() * 6)]), 0))
    # 本轮真实 acc：L1 常驻用联合合并值，否则完整 LoRA
    if s in L1:
        merger.merge([distill[i] for i in L1])
        a = next_token_acc(model, chains[s], rng=np.random.default_rng(7))
        merger.restore()
    else:
        merger.merge([deltas[s]])
        a = next_token_acc(model, chains[s], rng=np.random.default_rng(7))
        merger.restore()
    accs.append(a)
    scores[s] = 0.75 * scores[s] + 0.25 * 1.0
    if r % 10 == 9:
        for s2 in active:
            if s2 not in L1 and scores[s2] > 0.6 and keep_real[s2] >= 0.95:
                while l1_bytes() + distill[s2].numel() * 2 > L1_BUDGET and L1:
                    L1.discard(min(L1, key=lambda x: scores[x]))
                if s2 not in L1:
                    L1.add(s2); promo.append({"skill": s2, "round": r, "keep": round(keep_real[s2], 3)})
                break

win = lambda a, x, y: sum(a[x:y]) / max(1, y - x)
out = {
  "acc_bare": {str(k): round(v, 3) for k, v in acc_bare.items()},
  "acc_full_lora": {str(k): round(v, 3) for k, v in acc_full.items()},
  "distill_keep_real": {str(k): round(v, 3) for k, v in keep_real.items()},
  "joint_merge_interference": {str(k): round(v, 3) for k, v in interf.items()},
  "promotions": promo,
  "tax_p95_first50": round(sorted(tax[:50])[int(47)]),
  "tax_p95_last50": round(sorted(tax[150:])[int(47)]),
  "acc_first50": round(win(accs, 0, 50), 3), "acc_last50": round(win(accs, 150, 200), 3),
  "L1_final": sorted(L1),
}
out["verdict"] = ("闭环成立：真实晋升发生、加载税下降、任务分由真实 acc 衡量"
                  if promo and out["tax_p95_last50"] < out["tax_p95_first50"] else "机制未闭环，检查阈值")
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(ART_DIR, "exp1_sediment_real.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
