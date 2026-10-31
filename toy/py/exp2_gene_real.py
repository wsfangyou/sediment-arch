# -*- coding: utf-8 -*-
"""实验 2（真实版）+ 联合合并复测（v2 语料）：
 A. 联合合并干扰复测：技能区间互斥后，多 ΔW 直接相加常驻是否可行
    （对照 exp1 在旧共享区间语料上的 0.08–0.13 崩溃 → 证明 R4 干扰取决于技能重叠度）
 B. L4 基因化：6 个 ΔW 做共享 PCA 码本 → K 维潜码 → 重建 → 真实 acc 保真
    （对照 Node 版只看参数余弦的失败，这里以行为保真为准绳）"""
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
skills = sorted(deltas.keys())
rng_ev = np.random.default_rng(7)

acc_full = {}
for i in skills:
    merger.merge([deltas[i]])
    acc_full[i] = next_token_acc(model, chains[i], rng=rng_ev)
merger.restore()

# ---- A. 联合合并（全部 6 个完整 ΔW 同时常驻）----
merger.merge([deltas[i] for i in skills])
acc_joint = {i: next_token_acc(model, chains[i], rng=rng_ev) for i in skills}
merger.restore()
keep_joint = {i: acc_joint[i] / acc_full[i] for i in skills}

# ---- B. 基因化：共享 PCA 码本（对 6 个 delta 展平做 SVD）----
flats = torch.stack([deltas[i].reshape(-1).double() for i in skills])  # 6×DD
mean = flats.mean(0)
U, S, Vh = torch.linalg.svd(flats - mean, full_matrices=False)
gene = []
for K in [2, 3, 4, 5, 6]:
    rec = (flats - mean) @ Vh[:K].T @ Vh[:K] + mean
    keeps = []
    for n, i in enumerate(skills):
        d_r = rec[n].reshape(deltas[i].shape).float().to(DEV)
        merger.merge([d_r])
        keeps.append(next_token_acc(model, chains[i], rng=rng_ev) / acc_full[i])
        merger.restore()
    raw_mb = 6 * deltas[skills[0]].numel() * 4 / 1024 / 1024
    kb_mb = (6 * K + 6 * deltas[skills[0]].numel()) * 4 / 1024 / 1024  # 潜码+码本
    gene.append({"K": K, "keep_min": round(float(min(keeps)), 3), "keep_mean": round(float(np.mean(keeps)), 3),
                 "raw_kb": round(raw_mb * 1024, 1), "latent_plus_codebook_kb": round(kb_mb * 1024, 1)})

out = {
  "A_joint_merge_v2": {"keep": {str(k): round(v, 3) for k, v in keep_joint.items()},
                       "contrast_v1_shared_band": "旧语料联合合并 0.08–0.13"},
  "B_geneification": gene,
  "acc_full": {str(k): round(v, 3) for k, v in acc_full.items()},
}
out["verdicts"] = {
  "joint": ("区间互斥技能可直接联合常驻（R4 干扰取决于技能重叠度，非合并本身）"
            if min(keep_joint.values()) > 0.9 else "联合合并仍显著干扰，须隔离挂载"),
  "gene": ("存在 K 使行为保真≥0.95 → 基因化在“技能近正交”的域成立（码本含全部方差，注意其大小）"
           if any(g["keep_min"] >= 0.95 for g in gene) else
           "全部 K 行为保真 <0.95 → 基因化只可用于可容忍损失的长尾（与 Node 版 N5 结论互证）"),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(ART_DIR, "exp2_gene_real.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
