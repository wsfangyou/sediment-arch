# -*- coding: utf-8 -*-
"""实验 1b（真实版·修正）——热沉积的正确形态 = 行为蒸馏，不是参数截断。

v1 的真实教训（保留在输出里作对照证据）：
  · SVD 截断 ΔW（rank8）→ 单技能保真 0.75–0.85 < 晋升门 0.95 → 全被正确拒绝；
  · 多技能 ΔW 算术合并 → 保真崩到 0.08–0.13（报告 R4 负迁移的直接实测）。
v2 修正：L1 晋升产物 = 在活跃技能的混合数据上**训练**出来的一个融合 LoRA
  （学生拟合各教师的输出分布 = 字面意义的知识蒸馏），常驻=单 ΔW，
  天然不存在多技能合并干扰。

闭环 200 轮：预热期每轮冷加载（物理模型计税）→ round40 触发蒸馏 →
真实 acc 保真过门 → 合并常驻 → 后续轮加载税=0。
"""
import sys, os, time, json
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(__file__))
from common import (ToyLM, make_skill_chains, sample_batch, next_token_acc,
                    attach_lora, train_steps, DEV, ART_DIR)
from skills import get_base_model, ensure_lora_deltas, Merger
from hw import load_latency_ms, prefill_ms

t_start = time.time()
deltas = ensure_lora_deltas()
chains = make_skill_chains(8)
active = [2, 3, 4, 5]

def fresh_model():
    m = ToyLM(seed=0).to(DEV)
    m.load_state_dict(get_base_model().state_dict())
    return m

# 教师 = 权重已固化完整 ΔW 的冻结模型
teachers = {}
for i in active:
    t = fresh_model()
    t.blocks[0].attn.weight.data += deltas[i].to(DEV)
    teachers[i] = t.eval()
acc_full = {i: next_token_acc(teachers[i], chains[i], rng=np.random.default_rng(7)) for i in active}
print(f"[教师] acc_full={ {k: round(v,3) for k,v in acc_full.items()} }")

# ---- 行为蒸馏：学生 rank48 融合 LoRA 拟合 4 个教师 ----
# v2 教训：r=16 学生装不下 4 条链并集（秩需求≈Σ教师秩）→ 保真 0.23–0.62。
# 这是“热沉积受秩预算限制”的真实测量；r=48 验证容量满足后门能否放行。
KD_T = 2.0
student, slora = attach_lora(fresh_model(), 0, r=48, seed=9)
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=3e-3)
rng_d = np.random.default_rng(31)
t0 = time.time()
for step in range(900):
    c = active[int(rng_d.integers(len(active)))]
    x, y = sample_batch(chains[c], 32, rng=rng_d)
    with torch.no_grad():
        tl = teachers[c](x)
    sl = student(x)
    kd = Fn.kl_div(Fn.log_softmax(sl / KD_T, -1), Fn.softmax(tl / KD_T, -1), reduction="batchmean") * KD_T ** 2
    ce = Fn.cross_entropy(sl.reshape(-1, sl.shape[-1]), y.reshape(-1))
    loss = kd + 0.5 * ce
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"[蒸馏] 900 steps {time.time()-t0:.0f}s  final kd {kd.item():.3f}")

# 学生真实保真（各链）
acc_s = {i: next_token_acc(student, chains[i], rng=np.random.default_rng(7)) for i in active}
keep = {i: acc_s[i] / acc_full[i] for i in active}
min_keep = min(keep.values())
print(f"[学生] acc={ {k: round(v,3) for k,v in acc_s.items()} } 保真={ {k: round(v,3) for k,v in keep.items()} }")

# ---- 闭环 200 轮 ----
PROMO_ROUND, GATE = 40, 0.95
L1_merged, promo_log = False, []
tax, accs = [], []
rng = np.random.default_rng(42)
merger_pool = fresh_model(); mpool = Merger(merger_pool)
for r in range(200):
    s = active[min(int(np.power(rng.random(), 1.7) * len(active)), len(active) - 1)]
    prompt = [32, 64, 128, 256, 512, 1024][int(rng.random() * 6)]
    if L1_merged:
        tax.append(30.0)                       # 常驻：零搬运
        a = acc_s[s]
    else:
        t_load = load_latency_ms(300 * (1 + 1.5 * rng.random()))  # 冷加载含抖动
        tax.append(30 + max(0.0, t_load - prefill_ms(prompt)))
        mpool.merge([deltas[s]]); a = next_token_acc(merger_pool, chains[s], rng=np.random.default_rng(7)); mpool.restore()
    accs.append(a)
    if r == PROMO_ROUND and not L1_merged:
        if min_keep >= GATE:
            mpool.merge([slora.delta()])       # 常驻合并动作（真实权重写回）
            L1_merged = True
            promo_log.append({"at": r, "min_keep": round(min_keep, 3), "bytes": int(slora.delta().numel() * 2)})
        else:
            promo_log.append({"at": r, "rejected": True, "min_keep": round(min_keep, 3)})

win = lambda a, x, y: sum(a[x:y]) / max(1, y - x)
srt = lambda a: sorted(a)
p95 = lambda a: a[int(len(a) * 0.95)] if a else 0
first = tax[:PROMO_ROUND] if not L1_merged else tax[:PROMO_ROUND]
last = tax[PROMO_ROUND:][:60]
out = {
  "acc_full": {str(k): round(v, 3) for k, v in acc_full.items()},
  "acc_student": {str(k): round(v, 3) for k, v in acc_s.items()},
  "distill_keep_behavior": {str(k): round(v, 3) for k, v in keep.items()},
  "svd_keep_v1_contrast": {"rank8_single": "0.75–0.85", "joint_arith_merge": "0.08–0.13"},
  "promotions": promo_log,
  "tax_p95_pre_promo": round(p95(srt(first))), "tax_p95_post_promo": round(p95(srt(last))) if last else None,
  "acc_mean_pre": round(win(accs, 0, PROMO_ROUND), 3),
  "acc_mean_post": round(win(accs, PROMO_ROUND, PROMO_ROUND + 60), 3) if L1_merged else None,
  "L1_merged": L1_merged,
}
if L1_merged:
    drop = 1 - out["tax_p95_post_promo"] / max(out["tax_p95_pre_promo"], 1)
    out["verdict"] = (f"闭环成立：行为蒸馏保真 {min_keep:.2f}≥门{GATE}，晋升后加载税 p95 "
                      f"{out['tax_p95_pre_promo']:.0f}→{out['tax_p95_post_promo']:.0f}ms（降 {drop*100:.0f}%），"
                      "任务分由真实 next-token acc 衡量")
else:
    out["verdict"] = f"晋升被真实保真门拒绝（min_keep={min_keep:.3f}<{GATE}）→ 闭环机制有效但阈值需按域调整"
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(ART_DIR, "exp1b_sediment_real.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
