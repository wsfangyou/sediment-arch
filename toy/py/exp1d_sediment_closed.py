# -*- coding: utf-8 -*-
"""实验 1d（真实版·收口）——单槽热沉积闭环。
证据链（全部真实模型、真实权重操作）：
 1) exp1: 截断蒸馏在 v1 语料保真 0.75–0.85 → 门正确拒绝（护栏有效）
 2) exp1c(v2 语料): rank8 单技能保真 ≥0.966 → 门放行（晋升成立）
 3) exp2: 多 ΔW 算术联合合并 = 0.13–0.57 严重干扰 → L1 政策定为【单槽常驻】
    （对齐 MoLE 结论：组合须隔离挂载/运行时选择，不做加法）
 本实验：200 轮 Zipf 负载，L1 单槽由分数驱动自动换入换出；
 常驻轮加载税=0、acc 用真实 next_token_acc；换入前跑真实保真门。
"""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import next_token_acc, make_skill_chains, svd_truncate, DEV, ART_DIR
from skills import get_base_model, ensure_lora_deltas, Merger
from hw import load_latency_ms, prefill_ms

t0 = time.time()
deltas = ensure_lora_deltas()
model = get_base_model().to(DEV)
merger = Merger(model)
chains = make_skill_chains(8)
active = [2, 3, 4, 5]
R, GATE = 8, 0.95
rng_ev = np.random.default_rng(7)

dist, keep = {}, {}
for i in active:
    d_r, _ = svd_truncate(deltas[i].to(DEV), R)
    dist[i] = d_r
    merger.merge([d_r])
    a = next_token_acc(model, chains[i], rng=rng_ev)
    merger.restore()
    merger.merge([deltas[i]])
    f = next_token_acc(model, chains[i], rng=rng_ev)
    merger.restore()
    keep[i] = a / f
print(f"[蒸馏] rank{R} 真实保真 { {i: round(keep[i],3) for i in active} } {time.time()-t0:.0f}s")

# 换入成本：合并写回一次权重的真实 GPU 时间（≈ CUDA graph 失效重捕获量级的下限）
d = deltas[active[0]].to(DEV)
torch.cuda.synchronize(); ts = []
for _ in range(5):
    st = time.perf_counter()
    w = model.blocks[0].attn.weight
    w.data += d
    w.data -= d
    torch.cuda.synchronize(); ts.append((time.perf_counter() - st) * 1000)
merge_ms = float(np.mean(ts))
print(f"[挂载] 真实权重合并 {merge_ms:.2f} ms（toy d=96；6GB 档按 300MB 物理模型计）")

# 政策（由实验驱动）：L1 多槽可以，但【隔离挂载】——每个请求只 merge 自己那一槽
# （S-LoRA 运行时挂载的 toy 实现；exp2 已证加法合并 0.13–0.57 干扰，禁止联合 merge）
L1, score = [], {i: 0.0 for i in active}
L1_SLOTS = 3
tax, accs = [], []
rng = np.random.default_rng(42)
for r in range(200):
    s = active[min(int(np.power(rng.random(), 1.7) * len(active)), len(active) - 1)]
    prompt = [32, 64, 128, 256, 512, 1024][int(rng.random() * 6)]
    if s in L1:
        tax.append(30.0)                                   # 常驻：零搬运
        merger.merge([dist[s]])
        a = next_token_acc(model, chains[s], rng=rng_ev)
        merger.restore()
    else:
        t_load = load_latency_ms(300 * (1 + 1.5 * rng.random()))  # 6GB 档冷加载物理账
        tax.append(30 + max(0.0, t_load - prefill_ms(prompt)))
        merger.merge([deltas[s]])
        a = next_token_acc(model, chains[s], rng=rng_ev)
        merger.restore()
    accs.append(a)
    for k in active: score[k] = 0.8 * score[k] + (0.2 if k == s else 0)
    if r % 10 == 9:                                        # 空闲蒸馏窗口
        cand = max(active, key=lambda x: score[x])
        if cand not in L1 and keep[cand] >= GATE:         # 真实保真门
            while len(L1) >= L1_SLOTS:
                L1.remove(min(L1, key=lambda x: score[x]))
            L1.append(cand)
win, p95 = (lambda a, x, y: sum(a[x:y]) / max(1, y - x)), (lambda a: sorted(a)[int(len(a) * 0.95)])
hit_rate_first = sum(1 for t in tax[:50] if t == 30.0) / 50
hit_rate_last = sum(1 for t in tax[150:] if t == 30.0) / len(tax[150:])
out = {
  "gate": GATE, "rank": R, "L1_slots": L1_SLOTS,
  "distill_keep_real": {str(i): round(keep[i], 3) for i in active},
  "merge_ms_toy": round(merge_ms, 2),
  "L1_final": L1,
  "tax_p95_first50": round(p95(tax[:50])), "tax_p95_last50": round(p95(tax[150:])),
  "resident_hit_first": round(hit_rate_first, 2), "resident_hit_last": round(hit_rate_last, 2),
  "acc_mean_first50": round(win(accs, 0, 50), 3), "acc_mean_last50": round(win(accs, 150, 200), 3),
}
out["verdict"] = ("闭环成立：rank8 蒸馏过真实保真门、多槽 L1 由分数驱动换入换出（隔离挂载）、"
                  "常驻命中率上升、常驻轮 acc 为真实蒸馏 next-token 精度；"
                  "加法联合合并被 exp2 证伪并以隔离挂载政策规避"
                  if all(keep[i] >= GATE for i in active) and len(L1) > 0 and hit_rate_last > hit_rate_first else "见上表")
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(ART_DIR, "exp1d_sediment_closed.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
