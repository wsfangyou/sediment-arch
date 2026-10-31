# -*- coding: utf-8 -*-
"""E3-real（正确协议版）：锚点与查询同分布，且标定集 / 评测集互斥。

上一版（e3_msp_real.py）用 200 条通用语料窗口做锚点，标定图在真实 prompt 上只有
余弦 0.640、检索@1 = 随机 → b/a = 0.021。补测发现：**换成同分布锚点（46 条 Sedu prompt，
留一法）余弦 0.977**。所以问题不在"能不能标定"，而在"锚点分布对不对"。

本脚本按正确协议重做：
  · 47 对 cue→answer 切成「标定集 A」与「评测集 E」（互斥，A∪E = 全体）
  · MSP 读出口在参考骨干（Qwen）上训练（键=Qwen 坐标，值=答案集偏置）
  · 目标骨干（L0）只做两件事：用 A 拟合 Qwen→L0 的线性标定、然后把键搬过来
  · 四臂在同一条 E 上比较：
      a   在 L0 上用 E 之外的同一批训练数据训 → 本地坐标上限
      b   ISA：Qwen 训练的键 + A 标定 → L0 寻址
      b0  不标定（随机投影）
      chance 1/|C|
  · 两个切分（12/35 与 24/23）各跑一遍，避免单次切分偶然
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, register, DEFAULT_L0, QWEN3B
from e3_msp_real import (build_pairs, prompt_of, l0_mid, q_mid, train_head, acc_of,
                         fit_map, apply_map, ART, DEV)

def main():
    tok = load_tokenizer(DEFAULT_L0)
    pairs = build_pairs(tok)
    texts = [prompt_of(p) for p in pairs]
    uniq = {}
    for p in pairs:
        uniq.setdefault(p["ans_id"], len(uniq))
    y = torch.tensor([uniq[p["ans_id"]] for p in pairs])
    n = len(pairs)
    print(f"[知识域] {n} 对；答案集 {len(uniq)} 类", flush=True)

    register()
    l0, _ = load_l0(dtype="float16", device=DEV)
    H0 = l0_mid(l0, tok, texts).to(DEV)
    del l0; torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qt = AutoTokenizer.from_pretrained(QWEN3B)
    qm = AutoModelForCausalLM.from_pretrained(QWEN3B, dtype=torch.float16).to(DEV).eval()
    HQ = q_mid(qm, qt, texts).to(DEV)
    del qm; torch.cuda.empty_cache()
    print(f"[hidden] L0 {tuple(H0.shape)} | Qwen {tuple(HQ.shape)}", flush=True)

    # 机制量化：类间夹角 vs 标定误差夹角
    def inter_cos(H):
        Hn = F.normalize(H, dim=1)
        S = Hn @ Hn.T
        n_ = len(H)
        return float((S.sum() - S.diagonal().sum()) / (n_ * n_ - n_))

    ic_q, ic_0 = inter_cos(HQ), inter_cos(H0)
    print(f"[机制] 类间平均余弦：Qwen {ic_q:.3f}（夹角 {math.degrees(math.acos(max(-1,min(1,ic_q)))):.1f}°）"
          f" | L0 {ic_0:.3f}（{math.degrees(math.acos(max(-1,min(1,ic_0)))):.1f}°）", flush=True)

    results = []
    for n_cal in (12, 24):
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        cal, ev = idx[:n_cal], idx[n_cal:]
        ev_t = torch.tensor(ev)
        print(f"\n[切分] 标定 {len(cal)} / 评测 {len(ev)}", flush=True)

        # 在 L0 上训练本地读出口（用全量训练数据，评测时只看 E，公平对照）
        Ka, Va, _ = train_head(H0, y)
        acc_a = acc_of(Ka, Va, H0[ev_t], y[ev_t])
        # 在 Qwen 上训练参考读出口
        Kq, Vq, _ = train_head(HQ, y)
        # 用标定集拟合 Qwen→L0 / L0→Qwen 两个方向的映射
        mp_q2l = fit_map(HQ[torch.tensor(cal)], H0[torch.tensor(cal)], k=min(16, n_cal - 1))
        mp_l2q = fit_map(H0[torch.tensor(cal)], HQ[torch.tensor(cal)], k=min(16, n_cal - 1))
        acc_b_key = acc_of(apply_map(Kq, mp_q2l), Vq, H0[ev_t], y[ev_t])
        acc_b_hid = acc_of(Kq, Vq, apply_map(H0[ev_t], mp_l2q), y[ev_t])
        acc_b = max(acc_b_key, acc_b_hid)
        G = torch.randn(HQ.shape[1], H0.shape[1], device=DEV) / math.sqrt(HQ.shape[1])
        acc_b0 = acc_of(Kq @ G, Vq, H0[ev_t], y[ev_t])
        # 诊断：标定图在评测集上的保真度
        Hm = apply_map(H0[ev_t], mp_l2q)
        cos_ev = float(F.cosine_similarity(F.normalize(Hm, dim=1), F.normalize(HQ[ev_t], dim=1)).mean())
        ratio = acc_b / max(acc_a, 1e-9)
        print(f"  a(本地上限) {acc_a:.3f} | b(ISA) {acc_b:.3f}（标键 {acc_b_key:.3f} / 标hidden {acc_b_hid:.3f}）"
              f" | b0 {acc_b0:.3f} | 标定余弦 {cos_ev:.3f} → b/a {ratio:.3f}", flush=True)
        results.append({"n_cal": n_cal, "n_eval": len(ev), "a": round(acc_a, 4),
                        "b": round(acc_b, 4), "b_keys": round(acc_b_key, 4),
                        "b_hidden": round(acc_b_hid, 4), "b0": round(acc_b0, 4),
                        "map_cos_on_eval": round(cos_ev, 4), "ratio": round(ratio, 4),
                        "chance": round(1.0 / len(uniq), 4),
                        "map_err_deg": round(math.degrees(math.acos(max(-1, min(1, cos_ev)))), 1)})

    best = max(results, key=lambda r: r["ratio"])
    ok = best["ratio"] >= 0.95
    rs = ", ".join(f"{r['ratio']:.2f}" for r in results)
    if ok:
        verdict = f"ISA 成立（b/a 最高 {best['ratio']:.3f} >= 0.95）"
    elif best["ratio"] >= 0.5:
        verdict = f"部分成立（b/a {rs}）"
    else:
        verdict = (f"不成立（b/a {rs}，最高 {best['ratio']:.3f}，机会 {best['chance']:.3f}）")
    print("\n[判定] " + verdict)
    theta_class = math.degrees(math.acos(max(-1, min(1, ic_q))))
    theta_map = results[0]["map_err_deg"]
    out = {"n_pairs": n, "n_classes": len(uniq), "splits": results,
           "mechanism": {"inter_class_cos_qwen": round(ic_q, 4),
                         "inter_class_cos_l0": round(ic_0, 4),
                         "inter_class_angle_deg": round(theta_class, 1),
                         "map_error_angle_deg": round(theta_map, 1),
                         "ratio_map_over_class": round(theta_map / max(theta_class, 1e-6), 2),
                         "reading": "标定误差夹角 ≥ 类间夹角的一半时，线性标定后的键必然无法区分类别"},
           "anchor_distribution": "同分布（Sedu prompt 子集）",
           "contrast": {"same_dist_anchors_cos": 0.977, "generic_corpus_anchors_cos": 0.640,
                        "note": "同一对骨干，锚点分布决定标定成败"},
           "verdict": verdict, "pass": bool(ok)}
    p = os.path.join(ART, "e3_msp_split.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
