# -*- coding: utf-8 -*-
"""D3（MSP 存亡假设）在「参考骨干 Qwen2.5-3B ↔ 目标骨干 Qwen3-4B」这一对上重做。

为什么值得单独跑：如果 L0 换成 Qwen3-4B，它就是**同族同词表**的骨干对，
这是 MSP 最有利的情形（同 tokenizer、同架构家族、同训练分布）。
若连这一对都过不了（b/a < 0.95），"换 4B 能救 MSP"这个假设就被直接否掉。

协议与 e3_msp_split.py 完全一致（同分布锚点、标定集/评测集互斥、去均值余弦、打乱对照），
只把两个骨干的 hidden 提取换成 HF 通用实现。
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e3_msp_real import build_pairs, prompt_of, train_head, acc_of, fit_map, apply_map, DEV, ART
from loader import load_tokenizer, DEFAULT_L0

REF = "D:/Doubao-works/训练前置/02_基座模型/Qwen2.5-3B-Instruct"
TGT = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B"
LAYER = 18                     # 两者都是 36 层，取中层（L0 的 D3 结论也是中层可标定）


@torch.no_grad()
def hf_mid(path, tok, texts, layer=LAYER):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        h = m(input_ids=ids, output_hidden_states=True).hidden_states[layer]
        out.append(h[0, -1].float().cpu())
    H = torch.stack(out).to(DEV)
    n = m.config.num_hidden_layers
    del m
    torch.cuda.empty_cache()
    return H, n


def inter_cos(H):
    Hn = F.normalize(H, dim=1)
    S = Hn @ Hn.T
    n_ = len(H)
    return float((S.sum() - S.diagonal().sum()) / (n_ * n_ - n_))


def main():
    tok = load_tokenizer(DEFAULT_L0)          # Qwen2.5 词表（与 Qwen3-4B 的 vocab.json 逐字节相同）
    pairs = build_pairs(tok)
    texts = [prompt_of(p) for p in pairs]
    uniq = {}
    for p in pairs:
        uniq.setdefault(p["ans_id"], len(uniq))
    y = torch.tensor([uniq[p["ans_id"]] for p in pairs])
    n = len(pairs)
    print(f"[知识域] {n} 对；答案集 {len(uniq)} 类；中层 layer={LAYER}", flush=True)

    HQ, nq = hf_mid(REF, tok, texts)
    H0, n0 = hf_mid(TGT, tok, texts)
    print(f"[hidden] 参考(Qwen2.5-3B, {nq}层) {tuple(HQ.shape)} | 目标(Qwen3-4B, {n0}层) {tuple(H0.shape)}",
          flush=True)

    ic_q, ic_0 = inter_cos(HQ), inter_cos(H0)
    aq = math.degrees(math.acos(max(-1, min(1, ic_q))))
    a0 = math.degrees(math.acos(max(-1, min(1, ic_0))))
    print(f"[机制] 类间平均余弦：参考 {ic_q:.3f}（{aq:.1f}°）| 目标 {ic_0:.3f}（{a0:.1f}°）", flush=True)

    results = []
    for n_cal in (12, 24):
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        cal, ev = idx[:n_cal], idx[n_cal:]
        ev_t, cal_t = torch.tensor(ev), torch.tensor(cal)
        print(f"\n[切分] 标定 {len(cal)} / 评测 {len(ev)}", flush=True)

        Ka, Va, _ = train_head(H0, y)                       # 目标骨干本地上限
        acc_a = acc_of(Ka, Va, H0[ev_t], y[ev_t])
        Kq, Vq, _ = train_head(HQ, y)                       # 参考骨干读出口
        mp_q2l = fit_map(HQ[cal_t], H0[cal_t], k=min(16, n_cal - 1))
        mp_l2q = fit_map(H0[cal_t], HQ[cal_t], k=min(16, n_cal - 1))
        acc_b_key = acc_of(apply_map(Kq, mp_q2l), Vq, H0[ev_t], y[ev_t])
        acc_b_hid = acc_of(Kq, Vq, apply_map(H0[ev_t], mp_l2q), y[ev_t])
        acc_b = max(acc_b_key, acc_b_hid)
        G = torch.randn(HQ.shape[1], H0.shape[1], device=DEV) / math.sqrt(HQ.shape[1])
        acc_b0 = acc_of(Kq @ G, Vq, H0[ev_t], y[ev_t])
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
    verdict = (f"ISA 成立（b/a 最高 {best['ratio']:.3f} >= 0.95）" if ok else
               (f"部分成立（b/a {rs}）" if best["ratio"] >= 0.5 else
                f"不成立（b/a {rs}，机会 {best['chance']:.3f}）"))
    print("\n[判定] " + verdict)
    out = {"pair": {"reference": REF, "target": TGT, "layer": LAYER},
           "n_pairs": n, "n_classes": len(uniq), "splits": results,
           "mechanism": {"inter_class_cos_ref": round(ic_q, 4), "inter_class_cos_tgt": round(ic_0, 4),
                         "inter_class_angle_ref_deg": round(aq, 1),
                         "inter_class_angle_tgt_deg": round(a0, 1),
                         "map_error_angle_deg": results[0]["map_err_deg"],
                         "ratio_map_over_class": round(results[0]["map_err_deg"] / max(aq, 1e-6), 2)},
           "verdict": verdict, "pass": bool(ok)}
    p = os.path.join(ART, "e4_msp_qwen4b.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
