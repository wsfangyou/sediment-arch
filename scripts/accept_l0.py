# -*- coding: utf-8 -*-
"""L0 中心模型验收 —— 主脚本（D1 语言能力与显存预算 / D4 调度接口 + 汇总）。

2026-10-30 重做要点：
  · D1 的生成抽查必须在**修好因果掩码之后**重跑：旧读数里的「于于于于」不是模型不行，
    而是 attention_mask 让注意力变双向导致的解码退化（见 model_l0.hidden_states 注释）。
  · D2/D3 的测量装置已重写为独立脚本（accept_l0_d2.py / accept_l0_d3.py），各自产出
    d2_*.json / d3_msp.json，本脚本只做汇总，避免"边跑边改判分口径"。
  · 汇总判定沿用旧规则：四项职责过 ≥3 项 → 可承担中心模型职位。

用法：
  python accept_l0.py            # 只跑 D1、D4 并汇总已有 D2/D3 结果
  python accept_l0.py --d1-only  # 只跑 D1
"""
import os, sys, json, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, register, DEFAULT_L0
from model_l0 import SedimentConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
os.makedirs(ART, exist_ok=True)
DEV = "cuda:0"


def log(*a):
    print(*a, flush=True)


def duty1(model, tok, cfg, stream):
    log("\n" + "=" * 68)
    log("D1 语言能力与显存预算")
    log("=" * 68)
    model.eval()
    rng = np.random.default_rng(7)
    losses = []
    with torch.no_grad():
        for _ in range(24):
            x, y = stream.batch(4, train=False, rng=rng)
            x, y = x.to(DEV), y.to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                losses.append(model(x, y).loss.item())
    vl = float(np.mean(losses))
    ppl = math.exp(min(vl, 20))
    log(f"验证集 loss {vl:.4f} → ppl {ppl:,.1f}（152k 词表随机基线 ppl=151936）")

    prompts = ["人工智能的核心挑战在于", "中国的传统文化包括", "在水处理过程中，"]
    gens = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=48,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 repetition_penalty=1.0)
        g = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        gens.append({"prompt": p, "completion": g})
        log(f"  「{p}」→「{g[:56]}」")

    nparam = sum(p.numel() for p in model.parameters())
    uniq = nparam - model.lm_head.weight.numel() if bool(torch.equal(
        model.lm_head.weight.data, model.embed_tokens.weight.data)) else nparam
    w_fp16_mb = nparam * 2 / 1024 / 1024
    kv = model.kv_bytes_per_token()
    for ctx in (2048, 4096):
        kv_mb = kv * ctx / 1024 / 1024
        room = 6144 - w_fp16_mb - kv_mb - 150
        log(f"6GB 档 @ctx{ctx}: 权重 {w_fp16_mb:.0f}MB + KV {kv_mb:.0f}MB + 工作区 150MB "
            f"→ L1 热沉积带可留 {room/1024:.2f} GB")
    kv_mha = 2 * cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim * 2
    res = {"val_loss": round(vl, 4), "val_ppl": round(ppl, 1),
           "random_ppl": cfg.vocab_size, "params_B": round(nparam / 1e9, 3),
           "params_B_tied_dedup": round(uniq / 1e9, 3),
           "weight_fp16_mb": round(w_fp16_mb),
           "weight_fp16_mb_if_tied": round(uniq * 2 / 1024 / 1024),
           "kv_bytes_per_token": kv, "kv_vs_mha": round(kv / kv_mha, 2),
           "l1_room_gb_at_4k": round((6144 - uniq * 2 / 1024 / 1024 - kv * 4096 / 1024 / 1024 - 150) / 1024, 2),
           "generations": gens}
    res["pass"] = bool(ppl < cfg.vocab_size / 10)
    log(f"判定: {'✅' if res['pass'] else '❌'} 语言能力显著优于随机基线")
    return res


def duty4(cfg):
    log("\n" + "=" * 68)
    log("D4 调度接口（generate / hidden_states / KV 记账）")
    log("=" * 68)
    m, tok = load_l0(dtype="float16")
    ids = tok("沉积架构的核心思想是", return_tensors="pt").input_ids.to(DEV)
    t0 = __import__("time").time()
    with torch.no_grad():
        o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
                       do_sample=False, pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
    gen_ms = (__import__("time").time() - t0) * 1000
    h = m.hidden_states(ids)
    kv = m.kv_bytes_per_token()
    res = {"generate_ok": bool(o.shape[1] > ids.shape[1]),
           "generate_ms_for_32tok": round(gen_ms),
           "hidden_states_shape": list(h.shape),
           "kv_bytes_per_token": kv,
           "kv_4k_mb": round(kv * 4096 / 1024 / 1024, 1),
           "hf_compatible": isinstance(m, torch.nn.Module) and hasattr(m, "generate")}
    log(f"generate 32 tok 用时 {gen_ms:.0f} ms | hidden_states {tuple(h.shape)}")
    log(f"KV {kv} B/token → 4096 上下文 {res['kv_4k_mb']} MB（供竞价区记账）")
    res["pass"] = bool(res["generate_ok"] and h.shape[-1] == cfg.hidden_size)
    log(f"判定: {'✅' if res['pass'] else '❌'} 调度接口可用")
    return res


def load_json(name):
    p = os.path.join(ART, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-only", action="store_true")
    args = ap.parse_args()
    register()
    from pretrain_l0 import TokenStream
    cfg = SedimentConfig.from_pretrained(DEFAULT_L0)
    stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)

    prev = load_json("l0_acceptance.json") or {}
    out = dict(prev)
    model, tok = load_l0(dtype="float16")
    out["D1_language_vram"] = duty1(model, tok, cfg, stream)
    del model; torch.cuda.empty_cache()
    if not args.d1_only:
        out["D4_scheduler_api"] = duty4(cfg)
        torch.cuda.empty_cache()

    d2 = load_json("d2_curve.json")
    d3 = load_json("d3_msp.json")
    # D2：优先取 replay 版最佳配置（replay 混采 + EOS 修复），退回 d2_curve
    best = load_json("e2_replay2500.json") or load_json("e2_replay35long.json")
    if best and (best.get("after_eos_fix") or best.get("after")):
        b = dict(best.get("after_eos_fix") or best["after"])
        reg = b.get("ppl_regress") or best.get("ppl_regress_vs_bare")
        bare_gen = float(best.get("bare", {}).get("A_gen", 0.0))
        ok = bool(b["A_tf"] >= 0.90 and b["A_gen"] >= 0.80 and bare_gen <= 0.15)
        out["D2_skill_hosting"] = {
            "verdict": (f"OK 通过（replay 35% × 2500 步：A_tf {b['A_tf']:.3f} / A_gen {b['A_gen']:.3f}）"
                        if ok else f"未全过（A_tf {b['A_tf']:.3f} / A_gen {b['A_gen']:.3f}）"),
            "pass": ok,
            "A_tf": b["A_tf"], "A_gen": b["A_gen"], "bare_A_gen": bare_gen,
            "config": {"replay": best["protocol"].get("replay_ratio"),
                       "steps_total": 2500, "bs": best["protocol"]["bs"],
                       "eos_fix": "eos_token_id=[151645,151643]"},
            "c_gate": {"regress": reg, "limit": 1.02, "pass": bool(reg and reg <= 1.02)},
            "old_curve": d2["curve"] if d2 else []}
    elif d2:
        last = d2["curve"][-1]
        out["D2_skill_hosting"] = {
            "verdict": d2["verdict"], "pass": bool(d2["pass"]),
            "A_tf": last["A_tf"], "A_gen": last["A_gen"],
            "bare": d2["bare"], "steps": last["step"], "curve": d2["curve"]}
    if d3:
        out["D3_msp_anchor"] = {"verdict": d3["verdict"], "pass": bool(d3["pass"]),
                                "anchors": d3["anchors"], "results": d3["results"]}

    duties = ["D1_language_vram", "D2_skill_hosting", "D3_msp_anchor", "D4_scheduler_api"]
    passed = [k for k in duties if isinstance(out.get(k), dict) and out[k].get("pass")]
    cg = out.get("D2_skill_hosting", {}).get("c_gate")
    out["summary"] = {"l1_promotion_blocker": (None if (cg and cg.get("pass"))
                                               else (f"C 门 {cg['regress']}x > 1.02" if cg else None)),"duties_present": [k for k in duties if k in out],
                      "duties_passed": passed, "n_passed": len(passed),
                      "verdict": ("✅ 可承担中心模型（L0）职位" if len(passed) >= 3
                                  else "⚠️ 未达中心模型标准，见各项")}
    log("\n" + "=" * 68)
    log(f"通过职责: {passed} ({len(passed)}/4)")
    log(out["summary"]["verdict"])
    with open(os.path.join(ART, "l0_acceptance.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"结果 → {os.path.join(ART, 'l0_acceptance.json')}")


if __name__ == "__main__":
    main()
