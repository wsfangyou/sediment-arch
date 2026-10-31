# -*- coding: utf-8 -*-
"""Step B0 —— 训练前冒烟验证（长任务的风险闸门，v2）。

必须在启动长训练前确认：
 1. 参数量落在授权的 0.5–1.5B 区间
 2. fp32 主权重 + autocast fp16 + GradScaler 数值稳定（无 NaN/Inf）
 3. 显存配置可行 → 选出吞吐最优的 (优化器, bs, ga, 梯度检查点) 组合
    · 关键杠杆：bitsandbytes AdamW8bit 把 m/v 从 6.6GB 压到 1.66GB
 4. peft 能按 Qwen 同名 target_modules 注入 LoRA（项目适配承诺：
    已验收的 Sedu 技能管线可原样复用到自制基座）
 5. HF 兼容导出（config.json + safetensors + tied embedding）可重载且输出一致
用合成 token 流，不触碰 corpus/tokens.bin（后台正在写）。
"""
import os, sys, json, time, math
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_l0 import SedimentBase, Config

DEV = "cuda:0"
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smoke_tmp")
os.makedirs(TMP, exist_ok=True)
GB = 2**30


def log(*a):
    print(*a, flush=True)


def make_opt(model, kind, lr, wd):
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        (nodecay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": nodecay, "weight_decay": 0.0}]
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95))
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def run_cfg(name, opt_kind, bs, ga, seq, grad_ckpt, steps=6):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = Config()
    model = SedimentBase(cfg).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    model.train()
    opt = make_opt(model, opt_kind, 3e-4, 0.1)
    scaler = GradScaler("cuda")
    losses, gnorms, finite = [], [], True
    t0 = time.time()
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(ga):
            x = torch.randint(0, cfg.vocab_size, (bs, seq), device=DEV)
            y = torch.randint(0, cfg.vocab_size, (bs, seq), device=DEV)
            with autocast("cuda", dtype=torch.float16):
                loss = model(x, y, use_ckpt=grad_ckpt).loss
            scaler.scale(loss / ga).backward()
            acc += loss.item() / ga
        scaler.unscale_(opt)
        gn = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
        if not torch.isfinite(gn):
            finite = False
        scaler.step(opt); scaler.update()
        losses.append(acc); gnorms.append(float(gn))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    tps = bs * ga * seq / dt
    mem = torch.cuda.max_memory_allocated() / GB
    res = torch.cuda.max_memory_reserved() / GB
    h1b = 1e9 / tps / 3600
    log(f"\n── {name} [{opt_kind} bs={bs} ga={ga} ckpt={grad_ckpt}]")
    log(f"   {nparam/1e9:.3f}B | {dt*1000:.0f} ms/step | {tps:,.0f} tok/s | "
        f"alloc {mem:.1f} / reserved {res:.1f} GiB")
    log(f"   loss {[round(x,2) for x in losses]} | gn 有限={finite} | "
        f"NaN={'无' if finite and all(map(math.isfinite, losses)) else '有 ❌'}")
    log(f"   → 1B token 单卡 {h1b:.1f} h ({h1b/24:.1f} 天)")
    del model, opt, scaler
    torch.cuda.empty_cache()
    return {"cfg": name, "opt": opt_kind, "bs": bs, "ga": ga, "seq": seq,
            "grad_ckpt": grad_ckpt, "params_B": round(nparam / 1e9, 3),
            "ms_step": round(dt * 1000), "tok_s": round(tps),
            "mem_alloc_gib": round(mem, 1), "mem_reserved_gib": round(res, 1),
            "loss_first": round(losses[0], 3), "loss_last": round(losses[-1], 3),
            "finite": bool(finite), "hours_1b_tok": round(h1b, 1),
            "fits": res < 21.0 and finite}


def check_lora():
    """项目适配核心验证：peft 按 Qwen 同名 target_modules 注入。"""
    from peft import LoraConfig, TaskType, get_peft_model
    cfg = Config()
    model = SedimentBase(cfg).to(DEV).half()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                    lora_dropout=0.02, target_modules=targets)
    m = get_peft_model(model, lc)
    m.print_trainable_parameters()
    injected = sorted({n.split(".")[-2] for n, _ in m.named_modules() if "lora_A" in n})
    tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
    log(f"\n── LoRA 注入（Qwen 同名 target_modules）：✅ 成功")
    log(f"   注入位置 {injected} | 可训练 {tp/1e6:.2f}M ({tp/sum(p.numel() for p in model.parameters())*100:.2f}%)")
    x = torch.randint(0, cfg.vocab_size, (1, 64), device=DEV)
    with torch.no_grad(), autocast("cuda", dtype=torch.float16):
        out = m(input_ids=x).logits
    log(f"   前向 OK logits {tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
    del m, model
    torch.cuda.empty_cache()
    return {"peft_named_like_qwen": True, "injected": injected,
            "trainable_M": round(tp / 1e6, 2)}


def check_export():
    """HF 兼容导出 → 重载一致性；并给出低显存部署账。"""
    from safetensors.torch import save_file, load_file
    cfg = Config()
    model = SedimentBase(cfg).float()
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    tied = cfg.tie_word_embeddings and "lm_head.weight" in sd
    if tied:
        sd = {k: v for k, v in sd.items() if k != "lm_head.weight"}
    p = os.path.join(TMP, "model.safetensors")
    save_file(sd, p)
    size = os.path.getsize(p) / 1e9
    m2 = SedimentBase(cfg).float()
    missing, _ = m2.load_state_dict(load_file(p), strict=False)
    m2.tie_weights()          # HF 标准：tied 权重由 embed 复原
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        same = torch.allclose(model(x).logits, m2(x).logits, atol=1e-5)
    kv = model.kv_bytes_per_token()
    kv_mha = 2 * cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim * 2
    log(f"\n── 导出: {size:.2f} GB | tied={tied} | missing={list(missing)[:2]} | 重载一致={same}")
    log(f"   KV/token {kv} B = 同规模 MHA 的 {kv/kv_mha:.2f}× "
        f"→ 4096 上下文仅 {kv*4096/1e6:.0f} MB（省 {100*(1-kv/kv_mha):.0f}%）")
    log(f"   6GB 档预算: 权重 fp16 {size*1000/2:.0f} MB + KV(2k) {kv*2048/1e6:.0f} MB "
        f"+ 工作区 150 MB → L1 可留 {(6144 - size*1000/2 - kv*2048/1e6 - 150)/1024:.2f} GB")
    return {"export_gb": round(size, 2), "tied": tied, "reload_identical": bool(same),
            "kv_bytes_per_token": kv, "kv_vs_mha": round(kv / kv_mha, 2),
            "kv_4k_ctx_mb": round(kv * 4096 / 1e6),
            "fp16_weight_mb": round(size * 1000 / 2)}


if __name__ == "__main__":
    torch.manual_seed(0)
    log("=" * 70)
    log("Sediment-Base (L0) 训练前冒烟验证 v2")
    log("=" * 70)
    cfg = Config()
    log(f"架构 d={cfg.hidden_size} L={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
        f"kv_heads={cfg.num_key_value_heads} ffn={cfg.intermediate_size} "
        f"vocab={cfg.vocab_size} tie={cfg.tie_word_embeddings}")

    results = []
    # 8bit 优化器省 ~5GB + 分块CE省输出层显存 → 试免梯度检查点的大 batch
    for name, ok, bs, ga, gc in [
        ("8bit-bs4-nockpt", "adamw8bit", 4, 2, False),
        ("8bit-bs6-nockpt", "adamw8bit", 6, 1, False),
        ("8bit-bs8-nockpt", "adamw8bit", 8, 1, False),
        ("8bit-bs12-nockpt", "adamw8bit", 12, 1, False),
        ("8bit-bs4-ckpt", "adamw8bit", 4, 2, True),
    ]:
        try:
            results.append(run_cfg(name, ok, bs, ga, 1024, gc))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            log(f"\n── {name}: ❌ {type(e).__name__} {str(e)[:100]}")
            results.append({"cfg": name, "opt": ok, "bs": bs, "oom": True})
            torch.cuda.empty_cache()

    lora = check_lora()
    exp = check_export()

    ok = [r for r in results if r.get("fits")]
    best = max(ok, key=lambda r: r["tok_s"]) if ok else None
    summary = {"arch": {"d": cfg.hidden_size, "L": cfg.num_hidden_layers,
                        "heads": cfg.num_attention_heads,
                        "kv_heads": cfg.num_key_value_heads,
                        "ffn": cfg.intermediate_size, "vocab": cfg.vocab_size},
               "configs": results, "lora": lora, "export": exp, "recommended": best}
    log("\n" + "=" * 70)
    if best:
        log(f"✅ 推荐: {best['cfg']} → {best['tok_s']:,} tok/s, "
            f"reserved {best['mem_reserved_gib']} GiB, 1B token ≈ {best['hours_1b_tok']} h")
    else:
        log("❌ 无可行配置")
    log(f"LoRA: Qwen 同名注入 {'✅' if lora['peft_named_like_qwen'] else '❌'} {lora['injected']}")
    log(f"导出重载一致: {exp['reload_identical']}")
    with open(os.path.join(TMP, "smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    log(f"摘要 → {os.path.join(TMP, 'smoke_summary.json')}")
