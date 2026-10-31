# -*- coding: utf-8 -*-
"""单卡 4B 训练可行性实测台。

用一个与 Qwen3-4B 完全同构的随机权重模型（hidden 2560 / 36 层 / GQA 32:8 / head_dim 128 /
vocab 151936 / tied embedding → 4.022B 参数）在**单张 22GB 2080 Ti** 上实测各训练方案的
峰值显存与吞吐。随机权重对显存/速度没有影响（只影响 loss 数值）。

模式：
  full_fp32opt   全参微调，AMP fp16 + 原生 AdamW(fp32 状态) + 梯度检查点   ← 预期 OOM
  full_8bitopt   全参微调，AMP fp16 + bitsandbytes AdamW8bit + 梯度检查点
  lora           LoRA r16 全线性 + 梯度检查点
  lora_nockpt    LoRA r16，不开梯度检查点（看激活占多少）
  qlora          NF4 4bit + LoRA + 梯度检查点（需要 --ckpt-dir，先跑 --mode make_ckpt）

每个模式在独立子进程里跑（--all 驱动），一个 OOM 不影响其它。
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import torch


def make_config():
    from transformers import Qwen3Config
    return Qwen3Config(
        vocab_size=151936, hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, tie_word_embeddings=True,
    )


def log(*a):
    print(*a, flush=True)


def gpu_used_gib():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 30


def build_fp16(device="cuda:0", dtype=torch.float16):
    """构造 4B 同构模型。

    `no_init_weights` 跳过随机初始化（4B 元素的 randn 在 CPU 上要几十秒，对显存/速度测量无影响）。
    """
    from transformers import Qwen3ForCausalLM
    from transformers.initialization import no_init_weights   # transformers 5.x 的新位置
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(0)
        with no_init_weights():
            m = Qwen3ForCausalLM(make_config())
    finally:
        torch.set_default_dtype(old)
    m = m.to(device)
    with torch.no_grad():          # 在 GPU 上填充，比 CPU 初始化快一个量级
        for p in m.parameters():
            p.normal_(0, 0.02)
    m.tie_weights()                # no_init_weights 会丢掉 tied 关系，必须补回（否则多算 0.39B）
    return m


def build_lora(m, r=16):
    from peft import LoraConfig, TaskType, get_peft_model
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=2 * r, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(m, lc)


def chunked_ce_loss(m, ids, labels, chunk=512):
    """分块交叉熵：把 lm_head + CE 按序列维切块算。

    4B 模型的词表 151936，logits = bs×seq×151936×2 字节 —— bs4×2048 就是 2.5 GiB，
    CE 内部再 upcast 到 fp32 变 5 GiB 单个分配，这才是 22GB 卡上真正撞墙的地方。
    分块后峰值只与 chunk×vocab 有关，与 batch 无关。
    """
    import torch.nn.functional as F
    dec = m.get_decoder() if hasattr(m, "get_decoder") else None
    if dec is None:                       # 兜底：剥掉 peft 外壳
        dec = m.base_model.model.model
    h = dec(input_ids=ids, use_cache=False).last_hidden_state
    head = m.get_output_embeddings()
    total, n = None, 0
    for i in range(0, h.shape[1], chunk):
        lg = head(h[:, i:i + chunk]).float()
        y = labels[:, i:i + chunk].reshape(-1)
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y, reduction="sum")
        total = loss if total is None else total + loss
        n += y.numel()
    return total / n


def run(args):
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = "cuda:0"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mode = args.mode
    if mode == "make_ckpt":
        os.makedirs(args.ckpt_dir, exist_ok=True)
        log(f"[make_ckpt] 生成 4B 同构随机权重 → {args.ckpt_dir}（约 7.5GB，仅本次需要）")
        m = build_fp16("cpu")
        m.save_pretrained(args.ckpt_dir, safe_serialization=True)
        log("[make_ckpt] 完成")
        return {"mode": mode, "ok": True, "ckpt": args.ckpt_dir}

    # ---------- 构造模型 ----------
    quantized = mode == "qlora"
    if quantized:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(args.ckpt_dir, dtype=torch.float16,
                                                 quantization_config=bnb, device_map={"": 0})
    else:
        m = build_fp16(dev, torch.float32 if args.param_dtype == "fp32" else torch.float16)

    use_ckpt = mode in ("full_fp32opt", "full_8bitopt", "lora", "qlora") or args.grad_ckpt

    if quantized:
        # 顺序很重要：先做 k-bit 准备（含梯度检查点），再挂 LoRA，
        # 否则 prepare_model_for_kbit_training 会把 LoRA 权重一起 upcast 到 fp32。
        from peft import prepare_model_for_kbit_training
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=use_ckpt)
        m = build_lora(m, r=args.lora_r)
    elif mode in ("lora", "lora_nockpt"):
        m = build_lora(m, r=args.lora_r)

    if use_ckpt and not quantized:
        m.config.use_cache = False
        m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.ckpt_mode == "reentrant"})
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
    m.train()

    params = [p for p in m.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in m.parameters())
    log(f"[{mode}] 参数 {n_all/1e9:.3f}B，可训练 {n_train/1e6:.2f}M（{n_train/n_all*100:.3f}%）"
        f"，梯度检查点 {'开' if use_ckpt else '关'}")

    # ---------- 优化器 ----------
    if mode == "full_fp32opt":
        opt = torch.optim.AdamW(params, lr=1e-5)
    elif mode == "full_8bitopt":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=1e-5)
    else:
        opt = torch.optim.AdamW(params, lr=2e-4)

    # ⚠️ GradScaler 只能配 fp32 参数（AMP 标准用法）。参数本身是 fp16 时 gradient 也是 fp16，
    # scaler 会直接抛 "Attempting to unscale FP16 gradients"，所以那一路不挂 scaler。
    use_scaler = (args.amp == "fp16" and args.param_dtype == "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ---------- 跑几步 ----------
    bs, seq = args.bs, args.seq
    ids = torch.randint(0, 151936, (bs, seq), device=dev)
    labels = ids.clone()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    step_times = []
    try:
        for i in range(args.steps):
            torch.cuda.synchronize()
            ts = time.time()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(args.amp == "fp16")):
                if args.chunk_ce:
                    loss = chunked_ce_loss(m, ids, labels, args.chunk_size)
                else:
                    out = m(input_ids=ids, labels=labels)
                    loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i == 0:
                log(f"[{mode}] 第 1 步完成，峰值 allocated "
                    f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
            if i > 0:                     # 第一步含 cudnn/编译预热，不计
                step_times.append(time.time() - ts)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and not isinstance(e, torch.OutOfMemoryError):
            raise
        peak_a = torch.cuda.max_memory_allocated() / 2**30
        peak_r = torch.cuda.max_memory_reserved() / 2**30
        log(f"[{mode}] ❌ OOM：峰值 allocated {peak_a:.2f} / reserved {peak_r:.2f} GiB")
        return {"mode": mode, "ok": False, "oom": True, "bs": bs, "seq": seq,
                "peak_alloc_gib": round(peak_a, 2), "peak_reserved_gib": round(peak_r, 2),
                "trainable_m": round(n_train / 1e6, 2),
                "err": str(e).split("\n")[0][:200]}

    peak_a = torch.cuda.max_memory_allocated() / 2**30
    peak_r = torch.cuda.max_memory_reserved() / 2**30
    step_s = sum(step_times) / len(step_times) if step_times else float("nan")
    tps = bs * seq / step_s if step_s == step_s else 0
    log(f"[{mode}] ✅ bs{bs}×seq{seq}：峰值 allocated {peak_a:.2f} / reserved {peak_r:.2f} GiB"
        f" | {step_s*1000:.0f} ms/步 | {tps:,.0f} tok/s"
        f" | 可用 {gpu_used_gib():.1f} GiB 已占")

    out = {"mode": mode, "ok": True, "bs": bs, "seq": seq,
           "peak_alloc_gib": round(peak_a, 2), "peak_reserved_gib": round(peak_r, 2),
           "weights_gib_equiv": round(n_all * (0.5 if quantized else 2) / 2**30, 2),
           "trainable_m": round(n_train / 1e6, 2),
           "step_ms": round(step_s * 1000, 1), "tok_per_s": round(tps),
           "chunk_ce": args.chunk_ce,
           "grad_ckpt": use_ckpt}
    del m, opt
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--amp", default="fp16", choices=["fp16", "none"])
    ap.add_argument("--param-dtype", default="fp16", choices=["fp16", "fp32"],
                    help="模型参数本身的精度。fp32 = AMP 标准用法（带 GradScaler + fp32 主权重）")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--ckpt-mode", default="nonreentrant", choices=["nonreentrant", "reentrant"],
                    help="梯度检查点实现：nonreentrant=use_reentrant False，reentrant=旧实现")
    ap.add_argument("--chunk-ce", action="store_true", help="分块交叉熵（省掉词表级 logits 峰值）")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--ckpt-dir", default=os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_tmp", "qwen3_4b_rand")))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = run(args)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
