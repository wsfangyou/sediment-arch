# -*- coding: utf-8 -*-
"""Step B —— Sediment-Base (L0 中心模型) 预训练。

数值配方（Turing sm75 实测决定）：
· 参数 fp32 主权重 + autocast fp16 计算 + GradScaler
  —— 纯 fp16 参数会让 AdamW 二阶矩下溢；fp32 主权重在 22.5GB 上可容纳，
     且 fp16 矩阵乘仍吃到 Turing 张量核。
· 双卡 DDP 在本机不可用（Windows 无 NCCL；gloo 同步 1.6GB fp16 梯度
  实测需 10–30s/步，而计算仅 0.27s/步）→ 单卡训练，第二张卡闲置。

长任务保障：每 ckpt_every 步保存 model+optim+scheduler+step+rng，
支持 --resume 断点续训（27 小时级任务必须可恢复）。
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_l0 import SedimentBase, Config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CORPUS = os.path.join(ROOT, "corpus")
CKPT = os.path.join(ROOT, "l0_checkpoints")
FINAL = os.path.join(ROOT, "l0_base")
os.makedirs(CKPT, exist_ok=True)

DEV = "cuda:0"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class TokenStream:
    """np.memmap 零拷贝随机切片；末尾 5% 留作验证集（永不参与训练）。"""

    def __init__(self, path, seq, seed=0, val_frac=0.05):
        self.buf = np.memmap(path, dtype=np.uint32, mode="r")
        self.seq = seq
        self.n = len(self.buf)
        self.split = int(self.n * (1 - val_frac))
        self.rng = np.random.default_rng(seed)
        log(f"token 流: 总 {self.n/1e6:.1f}M | 训练 {self.split/1e6:.1f}M | 验证 {(self.n-self.split)/1e6:.1f}M")

    def batch(self, bs, train=True, rng=None):
        hi = self.split - self.seq - 1 if train else self.n - self.seq - 1
        lo = 0 if train else self.split
        r = rng or self.rng
        idx = r.integers(lo, hi, size=bs)
        x = np.stack([self.buf[i:i + self.seq] for i in idx]).astype(np.int64)
        y = np.stack([self.buf[i + 1:i + self.seq + 1] for i in idx]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def train_tokens(self):
        return self.split


@torch.no_grad()
def evaluate(model, stream, bs, steps=8, rng=None):
    model.eval()
    rng = rng or np.random.default_rng(999)
    tot = 0.0
    for _ in range(steps):
        x, y = stream.batch(bs, train=False, rng=rng)
        x, y = x.to(DEV), y.to(DEV)
        with autocast("cuda", dtype=torch.float16):
            loss = model(x, y).loss
        tot += loss.item()
    model.train()
    return tot / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=2, help="梯度累积步数")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 则覆盖 epochs")
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--grad-ckpt", action="store_true", help="梯度检查点（省显存换时间）")
    ap.add_argument("--opt", default="adamw8bit", choices=["adamw8bit", "adamw"],
                    help="优化器：8bit 省 5GB 显存（Turing 上吞吐更好）")
    ap.add_argument("--resume", default="")
    ap.add_argument("--smoke", type=int, default=0, help=">0 只跑 N 步做冒烟验证后退出")
    args = ap.parse_args()

    bin_path = os.path.join(CORPUS, "tokens.bin")
    if not os.path.exists(bin_path):
        log("缺少 tokens.bin，请先运行 prepare_corpus.py"); sys.exit(1)

    stream = TokenStream(bin_path, args.seq)
    cfg = Config()
    model = SedimentBase(cfg).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    kv_gqa = model.kv_bytes_per_token()
    kv_mha = 2 * cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim * 2
    log(f"模型参数 {nparam/1e9:.3f}B | KV(GQA {cfg.num_key_value_heads}头) {kv_gqa} B/token "
        f"= 同规模 MHA 的 {kv_gqa/kv_mha:.2f}×（为方案 §3 竞价区省 {100*(1-kv_gqa/kv_mha):.0f}% KV 预算）")
    if args.grad_ckpt:
        log("梯度检查点已启用（重算换显存）")
    model.train()

    decay, nodecay = [], []
    for n_, p in model.named_parameters():
        (nodecay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": args.wd},
              {"params": nodecay, "weight_decay": 0.0}]
    if args.opt == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=args.lr, betas=(0.9, 0.95))
        log("优化器: bitsandbytes AdamW8bit（m/v 状态 6.6GB→1.66GB）")
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))
        log("优化器: torch AdamW (fp32)")
    scaler = GradScaler("cuda")

    eff_bs = args.bs * args.ga
    if args.max_steps > 0:
        total = args.max_steps
    else:
        total = int(args.epochs * stream.train_tokens / (eff_bs * args.seq))
    log(f"有效 batch {eff_bs}×{args.seq} = {eff_bs*args.seq:,} token/step | 总步数 {total:,} "
        f"| 语料 {stream.train_tokens/1e9:.2f}B token × {args.epochs}")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, total - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    step0 = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); step0 = ck["step"]
        torch.set_rng_state(ck["rng"])
        log(f"续训自 {args.resume} step={step0}")

    hist = []
    t0 = time.time(); tlog = time.time(); run_loss = 0.0; run_n = 0
    for step in range(step0, total):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.ga):
            x, y = stream.batch(args.bs)
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with autocast("cuda", dtype=torch.float16):
                loss = model(x, y, use_ckpt=args.grad_ckpt).loss
            scaler.scale(loss / args.ga).backward()
            acc += loss.item() / args.ga
        scaler.unscale_(opt)
        gn = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], args.clip)
        scaler.step(opt); scaler.update()
        run_loss += acc; run_n += 1

        if (step + 1) % 20 == 0:
            el = time.time() - tlog
            sps = 20 * eff_bs * args.seq / el
            done = step + 1 - step0
            eta = (total - step - 1) / max(done / max(time.time() - t0, 1e-6), 1e-9) / 3600
            log(f"step {step+1:,}/{total:,} loss {run_loss/run_n:.4f} ppl {math.exp(min(run_loss/run_n,20)):.2f} "
                f"lr {lr:.2e} gn {float(gn):.2f} {sps:,.0f} tok/s ETA {eta:.1f}h "
                f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
            hist.append({"step": step + 1, "loss": round(run_loss / run_n, 4),
                         "tok_s": round(sps), "lr": lr, "eta_h": round(eta, 2)})
            run_loss = 0.0; run_n = 0; tlog = time.time()

        if (step + 1) % args.eval_every == 0:
            vl = evaluate(model, stream, args.bs)
            log(f"  ↳ 验证 loss {vl:.4f} ppl {math.exp(min(vl,20)):.2f}")
            hist.append({"step": step + 1, "val_loss": round(vl, 4)})

        if args.smoke and step + 1 >= args.smoke:
            log(f"冒烟模式完成 {step+1} 步，退出（不保存最终权重）")
            with open(os.path.join(CKPT, "smoke_hist.json"), "w") as f:
                json.dump({"args": vars(args), "params_B": nparam / 1e9, "hist": hist}, f, indent=1)
            return

        if (step + 1) % args.ckpt_every == 0:
            p = os.path.join(CKPT, f"step{step+1:07d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step + 1,
                        "rng": torch.get_rng_state(), "cfg": vars(cfg)}, p)
            log(f"  ↳ 检查点 {p} ({os.path.getsize(p)/1e9:.2f} GB)")
            olds = sorted([f for f in os.listdir(CKPT) if f.startswith("step")])[:-3]
            for f in olds:
                os.remove(os.path.join(CKPT, f))

    # ---- 导出 HF 兼容格式（config.json + safetensors + tokenizer 复制）----
    os.makedirs(FINAL, exist_ok=True)
    model = model.float()
    from safetensors.torch import save_file
    # ⚠️ 不剥离 tied 的 lm_head：transformers 5.8 对未注册自定义类会跳过 tied 声明的
    # embed/lm_head 键（留随机初始化，ppl 实测 67→47万）。全量存独立副本，
    # 并由 loader.load_l0 走"手动 load_state_dict"路径加载（safetensors 拒存共享张量 → clone）。
    sd = {k: v.float().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(FINAL, "model.safetensors"), metadata={"format": "pt"})
    hf_cfg = {
        "architectures": ["SedimentBase"], "model_type": "sediment_base",
        "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers, "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads, "intermediate_size": cfg.intermediate_size,
        "max_position_embeddings": cfg.max_position_embeddings, "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps, "tie_word_embeddings": cfg.tie_word_embeddings,
        "hidden_act": "silu", "torch_dtype": "float32",
        "project_note": "沉积架构 L0 自制中心模型；词表与 LoRA 注入位对齐 Qwen2.5 以支持 MSP 跨骨干转译与技能管线复用",
    }
    with open(os.path.join(FINAL, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_cfg, f, ensure_ascii=False, indent=1)
    import shutil
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "special_tokens_map.json", "generation_config.json"]:
        src = os.path.join(r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FINAL, fn))
    with open(os.path.join(CKPT, "train_hist.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "params_B": nparam / 1e9, "hist": hist,
                   "elapsed_h": round((time.time() - t0) / 3600, 2)}, f, ensure_ascii=False, indent=1)
    log(f"导出完成 → {FINAL}")


if __name__ == "__main__":
    main()
