# -*- coding: utf-8 -*-
"""Step 3 —— 秩截断热蒸馏（方案 E1c 的真实模型版）。

教师适配器（r=16）每个 LoRA 模块 ΔW = (alpha/r)·B@A，秩 ≤16。
对 ΔW 求（低秩）SVD，截到 k∈{8,4,2}，用 A'=√S_k·Vh_k、B'=U_k·√S_k
精确重构为新适配器（lora_alpha 设为 k 使 PEFT 缩放=1，Δ 逐位等于截断和）。
产出：distilled/adapter_svd_r{8,4,2} —— 体积递减的 L1 常驻候选。
实现注：ΔW 数学秩 ≤16，故用 svd_lowrank(q=k+4) 做随机化精确低秩分解，
比全量 cuSOLVER SVD 快若干数量级（后者在本机 11008×2048 上超时）。
"""
import os, json, torch
from safetensors.torch import load_file, save_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEACHER = os.path.join(ROOT, "teacher", "adapter")
OUT = os.path.join(ROOT, "distilled")

def truncate_to_rank(state, new_r, scale=1.0, dev="cuda:0"):
    out = {}
    keys = sorted({k.split(".lora_A.")[0] for k in state if ".lora_A." in k})
    ref = state[f"{keys[0]}.lora_A.weight"]
    for base in keys:
        A = state[f"{base}.lora_A.weight"].to(dev).float()   # r × d_in
        B = state[f"{base}.lora_B.weight"].to(dev).float()   # d_out × r
        dW = scale * (B @ A)                                 # 含教师 alpha/r 缩放
        k = min(new_r, A.shape[0])
        U, S, V = torch.svd_lowrank(dW, q=min(k + 4, min(dW.shape)))  # dW ≈ U diag(S) Vᵀ
        # 取前 k 个（svd_lowrank 按降序）
        Sk = S[:k].clamp(min=0)
        root = torch.sqrt(Sk)
        A2 = (torch.diag(root) @ V[:, :k].T)                 # k × d_in
        B2 = U[:, :k] @ torch.diag(root)                     # d_out × k
        out[f"{base}.lora_A.weight"] = A2.to("cpu").to(ref.dtype)
        out[f"{base}.lora_B.weight"] = B2.to("cpu").to(ref.dtype)
    return out

def write_adapter(new_r, src_state, scale, dev="cuda:0"):
    st = truncate_to_rank(src_state, new_r, scale, dev)
    cfg = {
        "peft_type": "LORA", "base_model_name_or_path":
            r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct",
        "task_type": "CAUSAL_LM", "r": new_r, "lora_alpha": new_r, "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_to_save": None, "fan_in_fan_out": False, "init_lora_weights": False,
    }
    d = os.path.join(OUT, f"adapter_svd_r{new_r}")
    os.makedirs(d, exist_ok=True)
    save_file(st, os.path.join(d, "adapter_model.safetensors"))
    with open(os.path.join(d, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    n = sum(p.numel() for p in st.values())
    return d, n, st

def main():
    src = load_file(os.path.join(TEACHER, "adapter_model.safetensors"))
    tc = json.load(open(os.path.join(TEACHER, "adapter_config.json")))
    r0, scale = int(tc["r"]), float(tc["lora_alpha"]) / int(tc["r"])
    print("teacher r =", r0, " alpha/r =", scale, flush=True)
    keys = sorted({k.split(".lora_A.")[0] for k in src if ".lora_A." in k})
    for new_r in [8, 4, 2]:
        if new_r >= r0:
            continue
        d, n, st = write_adapter(new_r, src, scale, dev="cuda:0")
        # 重构能量误差自检（抽样 3 个模块即可，全 GPU 计算）
        errs, tots = 0.0, 0.0
        for k2 in keys[:: max(1, len(keys) // 3)][:3]:
            A, B = src[f"{k2}.lora_A.weight"].cuda().float(), src[f"{k2}.lora_B.weight"].cuda().float()
            dW = scale * (B @ A)
            A2, B2 = st[f"{k2}.lora_A.weight"].cuda().float(), st[f"{k2}.lora_B.weight"].cuda().float()
            errs += float(((dW - B2 @ A2) ** 2).sum()); tots += float((dW ** 2).sum())
        print(f"rank {new_r}: params {n/1e6:.2f}M  截断能量误差(抽样) {errs/tots:.4f}  -> {d}", flush=True)

if __name__ == "__main__":
    main()
