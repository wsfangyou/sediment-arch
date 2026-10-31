# -*- coding: utf-8 -*-
"""技能库：基座（只见过链 0–1）+ 6 个技能 LoRA（链 2–7）。
每个技能 = 真实训练的 LoRA ΔW（qkv 3d×d），缓存到 artifacts/。
merge/restore 工具：把 ΔW 真实合并进 blocks[0].attn 权重（挂载/卸载）。"""
import os
import numpy as np
import torch
from common import (ToyLM, make_skill_chains, mix_batch, sample_batch, attach_lora,
                    train_steps, next_token_acc, D_MODEL, DEV, ART_DIR)

BASE_TOKENS = 2          # 基座只学链 0–1 → 链 2–7 对它都是“新技能”
SKILL_CHAINS = range(2, 8)


def get_base_model():
    p = os.path.join(ART_DIR, "base_sed_v2.pt")
    m = ToyLM(seed=0).to(DEV)
    if os.path.exists(p):
        m.load_state_dict(torch.load(p)); return m
    chains = make_skill_chains(8)
    rng = np.random.default_rng(5)
    sub = chains[:BASE_TOKENS]
    train_steps(m, lambda: mix_batch(sub, 32, rng), 300, lr=3e-3)
    torch.save(m.state_dict(), p)
    return m


def ensure_lora_deltas():
    """返回 {skill_idx: delta_cpu(3d×d fp32)}，训练一次后缓存。"""
    p = os.path.join(ART_DIR, "lora_deltas_v2.pt")
    if os.path.exists(p):
        return torch.load(p)
    chains = make_skill_chains(8)
    out = {}
    for i in SKILL_CHAINS:
        m = get_base_model().to(DEV)
        m, lora = attach_lora(m, 0, r=16, seed=100 + i)
        rng = np.random.default_rng(50 + i)
        train_steps(m, lambda: sample_batch(chains[i], 32, rng=rng), 350, lr=3e-3)
        out[i] = lora.delta().detach().cpu()
        acc = next_token_acc(m, chains[i], rng=np.random.default_rng(9))
        print(f"[技能{i}] LoRA 训练后 chain{i} acc={acc:.3f}")
    torch.save(out, p)
    return out


class Merger:
    """把一组 ΔW 真实合并进基座 attn 权重；restore 回原始权重。"""
    def __init__(self, model):
        self.model = model
        self.w0 = model.blocks[0].attn.weight.detach().clone()

    def merge(self, deltas):
        w = self.model.blocks[0].attn.weight
        w.data = self.w0.clone()
        for d in deltas:
            w.data += d.to(DEV)

    def restore(self):
        self.model.blocks[0].attn.weight.data = self.w0.clone()
