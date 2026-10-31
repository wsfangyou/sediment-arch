# -*- coding: utf-8 -*-
"""用真实的本地 Qwen2.5-3B-Instruct 复核实测台的可信度（合成 4B 数字是否可外推）。

只跑 3 步，量三件事：能否加载、峰值显存、每步耗时。
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda:0"
MODEL = "D:/Doubao-works/训练前置/02_基座模型/Qwen2.5-3B-Instruct"
BS, SEQ, STEPS, CHUNK = 2, 2048, 4, 512

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV)
n_all = sum(p.numel() for p in m.parameters())
m = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                                 lora_dropout=0.05,
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"]))
m.config.use_cache = False
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m.enable_input_require_grads()
m.train()
n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"[真实 3B] 参数 {n_all/1e9:.3f}B，可训练 {n_tr/1e6:.2f}M，fp16 权重 {n_all*2/2**30:.2f} GiB",
      flush=True)

opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
ids = torch.randint(0, tok.vocab_size, (BS, SEQ), device=DEV)
torch.cuda.reset_peak_memory_stats()
ts = []
for i in range(STEPS):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        h = m.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state
        head = m.get_output_embeddings()
        tot, n = None, 0
        for j in range(0, SEQ, CHUNK):
            lg = head(h[:, j:j + CHUNK]).float()
            y = ids[:, j:j + CHUNK].reshape(-1)
            l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y, reduction="sum")
            tot = l if tot is None else tot + l
            n += y.numel()
        loss = tot / n
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if i:
        ts.append(time.time() - t0)
peak = torch.cuda.max_memory_allocated() / 2 ** 30
step_s = sum(ts) / len(ts)
print(f"[真实 3B] bs{BS}×seq{SEQ} 峰值 {peak:.2f} GiB | {step_s*1000:.0f} ms/步 | "
      f"{BS*SEQ/step_s:,.0f} tok/s", flush=True)
