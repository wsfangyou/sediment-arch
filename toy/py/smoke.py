# -*- coding: utf-8 -*-
import time, sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (ToyLM, make_skill_chains, mix_batch, sample_batch, attach_lora,
                    train_steps, next_token_acc, DEV, LoraLinear)

t0 = time.time()
chains = make_skill_chains(8)
rng = np.random.default_rng(0)
model = ToyLM(seed=0).to(DEV)
train_steps(model, lambda: mix_batch(chains, 32, rng=rng), 300, lr=3e-3)
print(f"pretrain 300 steps: {time.time()-t0:.1f}s  device={DEV}")
for i, c in enumerate(chains[:3]):
    print(f"chain{i} 骨干裸 acc = {next_token_acc(model, c):.3f}")

t1 = time.time()
m2 = ToyLM(seed=0).to(DEV)
m2.load_state_dict(model.state_dict())
model2 = m2
m2, lora = attach_lora(m2, 0, r=16, seed=0)
train_steps(m2, lambda: sample_batch(chains[0], 32, rng=rng), 300, lr=3e-3)
print(f"lora 300 steps: {time.time()-t1:.1f}s")
print(f"chain0 LoRA 后 acc = {next_token_acc(m2, chains[0]):.3f}")
