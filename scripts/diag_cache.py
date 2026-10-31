# -*- coding: utf-8 -*-
"""KV cache 守卫：三件事必须同时成立。

  1. 训练路径不变：同一 batch 的 loss 与打补丁前的模型（model_l0_bak_tmp.py）**逐位一致**
  2. 解码等价：带 cache 的 greedy 与不带 cache 的 greedy 产出**完全相同的 token 序列**
  3. 真的变快：32 token 的解码耗时对比
"""
import os, sys, importlib.util, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, DEFAULT_L0
from model_l0 import SedimentConfig
import model_l0 as new_mod

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:0"

# ---- 载入打补丁前的模块做对照 ----
spec = importlib.util.spec_from_file_location("model_l0_bak", os.path.join(HERE, "model_l0_bak_tmp.py"))
bak = importlib.util.module_from_spec(spec); spec.loader.exec_module(bak)

from safetensors.torch import load_file
cfg = new_mod.SedimentConfig.from_pretrained(DEFAULT_L0)
sd = load_file(os.path.join(DEFAULT_L0, "model.safetensors"))

def build(mod):
    m = mod.SedimentBase(mod.SedimentConfig.from_pretrained(DEFAULT_L0))
    m.load_state_dict(sd, strict=False); m.tie_weights(); m.to(torch.float16).to(DEV); m.eval()
    return m

m_new, tok = load_l0(dtype="float16", device=DEV)
m_bak = build(bak)

print("=" * 66)
print("1) 训练路径一致性（同 batch loss 必须逐位相同）")
torch.manual_seed(0)
ids = torch.randint(0, 1000, (2, 128), device=DEV)
lb = ids.clone()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
    l_new = m_new(ids, labels=lb).loss.item()
    l_bak = m_bak(ids, labels=lb).loss.item()
print(f"   patched {l_new:.8f} | bak {l_bak:.8f} | 差 {abs(l_new-l_bak):.2e} "
      f"→ {'✅ 一致' if abs(l_new-l_bak) < 1e-6 else '❌ 训练路径被改变'}")

print("\n2) 解码等价性（cache vs 无 cache，同一 prompt 12 token）")
text = tok.apply_chat_template(
    [{"role": "system", "content": "你是精通Sedu语与中文互译的语言助手。"},
     {"role": "user", "content": "把这句Sedu语翻译成中文：numo ke toto"}],
    tokenize=False, add_generation_prompt=True)
ids_in = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
am = torch.ones_like(ids_in)

with torch.no_grad():
    o_cache = m_new.generate(ids_in, attention_mask=am, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.pad_token_id, repetition_penalty=1.0, use_cache=True)
    o_nocache = m_new.generate(ids_in, attention_mask=am, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.pad_token_id, repetition_penalty=1.0, use_cache=False)
a = o_nocache[0][ids_in.shape[1]:].tolist()
b = o_cache[0][ids_in.shape[1]:].tolist()
print(f"   无 cache {a}")
print(f"   有 cache {b}")
print(f"   → {'✅ 逐 token 相同' if a == b else '❌ 不一致'}")
print(f"   解码：{tok.decode(b, skip_special_tokens=True)!r}")

print("\n3) 速度（32 token，含 prefill）")
for tag, uc in (("无 cache", False), ("有 cache", True)):
    with torch.no_grad():
        m_new.generate(ids_in, attention_mask=am, max_new_tokens=4, do_sample=False,
                       pad_token_id=tok.pad_token_id, use_cache=uc)      # 预热
        t0 = time.time()
        m_new.generate(ids_in, attention_mask=am, max_new_tokens=32, do_sample=False,
                       pad_token_id=tok.pad_token_id, use_cache=uc)
        ms = (time.time() - t0) * 1000
    print(f"   {tag}: {ms:.0f} ms  ({ms/32:.0f} ms/token)")

print("\n4) 长上下文受益（512 token prefill + 64 解码）")
long_ids = torch.randint(0, 150000, (1, 512), device=DEV)
lam = torch.ones_like(long_ids)
for tag, uc in (("无 cache", False), ("有 cache", True)):
    with torch.no_grad():
        t0 = time.time()
        m_new.generate(long_ids, attention_mask=lam, max_new_tokens=64, do_sample=False,
                       pad_token_id=tok.pad_token_id, use_cache=uc)
        ms = (time.time() - t0) * 1000
    print(f"   {tag}: {ms:.0f} ms")
print("\n守卫结束")
