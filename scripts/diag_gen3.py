# -*- coding: utf-8 -*-
"""生成机制诊断 3 —— 一次判定：attention_mask 是否改变最后位置 logits。"""
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0

DEV = "cuda:0"
m, tok = load_l0(dtype="float16", device=DEV)
ids = tok("人工智能的核心挑战在于", return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
print("prompt:", ids.tolist(), tok.decode(ids[0]))

with torch.no_grad():
    l_no = m(ids).logits[0, -1].float()
    l_am = m(ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()

def top(l, n=5):
    v, i = torch.topk(l, n)
    return [(tok.decode([x]), round(float(y), 4)) for y, x in zip(v.tolist(), i.tolist())]

print("\n无 mask      top5:", top(l_no), "argmax=", int(l_no.argmax()))
print("全 1 mask    top5:", top(l_am), "argmax=", int(l_am.argmax()))
print("两者最大绝对差:", float((l_no - l_am).abs().max()), "| logits 尺度:", float(l_no.abs().max()))
print("argmax 处 logit: 无mask", round(float(l_no[l_no.argmax()]), 4),
      "| mask版同位置", round(float(l_am[l_no.argmax()]), 4))

# 再看 attention_mask 里 mask 的构造是否真的等价于因果
T = ids.shape[1]
am = torch.ones_like(ids)[:, None, None, :].to(torch.float16)
add = (1.0 - am) * torch.finfo(torch.float16).min
print("\nadditive mask 形状", tuple(add.shape), "唯一值", torch.unique(add).tolist())
print("→ is_causal 会被置为:", add is None)

# 直接对比 SDPA：零 mask + is_causal=False vs is_causal=True
q = torch.randn(1, 16, T, 96, device=DEV, dtype=torch.float16)
k = torch.randn(1, 16, T, 96, device=DEV, dtype=torch.float16)
v = torch.randn(1, 16, T, 96, device=DEV, dtype=torch.float16)
z = torch.zeros(1, 1, 1, T, device=DEV, dtype=torch.float16)
a1 = F.scaled_dot_product_attention(q, k, v, attn_mask=z, is_causal=False)
a2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("\nSDPA 零mask/非因果 与 因果 的最大差（最后位置）:",
      float((a1[:, :, -1] - a2[:, :, -1]).abs().max()),
      "| 全位置:", float((a1 - a2).abs().max()))
