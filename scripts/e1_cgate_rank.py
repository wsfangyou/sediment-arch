# -*- coding: utf-8 -*-
"""C 门 × 秩：技能沉积物截到更低秩，能否减轻对通用分布的伤害？

假设（方案 §2.2 隐含假设）：沉积物越小越安全。本脚本在 98k token 验证集上量它。
同时给出「技能能力 vs 通用伤害」的权衡曲线 —— 这是 L1 晋升门在小骨干上的实际形状。
"""
import os, sys, json, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, register, DEFAULT_L0
from pretrain_l0 import TokenStream
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEV = "cuda:0"

def val_ppl(m, stream, n=24):
    m.eval(); rng = np.random.default_rng(7); ls = []
    with torch.no_grad():
        for _ in range(n):
            x, y = stream.batch(4, train=False, rng=rng)
            x, y = x.to(DEV), y.to(DEV)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ls.append(m(x, y).loss.item())
    vl = float(np.mean(ls)); return math.exp(min(vl, 20))

stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)
register()
base, tok = load_l0(dtype="float32", device=DEV)
p0 = val_ppl(base, stream)
print(f"裸基座 ppl {p0:,.2f}", flush=True)
snap = {k: v.detach().float().cpu().clone() for k, v in base.state_dict().items()
        if k.endswith("_proj.weight")}

from peft import PeftModel
pm = PeftModel.from_pretrained(base, os.path.join(ROOT, "l0_acceptance", "adapter_sedu_r16"))
mg = pm.merge_and_unload(); mg.eval()
msd = mg.state_dict()
delta = {k: msd[k].float().cpu() - w for k, w in snap.items() if k in msd and msd[k].shape == w.shape}
svd = {}
for k, d in delta.items():
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    svd[k] = (U[:, :16].contiguous(), S[:16].contiguous(), Vh[:16].contiguous())
print(f"ΔW {len(delta)} 个矩阵，SVD 完成", flush=True)

res = []
for r in (16, 8, 4, 2):
    sd = mg.state_dict()
    for k, (U, S, Vh) in svd.items():
        rr = min(r, len(S))
        sd[k].copy_((snap[k] + (U[:, :rr] * S[:rr]) @ Vh[:rr]).to(sd[k].dtype).to(sd[k].device))
    p = val_ppl(mg, stream)
    res.append({"rank": r, "ppl": round(p, 2), "regress_vs_bare": round(p / p0, 4)})
    print(f"  r={r:<2} 通用 ppl {p:,.2f}  → {p/p0:.4f}×裸基座", flush=True)

out = {"bare_ppl": round(p0, 2), "results": res, "n_batches": 24, "batch": 4, "seq": 1024,
       "note": "同一 adapter_sedu_r16（600 步）截断到各秩；ppl 为验证集 98k token 口径"}
p = os.path.join(ROOT, "l0_acceptance", "e1_cgate_rank.json")
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("→", p)
