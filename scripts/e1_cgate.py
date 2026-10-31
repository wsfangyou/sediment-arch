# -*- coding: utf-8 -*-
"""C 门复核：用验证集 24×bs4（≈98k token）而不是单段 ppl，判断技能训练对通用分布的伤害。"""
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
    vl = float(np.mean(ls)); return vl, math.exp(min(vl, 20))

stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)
register()
base, tok = load_l0(dtype="float32", device=DEV)
l0, p0 = val_ppl(base, stream)
print(f"裸基座         val_loss {l0:.4f} ppl {p0:,.2f}", flush=True)
from peft import PeftModel
pm = PeftModel.from_pretrained(base, os.path.join(ROOT, "l0_acceptance", "adapter_sedu_r16"))
mg = pm.merge_and_unload()
l1, p1 = val_ppl(mg, stream)
print(f"挂 Sedu 技能后  val_loss {l1:.4f} ppl {p1:,.2f}  → 退化 {p1/p0:.4f}×（门限 1.02）", flush=True)
out = {"bare": {"val_loss": round(l0,4), "ppl": round(p0,2)},
       "with_skill": {"val_loss": round(l1,4), "ppl": round(p1,2)},
       "regress_ratio": round(p1/p0, 4), "n_batches": 24, "batch": 4, "seq": 1024,
       "note": "验证集 98k token 口径，替代单段落 ppl（后者的 1.043 属噪声区间）"}
p = os.path.join(ROOT, "l0_acceptance", "e1_cgate.json")
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("→", p)
