# -*- coding: utf-8 -*-
"""L0 fp16 导出 —— 去重 tied 副本，产出 1.66GB 的常驻权重并复测 ppl。

为什么能省 466MB：训练时 embedding 与 lm_head 是 tied（同一张矩阵），fp32 导出为了绕开
transformers 5.8 的 tied 加载缺陷而存了两份副本。fp16 版改为"只存一份 + 声明 tied"，
由 loader.load_l0 手动 tie（该路径已实测正确）。

产物：l0_base_fp16/（config.json tie_word_embeddings=true，无 lm_head 键，其余同）
用法：python export_fp16.py [--verify]
"""
import os, sys, json, shutil, math
import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import DEFAULT_L0

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DST = os.path.join(ROOT, "l0_base_fp16")
GB = 2 ** 30


def log(*a):
    print(*a, flush=True)


def main():
    verify = "--verify" in sys.argv
    os.makedirs(DST, exist_ok=True)
    src = os.path.join(DEFAULT_L0, "model.safetensors")
    sd = load_file(src)
    log(f"源: {src} ({os.path.getsize(src)/GB:.3f} GiB, {len(sd)} 键)")

    key_emb = "embed_tokens.weight"
    key_lm = "lm_head.weight"
    tied = key_lm in sd and torch.equal(sd[key_lm], sd[key_emb])
    log(f"tied 校验: embed == lm_head → {tied}")

    out = {}
    for k, v in sd.items():
        if key_lm in sd and tied and k == key_lm:
            continue                      # 省掉重复副本，由 loader 手动 tie
        out[k] = v.to(torch.float16).contiguous()
    p = os.path.join(DST, "model.safetensors")
    save_file(out, p, metadata={"format": "pt"})
    log(f"导出: {p} ({os.path.getsize(p)/GB:.3f} GiB, {len(out)} 键, fp16)")

    cfg = json.load(open(os.path.join(DEFAULT_L0, "config.json"), encoding="utf-8"))
    cfg["torch_dtype"] = "float16"
    cfg["tie_word_embeddings"] = bool(tied)
    json.dump(cfg, open(os.path.join(DST, "config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "special_tokens_map.json", "generation_config.json"):
        s = os.path.join(DEFAULT_L0, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(DST, fn))
    log(f"config/tokenizer 已复制（tie_word_embeddings={tied}）")

    if verify:
        from loader import load_l0
        from pretrain_l0 import TokenStream
        log("\n复测 ppl（与 fp32 版 56.9 对照）…")
        stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)
        ppls = {}
        for tag, path in (("fp32", DEFAULT_L0), ("fp16", DST)):
            m, _ = load_l0(path=path, dtype="float16")
            m.eval(); rng = np.random.default_rng(7); ls = []
            with torch.no_grad():
                for _ in range(24):
                    x, y = stream.batch(4, train=False, rng=rng)
                    x, y = x.to("cuda:0"), y.to("cuda:0")
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        ls.append(m(x, y).loss.item())
            vl = float(np.mean(ls))
            ppls[tag] = {"val_loss": round(vl, 4), "ppl": round(math.exp(min(vl, 20)), 2),
                         "weight_mib": round(sum(t.numel() for t in m.parameters()) * 2 / 1024 / 1024)}
            log(f"  {tag}: val_loss {vl:.4f} ppl {ppls[tag]['ppl']:,.2f} "
                f"| 权重 {ppls[tag]['weight_mib']} MiB")
            del m; torch.cuda.empty_cache()
        with open(os.path.join(HERE, "..", "l0_acceptance", "fp16_export.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"size_gib": round(os.path.getsize(p) / GB, 3),
                       "size_bytes": os.path.getsize(p), "keys": len(out),
                       "tied": bool(tied), "verify": ppls}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
