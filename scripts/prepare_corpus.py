# -*- coding: utf-8 -*-
"""Step A —— 语料下载 + 分词打包（后台长杆任务）。

datasets 库被系统策略拦截（_xxhash DLL）→ 直接 pyarrow 读 parquet。
产出 corpus/tokens.bin：uint32 小端连续 token 流（Qwen2.5 词表，151936），
训练时用 np.memmap 零拷贝随机切片，无需 datasets/流式解码。

规模：3 个分片 × ≈354M token ≈ 1.06B token（4.2GB bin）。
"""
import os, sys, time, glob
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RAW = os.path.join(ROOT, "corpus_raw")
OUT = os.path.join(ROOT, "corpus")
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)

REPO = "opencsg/chinese-fineweb-edu-v2"
FILES = ["data/00098.parquet", "data/00126.parquet", "data/00146.parquet"]
TOK_PATH = r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct"
MIN_CHARS, MAX_CHARS = 200, 20000     # 质量过滤：过短/过长丢弃
TARGET_TOKENS = 1_100_000_000

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------- 1. 下载 ----------
paths = []
for f in FILES:
    p = os.path.join(RAW, os.path.basename(f))
    if os.path.exists(p) and os.path.getsize(p) > 5e8:
        log(f"已存在 {os.path.basename(f)} ({os.path.getsize(p)/1e6:.0f} MB)")
        paths.append(p); continue
    t0 = time.time()
    lp = hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset", local_dir=RAW)
    # hf_hub_download 在 local_dir 下按原路径存放
    real = lp if os.path.exists(lp) else glob.glob(os.path.join(RAW, "**", os.path.basename(f)), recursive=True)[0]
    paths.append(real)
    log(f"下载 {os.path.basename(f)} {os.path.getsize(real)/1e6:.0f} MB  {time.time()-t0:.0f}s")

# ---------- 2. 分词打包 ----------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOK_PATH)
log(f"tokenizer 词表 {tok.vocab_size}")

bin_path = os.path.join(OUT, "tokens.bin")
meta_path = os.path.join(OUT, "meta.json")
CHUNK_ROWS = 4000
written = 0
kept = 0
dropped = 0
fh = open(bin_path, "wb")
t_start = time.time()
try:
    for pi, p in enumerate(paths):
        pf = pq.ParquetFile(p)
        log(f"分词 {os.path.basename(p)}  rows={pf.metadata.num_rows}")
        for batch in pf.iter_batches(batch_size=CHUNK_ROWS, columns=["text"]):
            texts = [t for t in batch.column(0).to_pylist()
                     if t and MIN_CHARS <= len(t) <= MAX_CHARS]
            dropped += batch.num_rows - len(texts)
            if not texts:
                continue
            enc = tok(texts, add_special_tokens=False)["input_ids"]
            flat = np.fromiter((t for e in enc for t in e), dtype=np.uint32)
            fh.write(flat.tobytes())
            written += flat.size
            kept += len(texts)
            if kept % 20000 < CHUNK_ROWS:
                el = time.time() - t_start
                log(f"  {written/1e6:7.1f}M token  {kept:7d} 篇  "
                    f"{written/el/1e6:.2f}M tok/s  剩 {(TARGET_TOKENS-written)/max(written/el,1)/3600:.2f}h")
            if written >= TARGET_TOKENS:
                break
        if written >= TARGET_TOKENS:
            log("达到目标 token 数，停止")
            break
finally:
    fh.close()

import json
meta = {"tokens": int(written), "docs_kept": int(kept), "docs_dropped": int(dropped),
        "vocab": int(tok.vocab_size), "dtype": "uint32", "bytes": os.path.getsize(bin_path),
        "tokenizer": TOK_PATH, "source_repo": REPO, "source_files": [os.path.basename(p) for p in paths],
        "min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
        "elapsed_sec": round(time.time() - t_start, 1)}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
log("完成:", json.dumps(meta, ensure_ascii=False))
