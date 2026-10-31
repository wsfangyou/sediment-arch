# -*- coding: utf-8 -*-
"""本机真实 I/O 带宽实测（RTX 2080 Ti），用于校准 hw.py 的 6GB 笔记本模型。
测 H2D(pinned) 与 VRAM 内部读写带宽。"""
import time, torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from hw import H

def bench():
    dev = "cuda"
    torch.cuda.init()
    print(f"GPU: {torch.cuda.get_device_name(0)} ×{torch.cuda.device_count()}")
    res = {}
    for mb in [100, 300, 1024, 2048]:
        n = mb * 1024 * 1024 // 4
        host = torch.empty(n, dtype=torch.float32, pin_memory=True)
        host.normal_()
        # 预热
        d = torch.empty(n, dtype=torch.float32, device=dev); d.copy_(host)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            d.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        res[mb] = mb / 1024 / dt
        del d, host
    print("H2D pinned 带宽 (GB/s):", {k: round(v, 2) for k, v in res.items()})
    # VRAM 内部：a+=b 每元素读2写1 → 流量=3×n×4B
    a = torch.zeros(512 * 1024 * 1024 // 4, device=dev); b = torch.ones_like(a)
    for _ in range(3): a += b
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): a += b
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
    vram_bw = a.numel() * 4 * 3 / dt / 1e9
    print(f"VRAM 有效读写带宽 ≈ {vram_bw:.0f} GB/s")
    print(f"\n对照模型假设：bw_pcie={H.bw_pcie}GB/s（x8 笔记本保守端）、bw_vram={H.bw_vram}GB/s")
    print("→ 本机实测可换算：会话级加载 300MB 在本机 ≈", round(300 / 1024 / res[300] * 1000, 1), "ms")
    return res, vram_bw

if __name__ == "__main__":
    bench()
