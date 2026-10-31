# -*- coding: utf-8 -*-
"""硬件/带宽数学模型（锚定报告 §2.1；本机 2080Ti 仅用于真实训练与张量驻留，
6GB 笔记本档的物理账仍按报告参数，标注区分 实测/推算）。"""
import dataclasses

@dataclasses.dataclass
class HW:
    vram_mb: int = 6144
    bw_vram: float = 90.0      # GB/s 有效
    bw_dram: float = 80.0
    bw_pcie: float = 12.0      # x8 笔记本保守
    bw_nvme: float = 6.0
    pcie_pin_eff: float = 0.88
    backbone_gb: float = 2.6
    kv_mb_per_token: float = 0.13
    prefill_tok_s: float = 800.0
    decode_tok_s: float = 22.0
    workspace_mb: float = 150.0
    l1_budget_mb: float = 1000.0

H = HW()

def token_stream_ceiling(params_gb: float, link_gbps: float = H.bw_pcie) -> float:
    return link_gbps / params_gb

def load_latency_ms(mb: float, from_nvme: bool = True) -> float:
    t = (mb / 1024 / H.bw_nvme) * 1000 if from_nvme else 0.0
    t += (mb / 1024 / (H.bw_pcie * H.pcie_pin_eff)) * 1000 + 20
    return t

def hot_load_ms(mb: float) -> float:
    return (mb / 1024 / (H.bw_pcie * H.pcie_pin_eff)) * 1000 + 15

def prefill_ms(tokens: float) -> float:
    return tokens / H.prefill_tok_s * 1000
