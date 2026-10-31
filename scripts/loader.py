# -*- coding: utf-8 -*-
"""L0 中心模型加载器 —— 让自制基座能被标准工具链加载。

用法（任何脚本/推理入口第一步）：
    from loader import load_l0, load_tokenizer
    model, tok = load_l0()                 # 默认加载 l0_base/
或：
    import loader; loader.register()       # 仅注册，之后可用
    AutoModelForCausalLM.from_pretrained(path)

注册后 AutoConfig / AutoModelForCausalLM 均认识 model_type="sediment_base"。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_l0 import SedimentBase, SedimentConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_L0 = os.path.join(ROOT, "l0_base")
QWEN3B = r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct"

_registered = False


def register():
    """把自定义架构注册进 HF Auto 类（幂等）。"""
    global _registered
    if _registered:
        return
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        AutoConfig.register("sediment_base", SedimentConfig)
    except ValueError:
        pass  # 已注册
    try:
        AutoModelForCausalLM.register(SedimentConfig, SedimentBase)
    except ValueError:
        pass
    _registered = True


def load_tokenizer(path=DEFAULT_L0):
    from transformers import AutoTokenizer
    p = path if os.path.exists(os.path.join(path, "tokenizer.json")) else QWEN3B
    tok = AutoTokenizer.from_pretrained(p)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_l0(path=DEFAULT_L0, dtype="float16", device="cuda:0"):
    """加载 L0 中心模型。dtype 支持 float16/bfloat16/float32 或 torch dtype。

    ⚠️ 为什么不用 from_pretrained：transformers 5.8 对未收录的自定义 model_type，
    加载时会把 tied 声明涉及的 embed_tokens/lm_head 键跳过（留随机初始化）——
    已用 ppl 实测证实（手动载入 ppl≈67 vs from_pretrained ppl≈47万）。
    本函数按"构建结构 + safetensors 全量载入"的已验证正确路径加载。
    """
    import torch
    from safetensors.torch import load_file
    register()
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32, "fp16": torch.float16}.get(dtype, dtype)
    cfg = SedimentConfig.from_pretrained(path)
    model = SedimentBase(cfg)
    st = load_file(os.path.join(path, "model.safetensors"))
    missing, unexpected = model.load_state_dict(st, strict=False)
    bad = [k for k in missing if "lm_head" not in k]  # tied 下 lm_head 可能与 embed 共享而报 missing
    if bad or unexpected:
        raise RuntimeError(f"权重载入不完整 missing={bad[:5]} unexpected={list(unexpected)[:5]}")
    model.tie_weights()
    if dt is not torch.float32:
        model = model.to(dt)
    if device:
        model = model.to(device)
    model.eval()
    return model, load_tokenizer(path)
