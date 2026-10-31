# -*- coding: utf-8 -*-
"""Sediment-Base —— 本项目自制的中心模型（L0 岩基带）。

【针对沉积架构的项目适配】（不是通用小模型）
1. 词表 = Qwen2.5 的 151936，embedding/lm_head tied
   → 与 Qwen2.5-3B 教师共享坐标系：MSP 跨骨干 ISA 转译有共同参考帧（方案 §2.3）。
2. 子模块命名完全对齐 Qwen2（q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/
   down_proj/input_layernorm/post_attention_layernorm）
   → 已验收的 Sedu 技能管线（peft target_modules、SVD 蒸馏、保真门）可原样复用。
3. 继承 HF PreTrainedModel + 实现 prepare_inputs_for_generation
   → peft 的 PeftModelForCausalLM 与 model.generate() 可直接工作。
4. GQA：16 Q 头 / 4 KV 头 → KV cache 仅同规模 MHA 的 1/4
   → 直接扩大方案 §3 竞价区的 KV 预算（低显存核心诉求）。
5. 分块交叉熵：不 materialize (B·T, 151936) 全量 logits
   → 152k 大词表的输出层显存峰值从 ~5GB 降到 ~0.3GB，得以关闭梯度检查点。
6. 参数 ≈0.83B（落在授权的 0.5–1.5B）；fp16 权重 1.66GB，
   6GB 档设备可留 ~4GB 给 L1 热沉积带 + KV 竞价区（方案 §4 预算表）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin


class SedimentConfig(PretrainedConfig):
    model_type = "sediment_base"

    def __init__(self, vocab_size=151936, hidden_size=1536, num_hidden_layers=24,
                 num_attention_heads=16, num_key_value_heads=4, intermediate_size=4096,
                 max_position_embeddings=4096, rope_theta=10000.0, rms_norm_eps=1e-5,
                 tie_word_embeddings=True, ce_chunk=512, **kw):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kw)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.ce_chunk = ce_chunk

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_hidden(self):
        return self.num_key_value_heads * self.head_dim


# 兼容旧脚本名
Config = SedimentConfig


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(v + self.eps).to(x.dtype)) * self.weight


def rope_cache(seq, head_dim, theta, device, dtype, offset=0):
    """offset：增量解码时新 token 的绝对位置必须从 past_len 起算。"""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq, device=device).float() + float(offset)
    ang = pos[:, None] * inv[None, :]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def apply_rope(x, cos, sin):
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: SedimentConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_hidden, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_hidden, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask=None, past_key_values=None, layer_idx=0):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_proj(x).view(B, T, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_key_values is not None:
            # transformers 5.8 只接受 Cache 实例（tuple 已被移除），沿用 v4 的 update(k,v,layer_idx)
            k, v = past_key_values.update(k, v, layer_idx)
        rep = c.num_attention_heads // c.num_key_value_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=mask is None and q.shape[2] > 1)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, c.hidden_size))


class FFN(nn.Module):
    def __init__(self, cfg: SedimentConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: SedimentConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = FFN(cfg)

    def forward(self, x, cos, sin, mask=None, past_key_values=None, layer_idx=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask,
                               past_key_values=past_key_values, layer_idx=layer_idx)
        return x + self.mlp(self.post_attention_layernorm(x))


def chunked_cross_entropy(hidden, weight, labels, chunk, ignore_index=-100):
    """分块 + 逐块梯度检查点的 CE。
    关键：每块用 checkpoint 包裹 → forward 不保存 (chunk,V) logits，
    backward 时逐块重算。152k 大词表下显存峰值 = 单块 logits（~0.3GB）
    而非全量 (N,V)（~2.5GB fp32 + softmax 中间量）。
    这是能否关闭主干梯度检查点、把吞吐从 4.2k 拉回 ~9k tok/s 的决定性优化。
    """
    H = hidden.reshape(-1, hidden.shape[-1])
    L = labels.reshape(-1)
    dev = H.device
    total = torch.zeros((), device=dev, dtype=torch.float32)
    cnt = torch.zeros((), device=dev, dtype=torch.float32)

    def blk(h, w, lb):
        lg = F.linear(h, w)
        s = F.cross_entropy(lg.float(), lb, ignore_index=ignore_index, reduction="sum")
        c = (lb != ignore_index).sum().to(torch.float32)
        return s, c

    use_ckpt = H.requires_grad and torch.is_grad_enabled()
    for i in range(0, H.shape[0], chunk):
        hc, lc = H[i:i + chunk], L[i:i + chunk]
        if use_ckpt:
            s, c = torch.utils.checkpoint.checkpoint(blk, hc, weight, lc, use_reentrant=False)
        else:
            s, c = blk(hc, weight, lc)
        total = total + s
        cnt = cnt + c
    return total / cnt.clamp(min=1.0)


class SedimentBase(PreTrainedModel, GenerationMixin):
    config_class = SedimentConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _supports_cache_class = True
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config: SedimentConfig = None):
        config = config or SedimentConfig()
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # tied weights（HF 标准钩子）
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, v):
        self.embed_tokens = v

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, v):
        self.lm_head = v

    def _init_weights(self, m):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=std)

    def hidden_states(self, input_ids, use_ckpt=False, attention_mask=None,
                      past_key_values=None):
        """供 MSP 锚点标定使用：返回最后归一化隐状态。

        past_key_values 非空时进入增量解码路径：只算新 token，K/V 追加进 Cache。
        """
        T = input_ids.shape[1]
        past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        total = past_len + T
        cos, sin = rope_cache(T, self.config.head_dim, self.config.rope_theta,
                              input_ids.device, self.embed_tokens.weight.dtype, offset=past_len)
        mask = None
        # ⚠️ 因果性守卫（2026-10-30 修）：曾经这里只要 attention_mask 非空就构造
        # (1,1,1,T) 的"零加性掩码"并令 is_causal=False —— 零掩码等于不掩，注意力被
        # 变成双向：前序位置的 KV 被未来 token 改写，最后一个位置的隐状态也随之改变
        # （实测同输入下最后位置 logits 最大差 10.1，argmax 从 ',' 翻成 '于'）。
        # 症状：forward/teacher-forcing 正常，generate 一传 attention_mask 就退化复读。
        # 现在：无真实 padding 时保持 is_causal 快路径；有 padding 时用 bool 掩码
        # =（因果 ∧ 保留位置）。
        if past_len > 0:
            # 增量解码：q_len×kv_len 的因果掩码（单 token 解码时 T=1，等价于全见过去）
            if T > 1:
                causal = torch.ones(T, total, dtype=torch.bool,
                                    device=input_ids.device).tril(diagonal=past_len)
                mask = causal[None, None]
        elif attention_mask is not None and attention_mask.shape[1] == T and T > 1:
            if bool((attention_mask == 0).any()):
                causal = torch.ones(T, T, dtype=torch.bool, device=input_ids.device).tril()
                mask = attention_mask[:, None, None, :].bool() & causal[None, None]
        if mask is not None and attention_mask is not None \
                and attention_mask.shape[-1] == total and bool((attention_mask == 0).any()):
            mask = mask & attention_mask[:, None, None, :].bool()
        x = self.embed_tokens(input_ids)
        for i, blk in enumerate(self.layers):
            if use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, mask,
                                                      use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask, past_key_values=past_key_values, layer_idx=i)
        return self.norm(x)

    def forward(self, input_ids, labels=None, attention_mask=None,
                use_ckpt=False, past_key_values=None, use_cache=False, **kw):
        h = self.hidden_states(input_ids, use_ckpt=use_ckpt, attention_mask=attention_mask,
                               past_key_values=past_key_values)
        present = past_key_values if use_cache else None
        if labels is not None:
            loss = chunked_cross_entropy(h, self.lm_head.weight, labels,
                                         self.config.ce_chunk)
            return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=present,
                                          hidden_states=h)
        logits = self.lm_head(h)
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=present,
                                      hidden_states=None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, **kw):
        # 有 cache 时只把最后一个 token 喂进来（K/V 从 cache 取），否则全序列重算
        if past_key_values is not None and int(past_key_values.get_seq_length()) > 0:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": True}

    @torch.no_grad()
    def kv_bytes_per_token(self, dtype_bytes=2):
        c = self.config
        return 2 * c.num_hidden_layers * c.num_key_value_heads * c.head_dim * dtype_bytes
