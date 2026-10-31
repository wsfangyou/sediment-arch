# -*- coding: utf-8 -*-
"""把 KV cache 打进 SedimentBase 的最小改动（patch 脚本，改完即删）。

改四处：
  1. rope_cache 增加 offset（增量解码的位置必须从 past_len 起算）
  2. Attention.forward 接受 past_key_values/layer_idx，走 Cache.update
  3. Block.forward 透传
  4. SedimentBase.hidden_states/forward 接受 past_key_values/use_cache，
     并给出与 is_causal 等价的 bool 因果掩码；prepare_inputs_for_generation 在有 cache 时只喂最后一个 token
"""
import io, os, sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_l0.py")
s = open(p, encoding="utf-8").read()

# ---- 1) rope_cache 支持 offset ----
old = '''def rope_cache(seq, head_dim, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.arange(seq, device=device).float()[:, None] * inv[None, :]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)'''
new = '''def rope_cache(seq, head_dim, theta, device, dtype, offset=0):
    """offset：增量解码时新 token 的绝对位置必须从 past_len 起算。"""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq, device=device).float() + float(offset)
    ang = pos[:, None] * inv[None, :]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)'''
assert old in s; s = s.replace(old, new)

# ---- 2) Attention 支持 Cache ----
old = '''    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_proj(x).view(B, T, c.num_attention_heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = c.num_attention_heads // c.num_key_value_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=mask is None and T > 1)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, c.hidden_size))'''
new = '''    def forward(self, x, cos, sin, mask=None, past_key_values=None, layer_idx=0):
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
        return self.o_proj(o.transpose(1, 2).reshape(B, T, c.hidden_size))'''
assert old in s; s = s.replace(old, new)

# ---- 3) Block 透传 ----
old = '''    def forward(self, x, cos, sin, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))'''
new = '''    def forward(self, x, cos, sin, mask=None, past_key_values=None, layer_idx=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask,
                               past_key_values=past_key_values, layer_idx=layer_idx)
        return x + self.mlp(self.post_attention_layernorm(x))'''
assert old in s; s = s.replace(old, new)

# ---- 4) hidden_states 支持 cache ----
old = '''    def hidden_states(self, input_ids, use_ckpt=False, attention_mask=None):
        """供 MSP 锚点标定使用：返回最后归一化隐状态。"""
        T = input_ids.shape[1]
        cos, sin = rope_cache(T, self.config.head_dim, self.config.rope_theta,
                              input_ids.device, self.embed_tokens.weight.dtype)
        mask = None'''
new = '''    def hidden_states(self, input_ids, use_ckpt=False, attention_mask=None,
                      past_key_values=None):
        """供 MSP 锚点标定使用：返回最后归一化隐状态。

        past_key_values 非空时进入增量解码路径：只算新 token，K/V 追加进 Cache。
        """
        T = input_ids.shape[1]
        past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        total = past_len + T
        cos, sin = rope_cache(T, self.config.head_dim, self.config.rope_theta,
                              input_ids.device, self.embed_tokens.weight.dtype, offset=past_len)
        mask = None'''
assert old in s; s = s.replace(old, new)

# ---- 4b) 掩码：有 cache 或真实 padding 时用「因果 ∧ 保留」bool 掩码 ----
old = '''        if attention_mask is not None and attention_mask.shape[1] == T and T > 1:
            if bool((attention_mask == 0).any()):
                causal = torch.ones(T, T, dtype=torch.bool, device=input_ids.device).tril()
                mask = attention_mask[:, None, None, :].bool() & causal[None, None]
        x = self.embed_tokens(input_ids)
        for blk in self.layers:
            if use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, mask,
                                                      use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask)
        return self.norm(x)'''
new = '''        if past_len > 0:
            # 增量解码：q_len×kv_len 的因果掩码（单 token 解码时 T=1，等价于全见过去）
            if T > 1:
                causal = torch.ones(T, total, dtype=torch.bool,
                                    device=input_ids.device).tril(diagonal=past_len)
                mask = causal[None, None]
        elif attention_mask is not None and attention_mask.shape[1] == T and T > 1:
            if bool((attention_mask == 0).any()):
                causal = torch.ones(T, T, dtype=torch.bool, device=input_ids.device).tril()
                mask = attention_mask[:, None, None, :].bool() & causal[None, None]
        if mask is not None and attention_mask is not None \\
                and attention_mask.shape[-1] == total and bool((attention_mask == 0).any()):
            mask = mask & attention_mask[:, None, None, :].bool()
        x = self.embed_tokens(input_ids)
        for i, blk in enumerate(self.layers):
            if use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, mask,
                                                      use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask, past_key_values=past_key_values, layer_idx=i)
        return self.norm(x)'''
assert old in s; s = s.replace(old, new)

# ---- 5) forward 接受 cache 参数并回传 ----
old = '''    def forward(self, input_ids, labels=None, attention_mask=None,
                use_ckpt=False, **kw):
        h = self.hidden_states(input_ids, use_ckpt=use_ckpt, attention_mask=attention_mask)
        if labels is not None:
            loss = chunked_cross_entropy(h, self.lm_head.weight, labels,
                                         self.config.ce_chunk)
            return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=None,
                                          hidden_states=h)
        logits = self.lm_head(h)
        return CausalLMOutputWithPast(loss=None, logits=logits, past_key_values=None,
                                      hidden_states=None)'''
new = '''    def forward(self, input_ids, labels=None, attention_mask=None,
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
                                      hidden_states=None)'''
assert old in s; s = s.replace(old, new)

# ---- 6) prepare_inputs_for_generation：有 cache 时只喂最后一个 token ----
old = '''    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, **kw):
        return {"input_ids": input_ids, "attention_mask": attention_mask}'''
new = '''    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, **kw):
        # 有 cache 时只把最后一个 token 喂进来（K/V 从 cache 取），否则全序列重算
        if past_key_values is not None and int(past_key_values.get_seq_length()) > 0:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": True}'''
assert old in s; s = s.replace(old, new)

# ---- 7) 能力声明 ----
old = '''    _no_split_modules = ["Block"]'''
new = '''    _no_split_modules = ["Block"]
    _supports_cache_class = True'''
assert old in s; s = s.replace(old, new)

open(p, "w", encoding="utf-8").write(s)
print(f"patched {p} → {len(s)} chars")
