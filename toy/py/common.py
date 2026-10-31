# -*- coding: utf-8 -*-
"""真实可训练组件（全部真梯度下降，无模拟）：
- 马尔可夫语料族：可学的“技能域”过程化生成，有可验证真值
- ToyLM：2 层因果 Transformer（d=96，4 头），next-token 预测
- LoraLinear：低秩注入 attn(qkv 3d×d)；delta 可 SVD 截断 = “热蒸馏”真实形态
- MemoryPacket：M 槽可寻址记忆层 = MSP 槽位的真实形态；可 save/load（序列化）
- probe_linear：骨干间线性探针 = 锚点标定 head 的真实实现
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL, N_HEADS, N_LAYERS, VOCAB, CTX = 96, 4, 2, 120, 64
ART_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
os.makedirs(ART_DIR, exist_ok=True)
LANG = np.arange(10, 70)  # 语言链只用 10–69；70–119 保留为“新知识”token（骨干从未见过）


# ---------- 语料：每个技能 = 稀疏马尔可夫链，链 i 独占 token 区间（技能间不冲突，
#           多教师融合蒸馏才可能成立；共享输入区=冲突标签，v2 的失败根因） ----------
def chain_lo(i, n_per=14):  # 链 i 的输入 token 区间 [lo, lo+n_per)
    return 4 + i * n_per


def make_skill_chains(n_skills: int, base_seed: int = 11):
    rng = np.random.default_rng(base_seed)
    chains = []
    for i in range(n_skills):
        lo = chain_lo(i)
        band = np.arange(lo, min(lo + 14, VOCAB))
        P = np.zeros((VOCAB, VOCAB))
        for t in band:
            j = rng.choice(band, size=3)
            w = rng.random(3) + 0.2
            P[t, j] = w / w.sum()
        rs = P.sum(axis=1, keepdims=True)
        unif = np.zeros((1, VOCAB)); unif[0, band] = 1.0 / len(band)
        P = np.where(rs > 0, P / np.where(rs > 0, rs, 1.0), unif)
        chains.append(P)
    return chains


def sample_batch(chain, n_seq, L=CTX, rng=None, band=None):
    rng = rng or np.random.default_rng()
    band = band if band is not None else np.nonzero(chain.sum(axis=1) > 0)[0]
    x = np.zeros((n_seq, L), dtype=np.int64)
    for b in range(n_seq):
        t = rng.choice(band)
        for i in range(L):
            x[b, i] = t
            t = rng.choice(VOCAB, p=chain[t])
    xt = torch.from_numpy(x).to(DEV)
    return xt[:, :-1], xt[:, 1:]


def mix_batch(chains, n_seq, rng=None):
    c = chains[rng.integers(len(chains))]
    return sample_batch(c, n_seq, rng=rng)


# ---------- ToyLM ----------
class Block(nn.Module):
    def __init__(self, d=D_MODEL, h=N_HEADS):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x):
        b, t, d = x.shape
        z = self.ln1(x)
        q, k, v = self.attn(z).chunk(3, dim=-1)
        q, k, v = (u.view(b, t, self.h, d // self.h).transpose(1, 2) for u in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(o.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.ln2(x))


class ToyLM(nn.Module):
    def __init__(self, seed=0, vocab=VOCAB):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied

    def hidden(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.lnf(h)

    def forward(self, x):
        return self.head(self.hidden(x))


# ---------- LoRA（包装 attn 的 qkv 线性） ----------
class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.base, self.r = base, r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + ((x @ self.A.T) @ self.B.T) * self.scale

    @torch.no_grad()
    def delta(self):
        return (self.B @ self.A) * self.scale  # out×in

    @torch.no_grad()
    def merged(self):
        w = self.base.weight + self.delta()
        return w


def attach_lora(model, layer_idx=0, r=16, seed=0):
    blk = model.blocks[layer_idx]
    lora = LoraLinear(blk.attn, r=r, seed=seed).to(DEV)
    blk.attn = lora
    for p in model.parameters():
        p.requires_grad_(False)
    lora.A.requires_grad_(True); lora.B.requires_grad_(True)
    return model, lora


@torch.no_grad()
def apply_delta_to_attn(model, layer_idx, delta_flat, shape=None):
    """把（蒸馏后的）ΔW 合并进指定层 attn 权重——热沉积合并的真实动作。"""
    blk = model.blocks[layer_idx]
    attn = blk.attn
    w = attn.weight if isinstance(attn, nn.Linear) else attn.base.weight
    w.data += delta_flat.reshape(w.shape)


def train_steps(model, get_batch, steps, lr=1e-3, opt=None):
    ps = [p for p in model.parameters() if p.requires_grad]
    opt = opt or torch.optim.AdamW(ps, lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        x, y = get_batch()
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


@torch.no_grad()
def next_token_acc(model, chain, n_seq=96, rng=None):
    rng = rng or np.random.default_rng(123)
    x, y = sample_batch(chain, n_seq, rng=rng)
    return (model(x).argmax(-1) == y).float().mean().item()


def svd_truncate(delta: torch.Tensor, rank: int):
    U, S, Vh = torch.linalg.svd(delta.double(), full_matrices=False)
    S2 = S.clone(); S2[rank:] = 0
    return (U @ torch.diag(S2) @ Vh).to(delta.dtype), (S[rank:].norm() / S.norm()).item()


# ---------- MSP 记忆包 ----------
class MemoryPacket(nn.Module):
    def __init__(self, M=256, d=D_MODEL, vocab=VOCAB, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.K = nn.Parameter(torch.randn(M, d) * 0.5)
        self.V = nn.Parameter(torch.randn(M, d) * 0.5)
        self.q = nn.Linear(d, d)
        self.out = nn.Linear(d, vocab)

    def read(self, h):
        att = torch.softmax(self.q(h) @ self.K.T / math.sqrt(h.shape[-1]), dim=-1)
        return self.out(att @ self.V)

    def state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def load_state_cpu(self, st):
        self.load_state_dict({k: v.to(DEV) for k, v in st.items()})

    def size_bytes(self):
        return sum(v.numel() * v.element_size() for v in self.state_dict().values())


def probe_linear(h_src, h_dst, lam=1e-2):
    X = torch.as_tensor(h_src, dtype=torch.float32, device=DEV)
    Y = torch.as_tensor(h_dst, dtype=torch.float32, device=DEV)
    A = X.T @ X + lam * torch.eye(X.shape[1], device=DEV)
    return torch.linalg.solve(A, X.T @ Y)
