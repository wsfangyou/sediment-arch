# -*- coding: utf-8 -*-
"""实验 3（真实版 v3，方案 §7 存亡假设）——MSP = 存储-检索型关联记忆。

设计修正（v1/v2 的教训都进了注释）：
  · v1 教训：cue 用未训练 token → 键空间坍缩不可寻址，上限臂不稳；
  · v2 教训：随机流补训把 embedding 全体压平 → 判别性反而消失。
  · v3 方案：键 = 骨干在语言分布上**已训练且判别**的 token hidden；
    事实 = 一个秘密置换 σ(t)（马尔可夫链语料里没有，骨干不可能学到），
    因此 (t → σ(t)) 是真正的“外挂知识”：只能靠记忆包，且上限臂可建立。

四臂（全部真实 next-token 精度）：
  (a)  联合上限：面向 B 写入 → B 检索
  (b)  ISA 转译：面向 A 写入并发布（序列化→读回），B 仅用公共 token
       的 (hB, hA) 锚点对做线性探针标定，槽一字节不动
  (b0) 朴素迁移：不标定
  (d)  骨干裸答：B 直接答 σ（验证骨干确实没学过这条知识）
  drift(k)：锚点数退化曲线
判定：acc(b) ≥ 95%×acc(a) → 升格；上限 <0.7 → 实验无效（不下结论）。
"""
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (ToyLM, MemoryPacket, probe_linear, train_steps, mix_batch,
                   make_skill_chains, VOCAB, D_MODEL, DEV)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ART, exist_ok=True)

# ---------- 骨干 A/B：链语料预训练（语言判别空间） ----------
chains = make_skill_chains(6)
rng_np = np.random.default_rng(5)
def get_backbone(name, seed):
    p = os.path.join(ART, f"{name}_v4.pt")
    m = ToyLM(seed=seed).to(DEV)
    if os.path.exists(p):
        m.load_state_dict(torch.load(p)); return m
    train_steps(m, lambda: mix_batch(chains, 32, rng_np), 400, lr=3e-3)
    torch.save(m.state_dict(), p)
    return m
t0 = time.time()
modelA, modelB = get_backbone("backboneA", 0), get_backbone("backboneB", 7)
print(f"[骨干] A/B 就绪 {time.time()-t0:.0f}s")

# ---------- 判别性检查：trained-token hidden 的类间分离度 ----------
@torch.no_grad()
def pooled_hidden(model, toks):
    seq = toks.view(-1, 1).repeat(1, 8)
    return model.hidden(seq)[:, -1]

toks_all = torch.arange(10, 70, device=DEV)      # 60 个已训练 token
hA_all, hB_all = pooled_hidden(modelA, toks_all), pooled_hidden(modelB, toks_all)
def sep_score(h):
    C = torch.nn.functional.normalize(h, dim=1)
    off = (C @ C.T) - torch.eye(len(h), device=DEV)
    return float(off.abs().mean())
print(f"[判别性] A {sep_score(hA_all):.3f} / B {sep_score(hB_all):.3f}（越低越可寻址）")

# ---------- 知识域：秘密置换 σ（链语料不含此信息） ----------
PAIRS = 50
perm_rng = np.random.default_rng(3)
sigma = perm_rng.permutation(PAIRS)              # t_i → 10+sigma[i]
cues = toks_all[:PAIRS]                          # 键：50 个 trained token
answers = (10 + torch.from_numpy(sigma).long().to(DEV))
train_idx = torch.arange(0, 35, device=DEV)      # 写入 35 条
test_idx = torch.arange(35, PAIRS, device=DEV)   # 评测已写入的…见下

# 存储-检索语义：全部 50 条都写入（MSP 是记忆不是泛化），评测全部 50 条
hA, hB = pooled_hidden(modelA, cues), pooled_hidden(modelB, cues)

def train_msp(hidden_cues, epochs=2500, tag=""):
    msp = MemoryPacket(M=512, d=D_MODEL).to(DEV)
    msp.out = torch.nn.Linear(D_MODEL, PAIRS).to(DEV)   # 域内 50 类答案
    opt = torch.optim.AdamW(msp.parameters(), lr=8e-3)
    hc, tgt = hidden_cues.detach(), (answers - 10)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(msp.read(hc), tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (msp.read(hc).argmax(-1) == tgt).float().mean().item()
    print(f"[MSP {tag}] loss {loss.item():.4f} 检索 acc {acc:.3f}")
    return msp

@torch.no_grad()
def eval_msp(msp, hidden_cues, head=None):
    hc = hidden_cues.detach()
    if head is not None:
        hc = hc @ head
    return (msp.read(hc).argmax(-1) == (answers - 10)).float().mean().item()

# 骨干裸答检查：B 自己能否答 σ（应≈随机）
with torch.no_grad():
    logitsB = modelB(cues.view(-1, 1))[:, -1]
    acc_backbone_raw = (logitsB.argmax(-1) == answers).float().mean().item()

# oracle 容量对照
oh = torch.zeros(PAIRS, D_MODEL, device=DEV)
oh[torch.arange(PAIRS), torch.arange(PAIRS)] = 1.0
msp_or = train_msp(oh, epochs=1500, tag="oracle one-hot")
with torch.no_grad():
    acc_oracle = (msp_or.read(oh).argmax(-1) == (answers - 10)).float().mean().item()
print(f"[oracle] 记忆容量 {acc_oracle:.3f} | 骨干裸答 σ {acc_backbone_raw:.3f}（≈随机={1/PAIRS:.3f}）")

t1 = time.time()
mspA = train_msp(hA, tag="面向A发布")
mspJoint = train_msp(hB, tag="面向B(上限)")
print(f"[MSP] 训练 {time.time()-t1:.0f}s")

# ---------- 发布：序列化 → 读回 ----------
path = os.path.join(ART, "msp_sigma_v3.pt")
torch.save(mspA.state(), path)
mspA_pub = MemoryPacket(M=512, d=D_MODEL).to(DEV)
mspA_pub.out = torch.nn.Linear(D_MODEL, PAIRS).to(DEV)
mspA_pub.load_state_cpu(torch.load(path))
identical = all(torch.equal(a, b) for a, b in zip(mspA.state().values(), mspA_pub.state().values()))
print(f"[序列化] {os.path.getsize(path)/1024:.1f} KB 一致={identical}；A 自检 {eval_msp(mspA_pub, hA):.3f}")

# ---------- 锚点标定：公共 token（含非 cue 的 60–69）→ 线性探针 ----------
anchor_toks = toks_all                       # 双方都能跑的 60 个公共锚点
head_full = probe_linear(hB_all, hA_all)
probe_err = ((hB_all @ head_full - hA_all).norm() / hA_all.norm()).item()
acc_a = eval_msp(mspJoint, hB)
acc_b = eval_msp(mspA_pub, hB, head=head_full)
acc_b0 = eval_msp(mspA_pub, hB)
drift = [{"anchors": k, "acc": round(eval_msp(mspA_pub, hB, head=probe_linear(hB_all[:k], hA_all[:k])), 3)}
         for k in [4, 8, 16, 32, 48, 60]]

out = {
  "chance": round(1.0 / PAIRS, 4),
  "acc_oracle_capacity": round(acc_oracle, 3),
  "acc_backbone_raw": round(acc_backbone_raw, 3),
  "probe_rel_err": round(probe_err, 3),
  "acc_joint_upper_a": round(acc_a, 3),
  "acc_isa_transferred_b": round(acc_b, 3),
  "acc_naive_b0": round(acc_b0, 3),
  "ratio_b_over_a": round(acc_b / max(acc_a, 1e-9), 3),
  "drift_curve": drift, "msp_KB": round(os.path.getsize(path) / 1024, 1),
}
if acc_a < 0.7:
    out["verdict"] = (f"无效对照：联合上限仅 {acc_a:.3f}（<0.7），键空间不可寻址——"
                      "先解决寻址可分性（不作升格/收缩结论）")
else:
    out["verdict"] = ("升格：ISA 转译 ≥95%×联合上限（真实骨干对·可建立上限的玩具知识域）"
                      if acc_b >= 0.95 * acc_a else
                      "部分成立：显著优于不转译但未达95%线" if acc_b > acc_b0 + 0.1 else
                      "收缩：标定不足以桥接")
print(json.dumps(out, ensure_ascii=False, indent=2))
with open(os.path.join(ART, "exp3_msp_real.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
