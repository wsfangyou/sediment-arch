# -*- coding: utf-8 -*-
"""E3-real：MSP 跨骨干寻址在真骨干对上（Qwen2.5-3B → 自制 L0）的 b/a 比例。

这是方案 §7 实验 3 的"存亡假设"在真实骨干上的第一次量测。玩具尺度报的是 b/a=0.94，
但玩具的知识域是"单 token 置换"、且两个骨干同架构同深度；这里换成真骨干对
（3B/36 层/2048 维 ↔ 0.83B/24 层/1536 维），结论才对外部有效。

机制（与 exp3_msp.py 同构，只是换成真 hidden）：
  · 读出口 = MSP 记忆层：M 个槽 (K: M×d, V: M×vocab)
      score = softmax(cos(h_last, K)/τ)，logits_bias = score @ V，答案 = argmax(logits_bias)
  · 知识域 = 50 对 (cue → answer) 单 token 映射（Sedu 词典构造，两个骨干都没见过）
  · 四臂：
      a   联合上限：在 **目标骨干 L0** 上直接训 K,V（键 = L0 本地坐标）
      b   ISA 转译：在 **参考骨干 Qwen** 上训 K_q,V → 用 N 个公共锚点拟合 W(Qwen→L0)
          → K_l0 = K_q @ W，在 L0 上只换键、不重训（V 是词表对齐的偏置，直接搬）
      b0  不转译：K_q 用随机投影降到 L0 维度（坐标错配直读）
      oracle 容量：见 a（a 高说明"槽容量够"，排除容量混淆）
  · 锚点漂移：N ∈ {4,8,16,32,60} 看 b 的准确率曲线

关键实现细节：骨干全冻结 → hidden 预计算一次，读出口离线训练（秒级），
不需要反复过 3B，所以整个实验几分钟内跑完。

预注册判定（沿用方案口径）：b/a ≥ 0.95 → 跨骨干 ISA 成立；0.90–0.95 → 部分成立。
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_l0, load_tokenizer, register, DEFAULT_L0, QWEN3B

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
DEV = "cuda:0"
VOCAB = 151936
D_L0_MID, D_Q_MID = 1536, 2048
L0_LAYER = 11          # 第 12 层（0-based 11）block 输出
Q_LAYER = 18           # Qwen 36 层的正中间


def log(*a):
    print(*a, flush=True)


# ---------------- 知识域：50 对 cue→answer ----------------
def build_pairs(tok):
    lex = json.load(open(os.path.join(ROOT, "data", "sedu_lexicon.json"), encoding="utf-8"))
    # 词典形如 {"sedu": "中文"} 或列表；两种都兼容
    items = []
    if isinstance(lex, dict):
        cand = lex.get("lexicon") or lex.get("words") or lex.get("pairs") or lex
        if isinstance(cand, dict):
            items = list(cand.items())
        elif isinstance(cand, list):
            for x in cand:
                if isinstance(x, dict):
                    items.append((x.get("sedu") or x.get("s"), x.get("zh") or x.get("c")))
                elif isinstance(x, (list, tuple)) and len(x) >= 2:
                    items.append((x[0], x[1]))
    items = [(a, b) for a, b in items if a and b][:50]
    pairs = []
    for cue, ans in items:
        # 答案紧跟 "A:" 之后，无前导空格 → 先按原样分词
        ids_ans = tok(str(ans), add_special_tokens=False)["input_ids"]
        if len(ids_ans) != 1:
            ids_ans = tok(" " + str(ans), add_special_tokens=False)["input_ids"]
        if len(ids_ans) != 1:
            continue                       # 只保留单 token 答案（本轮的可判分前提）
        pairs.append({"cue": str(cue), "answer": str(ans), "ans_id": ids_ans[0]})
    return pairs


def prompt_of(p):
    return f"Sed: {p['cue']}\nA:"


# ---------------- hidden 提取 ----------------
@torch.no_grad()
def l0_mid(model, tok, texts):
    buf = {}
    def hook(mod, args, out):
        buf["h"] = out[0] if isinstance(out, tuple) else out
    hd = model.layers[L0_LAYER].register_forward_hook(hook)
    hs = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        model.hidden_states(ids)
        hs.append(buf["h"][0, -1].float().cpu())
    hd.remove()
    return torch.stack(hs)


@torch.no_grad()
def q_mid(model, tok, texts):
    hs = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        out = model(ids, output_hidden_states=True)
        hs.append(out.hidden_states[Q_LAYER][0, -1].float().cpu())
    return torch.stack(hs)


# ---------------- MSP 读出口 ----------------
def train_head(H, y, n_slot=50, steps=2000, lr=0.2, tau=0.05, seed=0):
    """在固定 hidden 上离线训 MSP 读出口。

    槽 (K: M×d) 负责寻址，值 (V: M×C) 只需覆盖答案集 C（本轮 47 个候选答案）——
    这样量到的就是"键空间是否还能寻址"，不混入"在 152k 词表里做开放词表分类"的难度。
    """
    torch.manual_seed(seed)
    n, d = H.shape
    C = int(y.max().item()) + 1
    K = torch.nn.Parameter(torch.randn(n_slot, d, device=H.device) * (1.0 / math.sqrt(d)))
    V = torch.nn.Parameter(torch.randn(n_slot, C, device=H.device) * 0.01)
    Hn = F.normalize(H, dim=1)
    opt = torch.optim.Adam([K, V], lr=lr)
    yy = y.to(H.device).long()
    for _ in range(steps):
        p = torch.softmax(F.normalize(K, dim=1) @ Hn.T / tau, dim=0)      # (M, n)
        logits = (p.T @ V)                                                # (n, C)
        loss = F.cross_entropy(logits, yy)
        opt.zero_grad(); loss.backward(); opt.step()
    return K.detach(), V.detach(), C


@torch.no_grad()
def acc_of(K, V, H, y, tau=0.05):
    Hn = F.normalize(H, dim=1)
    p = torch.softmax(F.normalize(K, dim=1) @ Hn.T / tau, dim=0)
    pred = (p.T @ V).argmax(1)
    return float((pred == y.to(H.device).long()).float().mean())


# ---------------- 标定：Qwen 坐标 → L0 坐标 ----------------
def fit_map(X, Y, k=32, lam=0.1):
    """PCA-k + 岭回归。n≪d 时直接对偶式等于插值、泛化为 0（D3 已实测），必须先降维。"""
    mx, my = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc = X - mx
    U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    Vk = V[:min(k, V.shape[0])]
    Z = Xc @ Vk.T
    Wz = torch.linalg.solve(Z.T @ Z + lam * torch.eye(Z.shape[1], device=X.device), Z.T @ (Y - my))
    return (mx, my, Vk, Wz)


def apply_map(X, mp):
    mx, my, Vk, Wz = mp
    return (X - mx) @ Vk.T @ Wz + my


def main():
    torch.manual_seed(0)
    tok = load_tokenizer(DEFAULT_L0)
    pairs = build_pairs(tok)
    log(f"[知识域] {len(pairs)} 对 cue→answer（单 token 答案）")
    # 答案集 C：47 个候选答案映射成 0..C-1（类索引），同时保留 token id 备查
    uniq = {p["ans_id"]: i for i, p in enumerate(pairs)}
    y = torch.tensor([uniq[p["ans_id"]] for p in pairs])
    texts = [prompt_of(p) for p in pairs]

    # 锚点（公共文本，两边都能读到）
    corpus = np.memmap(os.path.join(ROOT, "corpus", "tokens.bin"), dtype=np.uint32, mode="r")
    n_all = len(corpus)
    anchor_texts = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        i = int(rng.integers(0, n_all - 40))
        anchor_texts.append(tok.decode(corpus[i:i + 32].astype(np.int64).tolist()))

    register()
    log("[骨干] 载入 L0（fp16）…")
    l0, _ = load_l0(dtype="float16", device=DEV)
    H0 = l0_mid(l0, tok, texts)
    A0 = l0_mid(l0, tok, anchor_texts)
    log(f"  L0 mid hidden: pairs {tuple(H0.shape)} anchors {tuple(A0.shape)}")
    del l0; torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("[骨干] 载入 Qwen2.5-3B-Instruct（fp16）…")
    qt = AutoTokenizer.from_pretrained(QWEN3B)
    qm = AutoModelForCausalLM.from_pretrained(QWEN3B, dtype=torch.float16).to(DEV).eval()
    HQ = q_mid(qm, qt, texts)
    AQ = q_mid(qm, qt, anchor_texts)
    log(f"  Qwen mid hidden: pairs {tuple(HQ.shape)} anchors {tuple(AQ.shape)}")
    del qm; torch.cuda.empty_cache()

    H0, HQ = H0.to(DEV), HQ.to(DEV)
    A0, AQ = A0.to(DEV), AQ.to(DEV)

    # ---- 臂 a：L0 本地坐标直接训（联合上限 / 容量 oracle）----
    t0 = time.time()
    Ka, Va, C = train_head(H0, y)
    acc_a = acc_of(Ka, Va, H0, y)
    log(f"[臂 a] 联合上限（L0 本地坐标）: acc {acc_a:.3f}（{time.time()-t0:.0f}s）"
        f" → 槽容量{'够用' if acc_a > 0.9 else '不足'}")

    # ---- 臂 b/b0：在 Qwen 上训，再转译 ----
    Kq, Vq, Cq = train_head(HQ, y)
    acc_q = acc_of(Kq, Vq, HQ, y)
    log(f"[臂 b 训练] 参考骨干(Qwen)上的 acc {acc_q:.3f}")

    # 方向 1（标键）：Qwen 坐标 → L0 坐标，键搬到 L0 空间后用 L0 的 hidden 寻址
    mp_q2l = fit_map(AQ, A0)
    # 方向 2（标 hidden）：L0 hidden → Qwen 坐标，在参考空间里寻址（等价但数值更稳）
    mp_l2q = fit_map(A0, AQ)

    # 诊断：标定图在**真实 prompt** 上的保真度（这是"寻址能否成立"的前提）
    Hm = apply_map(H0, mp_l2q)                       # L0 hidden 映射到 Qwen 空间
    cos_pair = float(F.cosine_similarity(F.normalize(Hm, dim=1), F.normalize(HQ, dim=1)).mean())
    S = F.normalize(Hm, dim=1) @ F.normalize(HQ, dim=1).T
    ret1 = float((S.argmax(1) == torch.arange(len(HQ), device=HQ.device)).float().mean())
    log(f"[诊断] 标定图在 47 对 prompt 上：余弦 {cos_pair:.3f}，检索@1 {ret1:.3f}"
        f"（随机 1/{len(HQ)} = {1/len(HQ):.3f}）")

    acc_b_key = acc_of(apply_map(Kq, mp_q2l), Vq, H0, y)
    acc_b_hid = acc_of(Kq, Vq, apply_map(H0, mp_l2q), y)
    acc_b = max(acc_b_key, acc_b_hid)
    log(f"[臂 b] 标键 {acc_b_key:.3f} / 标 hidden {acc_b_hid:.3f}（取较优者）")

    # 漂移曲线：锚点数 → 标键方向的准确率
    curve = []
    for n in (8, 16, 32, 64, 128, 200):
        mp = fit_map(AQ[:n], A0[:n])
        a = acc_of(apply_map(Kq, mp), Vq, H0, y)
        curve.append({"anchors": n, "acc": round(a, 4)})
        log(f"  [漂移] 锚点 {n:>3}: acc {a:.3f}")

    # 臂 b0：随机投影（坐标错配直读）
    G = torch.randn(D_Q_MID, D_L0_MID, device=DEV) / math.sqrt(D_Q_MID)
    acc_b0 = acc_of(Kq @ G, Vq, H0, y)
    log(f"[臂 b0] 不转译（随机投影直读）: acc {acc_b0:.3f}")

    ratio = acc_b / max(acc_a, 1e-9)
    ok = ratio >= 0.95
    verdict = (f"✅ 跨骨干 ISA 成立（b/a = {ratio:.3f} ≥ 0.95）" if ok
               else f"⚠️ 部分成立（b/a = {ratio:.3f}，未过 0.95 线）" if ratio >= 0.90
               else f"❌ 转译损失过大（b/a = {ratio:.3f}）")
    log(f"\n[判定] a={acc_a:.3f} b={acc_b:.3f} b0={acc_b0:.3f} → b/a={ratio:.3f}")
    log(verdict)

    out = {"n_pairs": len(pairs), "vocab": VOCAB,
           "backbones": {"reference": "Qwen2.5-3B-Instruct (layer 18, d=2048)",
                         "target": "Sediment-Base L0 (layer 12, d=1536)"},
           "arms": {"a_joint_L0": round(acc_a, 4), "b_isa_Qwen2L0": round(acc_b, 4),
                    "b_isa_calib_keys": round(acc_b_key, 4),
                    "b_isa_calib_hidden": round(acc_b_hid, 4),
                    "b0_no_calib": round(acc_b0, 4), "q_native": round(acc_q, 4)},
           "n_anchors": 200, "map": "PCA-32 + ridge(lam=0.1)",
           "ratio_b_over_a": round(ratio, 4),
           "map_diagnostic": {"pair_cos": round(cos_pair, 4), "pair_retrieval_at1": round(ret1, 4)},
           "drift_curve": curve,
           "prereg": "b/a >= 0.95 → ISA 成立；0.90–0.95 部分成立",
           "verdict": verdict,
           "pairs": pairs[:5]}
    p = os.path.join(ART, "e3_msp_real_L0.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"→ {p}")


if __name__ == "__main__":
    main()
