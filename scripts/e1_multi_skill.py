# -*- coding: utf-8 -*-
"""E1 闭环（多技能版）—— 在 Qwen3-4B 这条线上第一次跑「3 个技能共存」。

E1 要回答的三件事（玩具尺度已过，这是真模型上的第一次）：
  1. **技能之间会不会互相干扰**：挂 A 技能时，B/C 技能的准确率掉多少（干扰矩阵）
  2. **晋升代价**：每个技能注入后通用 ppl 涨多少（C 门），三个一起挂又是多少
  3. **隔离挂载是否等价于"各管各的"**：单独挂 vs 一起挂 的读数差

技能：sedu（已有 adapter_qwen4b_sedu，800 步）/ nuri / tavi（本脚本训，400 步）
判分：A_tf 全部技能都测（便宜）；A_gen 只测各自 holdout（生成慢）
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_tokenizer
from pretrain_l0 import TokenStream

DEV = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")
MODEL = "D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B"
THINK = re.compile(r"<think>.*?</think>", re.S)

SKILLS = [
    {"name": "sedu", "sys": "你是精通Sedu语与中文互译的语言助手。",
     "train": "data/sedu_train.jsonl", "hold": "data/sedu_holdout.jsonl",
     "adapter": "l0_acceptance/adapter_qwen4b_sedu", "prebuilt": True},
    {"name": "nuri", "sys": "你是精通Nuri语与中文互译的语言助手。",
     "train": "data/nuri_train.jsonl", "hold": "data/nuri_holdout.jsonl",
     "adapter": "l0_acceptance/adapter_qwen4b_nuri", "prebuilt": False},
    {"name": "tavi", "sys": "你是精通Tavi语与中文互译的语言助手。",
     "train": "data/tavi_train.jsonl", "hold": "data/tavi_holdout.jsonl",
     "adapter": "l0_acceptance/adapter_qwen4b_tavi", "prebuilt": False},
]


def log(*a, **kw):
    print(*a, flush=True)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower().replace("，", " ").replace(",", " "))


def build(tok, r, sys_prompt, trunc=192):
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": r["q"]},
            {"role": "assistant", "content": r["a"]}]
    full = tok.apply_chat_template(msgs, tokenize=False)
    pr = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"][:trunc]
    np_ = min(len(tok(pr, add_special_tokens=False)["input_ids"]), len(ids))
    return ids, [-100] * np_ + ids[np_:]


@torch.no_grad()
def tf_acc(m, tok, rows, sys_prompt):
    m.eval()
    hit = tot = 0
    for r in rows:
        ids, lb = build(tok, r, sys_prompt)
        x = torch.tensor([ids], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m(input_ids=x).logits[0].float()
        tgt = torch.tensor(lb[1:], device=DEV)
        keep = tgt != -100
        pred = lg[:-1].argmax(-1)
        hit += int((pred[keep] == tgt[keep]).sum())
        tot += int(keep.sum())
    return hit / max(tot, 1)


@torch.no_grad()
def gen_rows(m, tok, rows, sys_prompt, max_new=48):
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": r["q"]}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                       do_sample=False, pad_token_id=tok.pad_token_id,
                       repetition_penalty=1.0, eos_token_id=[151645, 151643])
        out.append(THINK.sub("", tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)).strip())
    return out


def gen_acc(rows, answers):
    hit = 0
    for r, a in zip(rows, answers):
        aa = a.strip().rstrip("。.!！")
        if r["task"] == "gloss":
            ok = r["a"] in a
        elif r["task"] == "s2c":
            ok = aa == r["a"].rstrip("。") or aa == r["a"]
        else:
            ok = norm(aa) == norm(r["a"])
        hit += int(ok)
    return hit / max(len(rows), 1)


@torch.no_grad()
def val_ppl(m, n=24):
    st = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 1024)
    m.eval()
    rng = np.random.default_rng(7)
    ls = []
    for _ in range(n):
        x, _ = st.batch(4, train=False, rng=rng)
        x = x.to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            ls.append(m(input_ids=x, labels=x).loss.item())
    return float(np.exp(min(float(np.mean(ls)), 20)))


def rows_of(path, n=None):
    rs = [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]
    return rs[:n] if n else rs


def train_one(base, tok, sk, steps, bs, replay):
    from peft import LoraConfig, TaskType, get_peft_model
    m = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.02,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m.train()
    tr = rows_of(sk["train"])
    stream = TokenStream(os.path.join(ROOT, "corpus", "tokens.bin"), 256)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    rng = np.random.default_rng(0)
    pk = tok.pad_token_id or tok.eos_token_id
    import time
    t0 = time.time()
    for s in range(steps):
        if rng.random() < replay:
            x, _ = stream.batch(bs, train=True, rng=rng)
            ids, labels = x.to(DEV), x.to(DEV).clone()
        else:
            pairs = [build(tok, tr[int(rng.integers(len(tr)))], sk["sys"]) for _ in range(bs)]
            L = max(len(p[0]) for p in pairs)
            ids = torch.full((len(pairs), L), pk, dtype=torch.long, device=DEV)
            labels = torch.full((len(pairs), L), -100, dtype=torch.long, device=DEV)
            for i, (a, b) in enumerate(pairs):
                ids[i, :len(a)] = torch.tensor(a, device=DEV)
                labels[i, :len(b)] = torch.tensor(b, device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = m(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (s + 1) % 100 == 0:
            log(f"    step {s+1}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
    m.save_pretrained(os.path.join(ROOT, sk["adapter"]))
    log(f"    adapter → {sk['adapter']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval", choices=["train", "eval", "all"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--replay", type=float, default=0.35)
    ap.add_argument("--n-train-hold", type=int, default=60)
    ap.add_argument("--n-gen", type=int, default=50)
    args = ap.parse_args()

    # ⚠️ 必须用 Qwen3 自己的 tokenizer：词表相同但 **chat template 不同**
    # （Qwen3 会插  thinking 块，Qwen2.5/L0 不会）。用错会导致 prompt 分布与训练时不一致，
    # 实测把 sedu 的 A_tf 从 0.994 压到 0.655。
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    def fresh_base():
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()

    base = fresh_base()
    if args.mode in ("train", "all"):
        for sk in SKILLS:
            if os.path.isdir(os.path.join(ROOT, sk["adapter"])):
                log(f"[{sk['name']}] 复用已有 adapter")
                continue
            log(f"[{sk['name']}] 训练 {args.steps} 步")
            del base
            torch.cuda.empty_cache()
            base = fresh_base()                     # 每个技能一份干净基座，避免 LoRA 套娃
            train_one(base, tok, sk, args.steps, args.bs, args.replay)
        del base
        torch.cuda.empty_cache()
        base = fresh_base()

    # ---------------- 评测：干扰矩阵 + C 门 ----------------
    from peft import PeftModel
    hold = {sk["name"]: rows_of(sk["hold"], args.n_train_hold) for sk in SKILLS}
    sysp = {sk["name"]: sk["sys"] for sk in SKILLS}

    res = {"skills": [s["name"] for s in SKILLS], "matrix": {}, "own": {}, "c_gate": {}}
    bare_ppl = val_ppl(base)
    log(f"[bare] 通用 ppl {bare_ppl:.2f}")
    for target in res["skills"]:
        res["matrix"]["bare"] = res["matrix"].get("bare", {})
        res["matrix"]["bare"][target] = round(tf_acc(base, tok, hold[target], sysp[target]), 4)
    log("  干扰矩阵 bare:", res["matrix"]["bare"], flush=True)

    total_ppl_sum = []
    for sk in SKILLS:
        name = sk["name"]
        path = os.path.join(ROOT, sk["adapter"])
        if not os.path.isdir(path):
            log(f"[{name}] 缺 adapter，跳过")
            continue
        m = PeftModel.from_pretrained(base, path).eval()
        res["matrix"][name] = {}
        for target in res["skills"]:
            a = tf_acc(m, tok, hold[target], sysp[target])
            res["matrix"][name][target] = round(a, 4)
        own_rows = rows_of(sk["hold"], args.n_gen)
        g_own = gen_acc(own_rows, gen_rows(m, tok, own_rows, sysp[name]))
        ppl = val_ppl(m)
        res["own"][name] = {"A_tf": res["matrix"][name][name], "A_gen": round(g_own, 4)}
        res["c_gate"][name] = {"ppl": round(ppl, 2), "regress": round(ppl / bare_ppl, 4)}
        total_ppl_sum.append(ppl)
        log(f"[{name}] 干扰: " + " ".join(f"{t}={res['matrix'][name][t]:.3f}" for t in res["skills"])
            + f" | 自己 A_gen {g_own:.3f} | ppl {ppl:.2f} ({ppl/bare_ppl:.3f}×)")
        m.unload()                      # 剥掉 adapter，保持 base 干净，供下一个技能挂载
        m = None
        torch.cuda.empty_cache()

    # 干扰量化：⚠️ 分母必须是**裸基座**，不是"自己的准确率"。
    # 自己的 0.94~1.00 里含技能特有的词表映射，跨技能不可能达到，用它当分母会把"正常"读成"干扰"。
    bare_m = res["matrix"]["bare"]
    transfer, degrade = {}, {}
    for a in res["skills"]:
        for b in res["skills"]:
            if a == b:
                continue
            cross = res["matrix"][a][b]
            transfer[f"{a}→{b}"] = round((cross - bare_m[b]) / max(res["matrix"][b][b] - bare_m[b], 1e-9), 3)
            degrade[f"{a}→{b}"] = round((bare_m[b] - cross) / max(bare_m[b], 1e-9), 3)
    res["transfer_ratio_vs_bare"] = transfer
    res["degradation_vs_bare"] = degrade
    res["bare_ppl"] = round(bare_ppl, 2)
    worst = max(degrade.values())
    res["verdict"] = (
        f"✅ 技能互不破坏（最坏破坏率 {worst:+.2f}，<=0 表示交叉准确率高于裸基座）；"
        f"格式迁移 {min(transfer.values()):.2f}~{max(transfer.values()):.2f}，"
        f"但远低于各自技能 → 仍需挂载/路由区分"
        if worst <= 0.05 else
        f"⚠️ 存在真实干扰（最坏破坏率 {worst:+.2f}），需要隔离挂载或逐技能路由")
    log(f"\n[干扰保持率] {inter}")
    log("[判定] " + res["verdict"])
    p = os.path.join(ART, "e1_multi_skill_qwen4b.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("→", p)


if __name__ == "__main__":
    main()
