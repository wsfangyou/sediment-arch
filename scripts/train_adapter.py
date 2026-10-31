# -*- coding: utf-8 -*-
"""Step 2/3 —— 在真实 Qwen2.5-3B-Instruct 上训练 Sedu 语技能 LoRA。

用法：
  python train_adapter.py teacher   # r=16 教师技能  -> teacher/adapter
  python train_adapter.py student   # r=8  行为蒸馏体 -> distilled/adapter_kd_r8

说明（对应方案 E1d 的“热蒸馏”真实模型版）：
· 教师 = 骨干从未见过的技能（程序化发明的密语翻译），LoRA r=16 全 linear 注入；
· 学生 = 同任务、同数据、秩减半 r=8 重训——这就是“从使用数据中蒸馏出
  更紧凑常驻表示”在真实模型上的实现形态（玩具尺度 E1b 已证“跨技能融合”
  不可行，故真实版只做单技能重蒸，符合方案改判）。
· loss 只算答案 token；chat template 与推理一致，评测无需域外技巧。
"""
import json, os, sys, argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct"
DEVICE = "cuda:0"
MAX_LEN = 192
SYS = "你是精通Sedu语与中文互译的语言助手。"

def build_features(tok, path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    feats = []
    for r in rows:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": r["q"]},
                {"role": "assistant", "content": r["a"]}]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids_full = tok(full, add_special_tokens=False)["input_ids"]
        ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids_full) > MAX_LEN:
            ids_full = ids_full[:MAX_LEN]
        labels = [-100] * min(len(ids_p), len(ids_full)) + list(ids_full[min(len(ids_p), len(ids_full)):])
        labels = labels[:len(ids_full)]
        feats.append({"input_ids": ids_full, "labels": labels})
    return feats

class DS(Dataset):
    def __init__(self, feats): self.f = feats
    def __len__(self): return len(self.f)
    def __getitem__(self, i): return self.f[i]

def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        n = m - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * n)
        out["labels"].append(b["labels"] + [-100] * n)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * n)
    return {k: torch.tensor(v) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teacher", "student"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "teacher", "adapter") if args.mode == "teacher" \
        else os.path.join(ROOT, "distilled", "adapter_kd_r8")
    r = 16 if args.mode == "teacher" else 8

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model.config.use_cache = False

    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=2 * r, lora_dropout=0.02,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    feats = build_features(tok, os.path.join(ROOT, "data", "sedu_train.jsonl"))
    print(f"[数据] {len(feats)} 条")
    dl = torch.utils.data.DataLoader(DS(feats), batch_size=args.bs, shuffle=True,
                                     collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps_per_ep = len(dl)
    total = steps_per_ep * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    model.to(DEVICE)
    model.train()
    losses = []
    gstep = 0
    for ep in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item()); gstep += 1
            if gstep % 50 == 0:
                w = losses[-50:]
                print(f"ep{ep} step{gstep}/{total} loss50={sum(w)/len(w):.4f} mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    model.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"mode": args.mode, "r": r, "epochs": args.epochs, "steps": gstep,
                   "final_loss50": sum(losses[-50:]) / 50}, f, indent=1)
    print("saved:", out_dir)

if __name__ == "__main__":
    main()
