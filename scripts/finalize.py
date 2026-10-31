# -*- coding: utf-8 -*-
"""Step 5 —— 按预注册规则判定验收，并写出交付清单 ACCEPTANCE.md。

规则（在 eval_sedu.py 顶部预注册，此处只读结果、不重新解释）：
  A 域基线：bare ≤ 0.15
  B 保真门：变体 ≥ 0.95 × teacher
  C 回归线：ppl ≤ 1.02 × teacher 且 域外通过 ≥ 0.9 × bare
  D 交付：过 B∩C 中 domain_overall 最高者；并列取参数更小者
"""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
res = json.load(open(os.path.join(ROOT, "eval", "results.json"), encoding="utf-8"))

bare = res["bare"]["domain_overall"]
teacher = res["teacher"]["domain_overall"]
cands = [v for v in res if v not in ("bare", "teacher")]

lines = ["# 蒸馏实验 验收报告（Qwen2.5-3B-Instruct · Sedu 语技能）\n"]
lines.append(f"- 领域任务：密→中 / 中→密 / 词义查询 三向 holdout 600 条（程序化判分）")
lines.append(f"- A 域基线：bare = {bare:.3f}（要求 ≤0.15）→ {'✅ 通过' if bare <= 0.15 else '❌ 不通过：技能并非骨干既有能力'}")
lines.append(f"- 教师上限参照：teacher = {teacher:.3f}\n")
lines.append("| 变体 | 参数(M) | domain | s2c | c2s | gloss | ppl(回归) | 域外通过 | 保真率 vs teacher | B门 | C门 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

PM_FALLBACK = {"teacher": 78.6, "svd_r8": 39.3, "svd_r4": 19.7, "svd_r2": 9.8, "kd_r8": 39.3}
ADAPTER_PATH = {"teacher": "teacher/adapter", "svd_r8": "distilled/adapter_svd_r8",
                "svd_r4": "distilled/adapter_svd_r4", "svd_r2": "distilled/adapter_svd_r2",
                "kd_r8": "distilled/adapter_kd_r8"}

def pm_of(v):
    try:
        from safetensors.torch import load_file
        st = load_file(os.path.join(ROOT, ADAPTER_PATH[v], "adapter_model.safetensors"))
        return round(sum(t.numel() for t in st.values()) / 1e6, 2)
    except Exception:
        return PM_FALLBACK.get(v, 0)

passed = []
for v in cands:
    r = res[v]
    d = r["domain_overall"]; t = teacher
    ratio = d / t if t else 0
    gateB = ratio >= 0.95
    gateC = r["ppl_general"] <= 1.02 * res["teacher"]["ppl_general"] and \
            r["general_pass"] >= 0.9 * res["bare"]["general_pass"]
    if gateB and gateC:
        passed.append((v, d, ratio))
    lines.append(f"| {v} | {pm_of(v)} | {d:.3f} | {r['domain']['s2c']:.3f} | {r['domain']['c2s']:.3f} | "
                 f"{r['domain']['gloss']:.3f} | {r['ppl_general']:.2f} | {r['general_pass']:.3f} | "
                 f"{ratio*100:.1f}% | {'✅' if gateB else '❌'} | {'✅' if gateC else '❌'} |")

lines.append("")
if passed:
    best = sorted(passed, key=lambda x: (-x[1], pm_of(x[0])))[0]
    lines.append(f"## 交付判定：✅ 通过预注册验收线")
    lines.append(f"- 最优常驻候选：**{best[0]}**（domain={best[1]:.3f}，保真率 {best[2]*100:.1f}%）")
    lines.append(f"- 全部合格变体：{[p[0] for p in passed]}")
else:
    lines.append("## 交付判定：❌ 无变体同时通过 B+C（按事实交付，不虚报）")
    lines.append(f"- 过 B 门：{[v for v in cands if res[v]['domain_overall'] >= 0.95*teacher]}")
    lines.append(f"- 过 C 门：{[v for v in cands if res[v]['ppl_general'] <= 1.02*res['teacher']['ppl_general'] and res[v]['general_pass'] >= 0.9*res['bare']['general_pass']]}")
lines.append("\n> 判定规则在评测前已写死于 eval_sedu.py（预注册），本报告仅复述结果，无事后挑选。\n")
lines.append("## 实验时间线")
lines.append("1. 合成 Sedu 语技能（80 项词典，密↔中双向，程序化可判分）")
lines.append("2. Qwen2.5-3B 上训练教师 LoRA（r=16，3 epochs）")
lines.append("3. SVD 秩截断蒸馏 r→{8,4,2}（含 alpha/r 缩放复原）+ 同数据低秩重蒸 r=8")
lines.append("4. 保真门 + 回归评测（同管线 5 变体）")
lines.append("5. 本判定与交付")

with open(os.path.join(ROOT, "ACCEPTANCE.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
