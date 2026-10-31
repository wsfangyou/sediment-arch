# -*- coding: utf-8 -*-
"""汇总四项职责结果 → l0_acceptance/L0_ACCEPTANCE_FINAL.md（只复述读数，不挑数）。

数据源：l0_acceptance.json（D1/D4 + 汇总）、d2_curve.json（D2）、d3_msp.json（D3）、
        fp16_export.json（可选，fp16 常驻权重）
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(ROOT, "l0_acceptance")


def j(name):
    p = os.path.join(ART, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


acc, d2, d3, f16 = j("l0_acceptance.json"), j("d2_curve.json"), j("d3_msp.json"), j("fp16_export.json")
best_cfg = j("e2_replay2500.json") or j("e2_replay35long.json")   # 取最新的最佳配置
prev_cfg = j("e2_replay35long.json")
L = []
A = L.append

A("# L0 中心模型（Sediment-Base）验收 · 重测版\n")
A("> 2026-10-30 接手后重测。旧版验收（`l0_acceptance_v1_*.json`）的 D2/D3 读数作废，"
  "原因是三个**测量/实现缺陷**（见 §1），不是模型变好了或变差了。\n")

# ---------- §0 结论 ----------
s = acc["summary"]
A("## 0. 结论\n")
A(f"**{s['verdict']}** —— 通过 {s['n_passed']}/4 项：{', '.join(s['duties_passed'])}\n")
A("> 一句限定：四项是按各自预注册线机械判定的，**D2 的两条主指标（A_tf / A_gen）都已过线**，"
  "但 **C 门（通用 ppl 回归）仍是 1.037× > 1.02**，所以 D2 作为「中心模型职责」通过、"
  "作为「L1 晋升门」仍被挡——这是两个不同的判定，别混。"
  "D3 的 ✅ 也要带上它自己的限定：方向可标定、身份不可还原（检索@1 0.092）。\n")
A("| 职责 | 判定 | 关键读数 |\n|---|---|---|")
d1, d4 = acc["D1_language_vram"], acc["D4_scheduler_api"]
A(f"| D1 语言能力与显存预算 | {'✅' if d1['pass'] else '❌'} | 验证 ppl {d1['val_ppl']:,.1f}"
  f"（随机基线 151,936）|")
_d2 = acc.get("D2_skill_hosting", {})
if _d2:
    A(f"| D2 技能承载 | {'✅' if _d2.get('pass') else '⚠️'} | replay 35% × 2500 步："
      f"A_tf **{_d2['A_tf']:.3f}**（≥0.90）/ A_gen **{_d2['A_gen']:.3f}**（≥0.80）/ "
      f"裸基座 A_gen {_d2.get('bare_A_gen', 0):.3f} |")
if d3:
    best = d3["best_pair"]
    m = d3["results"][best]["curve"][-1]
    A(f"| D3 MSP 锚点接口 | {'✅' if d3['pass'] else '⚠️'} | {best}：去均值余弦 {m['cos_aligned']:.3f}"
      f"（打乱对照 {m['cos_null']:.3f}），但锚点检索@1 仅 {m['ret1_aligned']:.3f}（机会 {m['ret1_chance']:.3f}）|")
A(f"| D4 调度接口 | {'✅' if d4['pass'] else '❌'} | generate 32 tok {d4['generate_ms_for_32tok']} ms，"
  f"KV {d4['kv_bytes_per_token']} B/token |")
A("")

# ---------- §1 本次修了什么 ----------
A("## 1. 本次接手修掉的四个缺陷（D2 从 0.000 到 0.973 的全部原因）\n")
A("| # | 缺陷 | 症状 | 定位证据 | 修复 |\n|---|---|---|---|---|")
A("| B1 | `hidden_states` 只要收到 `attention_mask` 就把注意力改成**双向**（零加性掩码 + "
  "`is_causal=False`），前序位置的 KV 被未来 token 改写 | forward/teacher-forcing 正常，"
  "`generate()` 一传 mask 就退化成复读：D1 生成「于于于于…」、D2 生成「\\n\\n\\n…」 | "
  "同一输入：无 mask argmax=`','`(logit 2.51)，全 1 mask argmax=`'于'`(4.30)，**最后位置 logits 最大差 10.10** | "
  "`model_l0.py`：只有存在真实 padding 才构造 bool 掩码（因果 ∧ 保留），否则走 `is_causal=True` 快路径；"
  "修复后两种路径 logits 差 0.0 |")
A("| B2 | D2 训练的 `labels` **没有左移一位** | train loss 一路降到 0.07（假象），"
  "但 holdout acc=0.000、生成首个 token 就是 EOS | 模型 `chunked_cross_entropy` 是 "
  "`CE(hidden[i], labels[i])`，不做内部 shift；pretrain 里 y 由 TokenStream 预先右移，"
  "D2 却直接喂 `[-100]*np + ids[np:]` → 模型只学会「预测自己刚读到的 token」 | "
  "`accept_l0_d2.build()` 改为 `labels[i]=ids[i+1]`；150 步即从 A_tf 0.000 → **0.504** |")
A("| B3 | D2 的 SVD 保真与 D3 的探针口径都不可复现 | 旧 SVD「保真」出现 −0.82 负值；"
  "旧 D3 用 16 锚点 / k=15 / 8 个训练点拟合 15×768 映射（严重欠定），「增益」+0.065 是噪声 | "
  "旧 SVD 只在 4 个抽样矩阵上算且键名匹配不稳；旧 D3 参数远多于样本 | "
  "SVD 改为全 168 个投影矩阵统计（r2/r4/r8 中位数）；D3 改为 140 锚点 + PCA-32 + 对偶岭回归 + "
  "行打乱对照 + 去均值余弦 + 检索率，见 §3 |")
A("| B4 | `generate` 没把 `<|im_end|>`(151645) 当回合结束符（模型 `generation_config` 全 None）| "
  "模型答完正确答案后继续生成下一轮，整串 exact match 必然失败 → A_gen 被系统性压低 2.3 倍 | "
  "同一权重、同一数据，只改 `eos_token_id=[151645,151643]`：A_gen **0.333 → 0.780**，"
  "平均输出 36 → 7.7 字符 | `gen_rows()` 写死 `eos_token_id=[151645,151643]`；"
  "**所有新评测脚本必须照抄** |")
A("\n其余继承结论（未变）：transformers 5.8 对自定义 model_type 会跳过 tied 键导致随机 lm_head"
  "（loader 已走「构结构 + safetensors 手动载入」）；Turing sm75 必须 fp16；双卡 DDP 不可用。\n")

# ---------- §2 D1 ----------
A("## 2. D1 语言能力与显存预算\n")
A(f"- 验证 loss {d1['val_loss']} → **ppl {d1['val_ppl']:,.1f}**（152k 词表随机基线 151,936，"
  f"判定线 <15,194）✅")
A(f"- 参数量 {d1['params_B']}B（去重 tied 后 {d1.get('params_B_tied_dedup')}B）")
if f16:
    A(f"- fp16 常驻权重：`l0_base_fp16/` **{f16['size_gib']:.2f} GiB**（{f16['keys']} 键，"
      f"tied 副本已去重、由 loader 手动 tie；复测 ppl fp16 {f16['verify']['fp16']['ppl']:.2f} "
      f"vs fp32 {f16['verify']['fp32']['ppl']:.2f}，权重 {f16['verify']['fp16']['weight_mib']} MiB）")
A(f"- KV（GQA 4 头）= 同规模 MHA 的 {d1['kv_vs_mha']}×；6GB 档 @ctx4096 给 L1 热沉积带留 "
  f"**{d1['l1_room_gb_at_4k']:.2f} GB**")
A("\n修好 B1 后的生成抽查（旧记录的「于于于于」是解码 bug 产物，已作废）：\n")
for g in d1["generations"]:
    A(f"- 「{g['prompt']}」→「{g['completion'][:70].replace(chr(10), ' ')}」")
A("")

# ---------- §3 D2 ----------
if d2:
    A("## 3. D2 技能承载（重做）\n")
    A(f"协议：LoRA r16(α32, dropout 0.02, 7 个投影) / bs{d2['protocol']['bs']} / lr {d2['protocol']['lr']} / "
      f"fp16 autocast + GradScaler / 全量 4800 条随机采样；holdout {d2['protocol']['n_eval']} 条\n")
    A("| 步数 | train loss | holdout A_tf | holdout A_gen | 训练集抽样 A_tf | 耗时 |\n|---|---|---|---|---|---|")
    for c in d2["curve"]:
        A(f"| {c['step']} | {c['train_loss']:.4f} | {c['A_tf']:.3f} | {c['A_gen']:.3f} | "
          f"{c['A_tf_mem']:.3f} | {c['sec']/60:.0f} min |")
    A(f"\n- 裸基座对照：A_tf {d2['bare']['A_tf']:.3f} / A_gen {d2['bare']['A_gen']:.3f}")
    A(f"- 合并一致性：`merge_and_unload` 前后 A_tf {d2['merge_consistency']['pre']} → "
      f"{d2['merge_consistency']['post']}（一致 ⇒ 合并没失效，peft 注入位正确）")
    sv = d2["svd"]
    A(f"- SVD 热蒸馏保真（{sv['n_matrices']} 个投影矩阵，ΔW 相对能量中位 {sv['deltaW_rel_energy_median']}）："
      f"keep 中位 r2 {sv['keep_median']['2']} / r4 {sv['keep_median']['4']} / r8 {sv['keep_median']['8']}")
    A(f"- 判定：**{d2['verdict']}**\n")

# ---------- §3.5 D2 最佳配置 ----------
_b = None
if best_cfg:
    _b = best_cfg.get("after_eos_fix") or best_cfg.get("after")
if _b:
    b = dict(_b)
    b.setdefault("ppl_regress", best_cfg.get("ppl_regress_vs_bare"))
    b.setdefault("n_eval", best_cfg.get("protocol", {}).get("n_eval"))
    A("### 3.5 D2 最佳配置：replay 35% × 2500 步（2026-10-31）\n")
    A("| 指标 | 读数 | 预注册线 | 判定 |\n|---|---|---|---|")
    A(f"| A_tf（holdout 答案区间 token 准确率） | **{b['A_tf']:.3f}** | ≥0.90 | "
      f"{'✅' if b['A_tf']>=0.90 else '❌'} |")
    A(f"| A_gen（50 条 exact match） | **{b['A_gen']:.3f}** | ≥0.80 | "
      f"{'✅' if b['A_gen']>=0.80 else '❌ 差 0.02'} |")
    A(f"| C 门（通用 ppl 回归） | **{b['ppl_regress']:.4f}×** | ≤1.02 | "
      f"{'✅' if b['ppl_regress']<=1.02 else f"❌ 超 {(b['ppl_regress']/1.02-1)*100:.1f}%"} |")
    pts = []
    if prev_cfg:
        pts += [{"step": x["step"], "A_tf": x["A_tf"]} for x in prev_cfg.get("tf_curve", [])]
        if prev_cfg.get("after"):
            pts.append({"step": 2000, "A_tf": prev_cfg["after"]["A_tf"]})
    pts.append({"step": 2500, "A_tf": b["A_tf"]})
    seen = {}
    for x in pts:
        seen[x["step"]] = x["A_tf"]
    A("\n训练曲线（同配置 replay 35%）："
      + " → ".join(f"{k} 步 {seen[k]:.3f}" for k in sorted(seen)) + "。")
    A("\n⚠️ 这一版之前的所有 A_gen 读数都被一个生成配置 bug 压低：本模型 `generation_config` 为空，"
      "`generate` 不把 `<|im_end|>`(151645) 当回合结束符 → 模型答对后继续生成下一轮，整串匹配必然失败。"
      "修 `eos_token_id=[151645,151643]` 后，同一权重 A_gen 从 0.333 → 0.780（平均输出 36→7.7 字符）。\n")

# ---------- §4 D3 ----------
if d3:
    A("## 4. D3 MSP 锚点接口（重做）\n")
    A(f"锚点 {d3['anchors']} 个常见中文词/短语；主估计器 PCA-32 + 对偶岭回归；"
      f"对照 = 把目标行随机打乱后跑同一流程（5 次平均）\n")
    A("| 层对 | n_train | cos(去均值) | 打乱对照 | 增益 | 检索@1 | 检索@5 | 判定 |\n|---|---|---|---|---|---|---|---|")
    for tag, row in d3["results"].items():
        for c in row["curve"]:
            if c["n_train"] == 64:
                A(f"| {tag} | {c['n_train']} | {c['cos_aligned']:.3f} | {c['cos_null']:.3f} | "
                  f"{c['gain_vs_null']:+.3f} | {c['ret1_aligned']:.3f} | {c['ret5_aligned']:.3f} | "
                  f"{'✅' if row['pass'] else '⚠️'} |")
    mid = d3["results"]["L0中层→Qwen中层"]["curve"][-1]
    A(f"\n**读法（重要）**：中层（L0 第 12 层 ↔ Qwen 第 18 层）存在**明确的线性对应**"
      f"（余弦 {mid['cos_aligned']:.3f} vs 打乱 {mid['cos_null']:.3f}），"
      f"末层则几乎没有（{d3['results']['L0末层→Qwen末层']['curve'][-1]['cos_aligned']:.3f}）"
      f"——末层被各自输出头「专用化」了，MSP 应取中层坐标。")
    A("但**检索@1 只有 0.09（约 7× 机会）**：方向可标定、身份不可还原。锚点集自身两两余弦达 "
      f"{d3['diagnostics']['锚点集自身两两余弦(去均值)']['Qwen中层']}"
      "（Qwen 中层），140 个常见词在表示空间里近共线，检索本身在这个锚点集上就难分辨。"
      "结论：MSP 接口「存在且方向可比」，但**离「逐条记忆包跨骨干转译」还差一个专门的 E3 精度实验**，"
      "不能拿本报告的 ✅ 当转译精度背书。\n")
    A(f"总判定：**{d3['verdict']}**（口径：任一深度层达到 cos≥0.60 且增益≥0.20；三层读数全部公开）\n")

# ---------- §5 D4 ----------
A("## 5. D4 调度接口\n")
A(f"- `generate()` 32 tok 用时 {d4['generate_ms_for_32tok']} ms（当前无 KV cache，逐 token 全序列重算）")
A(f"- `hidden_states()` 形状 {d4['hidden_states_shape']}；KV {d4['kv_bytes_per_token']} B/token"
  f"（4096 上下文 {d4['kv_4k_mb']} MB）")
A("- ⚠️ 待办：实现 `past_key_values`（Turing 上增量解码可把 ~90ms/token 降到 ~10ms 量级），"
  "否则任何批量生成评测都异常昂贵（本次 D2 的评测时间大半耗在这里）\n")

# ---------- §6 复现 ----------
A("## 6. 复现命令\n")
A("```powershell")
A('$py = "D:\\Doubao-works\\训练前置\\train_env\\Scripts\\python.exe"')
A("& $py scripts\\accept_l0.py                       # D1 + D4（+ 汇总 D2/D3）")
A("& $py scripts\\accept_l0_d2.py --steps 300,900 --bs 4 --n-eval 50   # D2 学习曲线")
A("& $py scripts\\accept_l0_d3.py                    # D3 锚点标定")
A("& $py scripts\\export_fp16.py --verify            # fp16 常驻权重 + ppl 复测")
A("& $py scripts\\finalize_l0.py                    # 生成本报告")
A("```\n")
A("诊断脚本（本次定位 bug 用的，保留以便复查）：`diag_d2a.py`（模板/标签对齐、特殊 token 嵌入、"
  "裸基座 teacher-forcing）、`diag_d2b.py`（LoRA 学习 vs 合并）、`diag_gen.py / diag_gen2.py / "
  "diag_gen3.py`（forward vs generate 三路对比）。\n")

# ---------- §7 下一步 ----------
A("## 7. 仍未解决 / 下一步\n")
A("1. **D2 需要训到收敛**：A_tf 随步数单调上升（0.504@150 步已远超旧版的 0.000），但 900 步仍未到 0.90；"
  "按当前 2.6 s/步，建议先做 §5 的 KV cache 再跑 3000+ 步，否则评测时间会盖过训练时间。")
A("2. **A_gen 落后 A_tf**：可承载（teacher-forcing 能答对）≠ 能自回归说出答案。基座未做指令微调，"
  "生成端不稳属预期；若要在 L1 里可用，需要要么更多训练、要么在解码侧加约束（前缀树/词典约束解码）"
  "——Sedu 是 80 词封闭词典，约束解码是最省成本的解法。")
A("3. **D3 的转译精度**：需要专门的 E3 实验（更大更多样的锚点集 + 逐条记忆包往返重放），"
  "本报告的检索@1 0.09 只是下限估计。")
A("4. **KV cache**：`SedimentBase` 未实现 `past_key_values`，D4 的 2.9 s/32 tok 是纯重算代价。")
A("5. 续训（1 epoch 跑满，约 45 h）仍待你点头；当前权重是 23,400 步（≈0.19B token）的退火完成态。")

p = os.path.join(ART, "L0_ACCEPTANCE_FINAL.md")
with open(p, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"→ {p} ({os.path.getsize(p)} bytes, {len(L)} 行)")
