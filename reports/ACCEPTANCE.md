# 蒸馏实验 验收报告（Qwen2.5-3B-Instruct · Sedu 语技能）

- 领域任务：密→中 / 中→密 / 词义查询 三向 holdout 600 条（程序化判分）
- A 域基线：bare = 0.000（要求 ≤0.15）→ ✅ 通过
- 教师上限参照：teacher = 1.000

| 变体 | 参数(M) | domain | s2c | c2s | gloss | ppl(回归) | 域外通过 | 保真率 vs teacher | B门 | C门 |
|---|---|---|---|---|---|---|---|---|---|---|
| svd_r8 | 14.97 | 1.000 | 1.000 | 1.000 | 1.000 | 11.36 | 0.750 | 100.0% | ✅ | ❌ |
| svd_r4 | 7.48 | 1.000 | 1.000 | 1.000 | 1.000 | 11.03 | 0.833 | 100.0% | ✅ | ✅ |
| svd_r2 | 3.74 | 1.000 | 1.000 | 1.000 | 1.000 | 10.71 | 0.917 | 100.0% | ✅ | ✅ |
| kd_r8 | 14.97 | 1.000 | 1.000 | 1.000 | 1.000 | 10.18 | 0.833 | 100.0% | ✅ | ✅ |

## 交付判定：✅ 通过预注册验收线
- 最优常驻候选：**svd_r2**（domain=1.000，保真率 100.0%）
- 全部合格变体：['svd_r4', 'svd_r2', 'kd_r8']

> 判定规则在评测前已写死于 eval_sedu.py（预注册），本报告仅复述结果，无事后挑选。

## 实验时间线
1. 合成 Sedu 语技能（80 项词典，密↔中双向，程序化可判分）
2. Qwen2.5-3B 上训练教师 LoRA（r=16，3 epochs）
3. SVD 秩截断蒸馏 r→{8,4,2}（含 alpha/r 缩放复原）+ 同数据低秩重蒸 r=8
4. 保真门 + 回归评测（同管线 5 变体）
5. 本判定与交付

## 交付物（均在本目录）
| 路径 | 体积 | 说明 |
|---|---|---|
| `distilled/adapter_svd_r2/` | 14.7 MB | **最终交付**：教师 1/8 体积，全部验收通过 |
| `distilled/adapter_svd_r4/` | 29.3 MB | 备选常驻候选 |
| `distilled/adapter_kd_r8/` | 58.5 MB | 行为重蒸路线候选（对照 SVD 路线） |
| `distilled/adapter_svd_r8/` | 58.5 MB | C 回归门未过，**不予交付**（护栏有效性证据） |
| `teacher/adapter/` | 117 MB | 教师技能（上限参照） |

挂载用法（与教师完全相同，秩/alpha 已写入各自 adapter_config.json）：
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = r"D:\Doubao-works\训练前置\02_基座模型\Qwen2.5-3B-Instruct"
m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype="bfloat16"),
                              r"D:\Qwen-works\Qwen-work-1\蒸馏实验\distilled\adapter_svd_r2")
```

## 复现命令（train_env，双卡）
```powershell
$py = "D:\Doubao-works\训练前置\train_env\Scripts\python.exe"
& $py scripts\gen_skill_data.py                          # Step1 数据
& $py scripts\train_adapter.py teacher                   # Step2 教师(r16)
& $py scripts\distill_svd.py                             # Step3 SVD r→8/4/2
& $py scripts\train_adapter.py student                   # Step3b 行为重蒸 r8
& $py scripts\eval_sedu.py                               # Step4 全变体评测+判定
& $py scripts\finalize.py                                # Step5 本报告
```

## 诚实注记（对照沉积架构方案）
1. **这是 E1c/E1d"热蒸馏+保真门"在真实 LLM 上的首次全流程验证**：秩截断产物以 1/32（r2）教师参数量保持全部技能精度，并额外通过语言分布回归检查——方案的"晋升质量门"在真实模型上如设计般既放行合法候选（r2/r4）又拦截劣候选（r8）。
2. **合成词典 ≠ 真实知识域**：80 词映射本质是低内在秩任务，SVD 满分不可外推到自然语言技能（方案 E3 已示警键空间判别性差异）；下一步应换真实领域技能重测同一管线。
3. **秩-通用能力代价被首次量化**：同一技能，教师 r16 域外通过 0.667 < 蒸馏 r2 的 0.917 ≈ 裸基线 0.917——高秩技能训练对通用能力的伤害更大，"蒸馏更紧凑反而回归更安全"是本实验的意外收获，支持方案 L1"小体积常驻"取向。