# Sediment Architecture · 沉积式技术

**English** | [中文](README.zh-CN.md)

> **A mechanism for plugging skills in and out of an already-trained model.** Each skill is a small
> module (12 MB at rank 8 on the 0.83B backbone): plug it in and it works, unplug it and everything
> is restored, and skills never damage each other.
> Deploy from 2.5 GB of VRAM; train from 6.4 GB.
>
> **It does not make the model smarter.** This is the single most important conclusion in this
> repository, and the thing we spent the most time measuring.

---

## 0. What it is, in one sentence

Giving a model a new craft — your company's internal jargon, the exact way a form must be filled,
a constructed language — traditionally leaves two bad options: retrain the whole model (expensive,
slow, and it erodes what the model already knew), or stuff instructions into the prompt (eats
context, forgotten once the conversation grows).

**Sediment proposes a third path: package the craft as a small plug-in and plug it into the model.**
Frequently used plug-ins stay resident in VRAM; rarely used ones sink to disk and are plugged back in
when needed — the way an operating system manages memory pages.

What it sells is **not "being smarter", but "cheaper, hot-pluggable, mutually harmless crafts".**

---

## 1. What this repository is / is not

| | |
|---|---|
| **Is** | A complete experimental record on real models (including many negative results), reproducible code, grading protocols, and an end-to-end run of a self-built 0.83B backbone |
| **Is** | Two parallel validation tracks: **Track A** — self-built 0.83B backbone; **Track B** — Qwen3-4B as a reference pick |
| **Is not** | A new pretrained model (the self-built L0 saw only 0.97B tokens — enough to validate the mechanism, not enough to ship) |
| **Is not** | A "small model catches up with large models" proposal — we measured it and it is false, see §4 |

---

## 2. Results at a glance

### 2.1 Four duty checks (both backbones pass)

| Duty | Track A: self-built L0 0.83B | Track B: Qwen3-4B |
|---|---|---|
| D1 language ability (ppl) | 56.9 | **16.76** |
| D2 skill hosting (A_tf / A_gen) | 0.973 / 0.880 | **0.994 / 0.960** |
| D2's C-gate (general regression, strict line 1.02) | **1.0223×** (after a calibration pass) | **0.861× (it actually improves)** |
| D3 MSP anchors (project criterion: cos ≥ 0.60) | 0.974 | 0.983 |
| D4 scheduling interface | ✅ | ✅ (native KV cache) |

Grading protocol lives in `scripts/accept_l0_d2.py`: A_tf = per-token accuracy over the answer span
(≥ 0.90); A_gen = exact-match rate of the full answer (≥ 0.80); bare backbone A_gen ≤ 0.15.

### 2.2 What the skills actually achieve (E1 multi-skill loop)

Three mutually distinct constructed-language skills (Sedu / Nuri / Tavi) were trained independently,
then mounted one at a time and evaluated across all three:

| Mounted → | on sedu | on nuri | on tavi | own A_gen |
|---|---|---|---|---|
| bare backbone | 0.333 | 0.366 | 0.364 | — |
| sedu | **0.995** | 0.627 | 0.614 | 0.960 |
| nuri | 0.641 | **0.996** | 0.638 | 0.980 |
| tavi | 0.588 | 0.684 | **0.944** | 0.660 |

**Zero cross-skill damage**: every cross accuracy is *higher* than the bare backbone (damage ratio
−0.69 to −0.92). There is a 0.39–0.50 "format transfer" effect (learning one skill makes answers to
the others better formatted), but no transferable knowledge.

> Measurement note: the denominator for interference must be the **bare backbone**, not "the skill's
> own accuracy" — the latter contains that skill's private vocabulary mapping and can never be reached
> across skills. Using it makes a normal result look like interference. We stepped on this ourselves.

Deposit size scale (fp16, measured):

| Backbone | r16 | r8 | r4 | r2 (fidelity 1.008) |
|---|---|---|---|---|
| self-built L0 0.83B | 24.2 MB | **12.1 MB** | 6.1 MB | **3.0 MB** (339 per GB) |
| Qwen3-4B | 63.5 MB | — | — | — |

Rank truncation is nearly free: on the 0.83B backbone, truncating r16 down to r2 keeps behavioural
fidelity at **1.008** (A_tf criterion) while shrinking the deposit to 1/8.

### 2.3 Deployment and training cost (measured on a single RTX 2080 Ti 22GB)

| Item | Value |
|---|---|
| 4B skill plug-in deployment (NF4 4-bit) | **2.50 GiB** resident, 0.56 GiB KV at ctx 4096; a 6 GB budget still leaves 2.6 GB for the skill area |
| Effect of quantization on the skill | A_tf 0.9939 / A_gen 0.960 — **bit-identical to fp16** |
| 4B LoRA training (fp16) | peak **16.8 GiB**, bs4×2048, 565 tok/s |
| 4B QLoRA training (NF4) | peak **6.43 GiB**, same quality → **an 8 GB GPU can train 4B LoRA** |
| 4B full fine-tuning | ❌ 22.5 GiB minimum before activations; not feasible on one card |

---

## 3. The most valuable part: **negative results**

These are the things we spent time disproving. They are written down so nobody else has to repeat them.

| # | Finding | Evidence |
|---|---|---|
| N1 | **Cross-backbone knowledge packs (MSP/ISA) do not work** | Two backbone pairs, same-family and cross-family: b/a only **0.00–0.04**, identical to a random projection (chance 0.021). Mechanism: calibration error angle 10–15° ≈ inter-class angle 13–16°, so SNR ≈ 1 |
| N2 | **Mounting skills does not raise intelligence; it lowers reasoning** | Same 4B: 3-shot reasoning probe 0.533 → **0.400** after mounting a skill; in chat mode it drops to chance (0.250) |
| N3 | **Anchor *distribution* matters far more than anchor count** | In-distribution anchors give calibration cosine 0.977; 200 generic-corpus anchors give only 0.640 |
| N4 | **Single-paragraph ppl underestimates "skill hurts general ability" by 8×** | Single paragraph says 1.043; the 98k-token validation set says 1.338 → the C-gate must use a validation-set protocol |
| N5 | **Gradient checkpointing does not save activations in the PEFT setting** | Measured ≈1.0–1.3 GiB per 1000 tokens, 5–6× above theory; both `use_reentrant` implementations behave the same |
| N6 | **4-bit quantization is not the deciding factor on a 22 GB card** | QLoRA saves only 3.4 GiB but is 9% slower, and large batches still OOM; the bottleneck is activations, not weights |

There are also reusable **positive** findings: `A_gen ≈ A_tf^5.85` (the exponent is the answer length in
tokens), replay mixing does no harm to the skill once the skill sample count is restored (it is a free
regularizer), and a general-domain calibration pass rescues the C-gate cheaply — with a U-shaped curve
that requires early stopping.

---

## 4. What it cannot do (hard boundary)

**It does not raise intelligence.** Local probe (programmatically generated and graded, chance 0.25):

| Model | Overall |
|---|---|
| self-built L0 0.83B | 0.283 |
| Qwen3-4B base | 0.533 |
| **Qwen3-4B + mounted skill** | **0.400 ↓** |
| Qwen3-4B-Instruct-2507 | 0.625 |

Public benchmarks (Oct 2026 figures):

| Model | GPQA Diamond | AIME25 |
|---|---|---|
| Qwen3-4B-Thinking-2507 | 65.8 | 81.3 |
| DeepSeek-V4.1-Flash (552B total / 8–16B active) | **90.9** | **99.9** |
| Kimi K3 (2.8T total / 104B active) | **93.5** | 89.0 |

Conclusion: **sediment is a skill layer, not a capability layer.** It is meant for "installing crafts
on a base that is already smart enough", not for "training a small model into a smart one". If you need
production-grade intelligence on this machine, use an API or a larger backbone and stack this mechanism
on top as a skill scheduler — which was the original design intent anyway (L0 schedules; it does not
reason).

One more measured fact worth remembering: **4B reasoning depends entirely on explicit chain-of-thought.**
A four-step word problem gets 12/12 when given 400 tokens to write out the work, and 0/12 when
forbidden from writing any process at all (it cannot even emit the answer format). The "reasoning
effort" knob on production models is, in essence, chain-of-thought length.

---

## 5. Repository layout

```
docs/      8 reports (Chinese): design → technical explainer → feasibility → execution log → 4B evaluation → intelligence comparison
scripts/   all experiment scripts (training / grading / probes / benchmarks), 40 in total, directly runnable
data/      three constructed-language skill datasets (Sedu / Nuri / Tavi, with lexicons), uniform format, programmatically gradable
results/   every raw reading (JSON); each number in the reports can be traced back here
reports/   handoff document and acceptance reports (HANDOFF / ACCEPTANCE)
toy/       early toy validation (dependency-free Node prototype + PyTorch d=96 skeleton); the mechanism prototype lives here
```

Key scripts:

| Script | Purpose |
|---|---|
| `accept_l0_d2.py` | D2 grading (A_tf / A_gen / C-gate); **every skill training run is graded with it** |
| `accept_l0_d3.py` / `e3_msp_split.py` / `e4_msp_qwen.py` | MSP anchor and ISA transplant experiments (the evidence for N1) |
| `e2_replay.py` | Replay mixing + general calibration pass (C-gate 1.338 → 1.0223) |
| `e1_l1_deposit.py` | Skill training → rank truncation → fidelity / general regression |
| `e1_multi_skill.py` | Multi-skill interference matrix (§2.2) |
| `bench_qwen_d2.py` / `bench_qwen_qlora.py` | Port the same grading to HF models; QLoRA training |
| `bench_4b_single_card.py` | Single-card 4B training feasibility sweep (VRAM / throughput) |
| `bench_reasoning_probe.py` / `bench_reasoning_extra.py` | The intelligence probe (§4) |

---

## 6. Quick start

Environment: Python 3.13 + `torch 2.11 (cu128)` + `transformers 5.8` + `peft 0.18` + `bitsandbytes 0.50`.
A single 22 GB GPU runs every experiment here (two are faster).

```bash
pip install -r requirements.txt

# 1) Regenerate the three skill datasets (already included in data/, so this is optional)
python scripts/gen_multi_skill.py

# 2) Train one skill on Qwen3-4B (~23 min, 16.8 GiB peak on one card)
python scripts/bench_qwen_d2.py --model <path-to-Qwen3-4B> --tag demo \
    --steps 800 --bs 4 --n-eval 50 --replay 0.35 --save-adapter out/demo

# 3) Multi-skill interference matrix (needs three adapters)
python scripts/e1_multi_skill.py --mode eval

# 4) Single-card 4B feasibility, including the full fine-tuning OOM boundary
python scripts/bench_4b_single_card.py --mode lora --bs 4 --seq 2048 --chunk-ce
```

> Two pitfalls, up front:
> 1. **Qwen3 is a thinking model** and its chat template inserts a ` thinking` block automatically.
>    Evaluation must pass `enable_thinking=False` or strip the block, otherwise A_gen is under-counted
>    by 2.3× (0.960 → 0.320).
> 2. **Grading must pass `eos_token_id` explicitly** (`[151645, 151643]` for Qwen). Otherwise the model
>    keeps generating the next turn after the correct answer and whole-string matching always fails.

---

## 7. What is not included (and how to get it)

| Item | Size | Note |
|---|---|---|
| Self-built L0 weights (0.828B) | 4 GB (fp32) / 1.54 GB (fp16) | Not uploaded; GitHub is the wrong place for weights. Retrain with `scripts/pretrain_l0.py` (~45 h), or download from Hugging Face (pending) |
| Pretraining corpus `tokens.bin` | 3.6 GB | Derived from `opencsg/chinese-fineweb-edu-v2`; rebuild with `scripts/prepare_corpus.py` |
| Skill adapters (12–127 MB each) | ~700 MB total | Not uploaded (a single file over 100 MB hits GitHub's limit). Training scripts and configs are in the repo, so they reproduce exactly |
| Qwen backbones | — | Fetch from Hugging Face: `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Instruct-2507` |

---

## 8. Citation and license

- **Code**: Apache License 2.0 (see `LICENSE`)
- **Docs and data** (`docs/`, `data/`, `results/`, `reports/`): CC BY 4.0
- **Third parties**: experiments are built on Qwen backbones (Apache-2.0) and the
  `opencsg/chinese-fineweb-edu-v2` corpus; see `NOTICE` for details

If the negative results here saved you some time, the citation format is in `CITATION.cff`.

---

## 9. A note on honesty

Every experiment in this repository ran on a single dual-2080-Ti machine (22 GB modded cards). Sample
sizes are small (30–60 items per grading run) and a single number carries roughly ±0.03 of jitter.
We tried to do three things consistently: **pre-register the decision thresholds, always include a
shuffled/random control, and publish the failures too.** Numbers marked "measured" can be traced to a
file under `results/`; numbers marked "public" come from vendor or third-party evaluations and were
not reproduced on this machine.
