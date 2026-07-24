# Project Plan — QLoRA Fine-Tuned VLM for Document Field Extraction

**Goal:** one honestly-measured fine-tune, an evaluation harness you designed
yourself, and the ability to explain every choice you made.

Not "become a deep learning researcher." That distinction is the whole plan.

---

## 0. Decisions locked

Settle these now so you're not re-litigating them at 2am on day 4.

| Decision | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` | Current small Qwen VLM; strong OCR; fits 4-bit on a free-tier GPU |
| Model class | `Qwen3VLForConditionalGeneration` | Needs **transformers ≥ 4.57** — older versions error with "does not recognize this architecture" |
| Dataset | **SROIE** (ICDAR 2019 Task 3) | 4 flat fields. CORD is richer but hierarchical, and parsing/scoring nested JSON will eat a day you don't have |
| Method | 4-bit QLoRA, LoRA on the LLM decoder only | ViT and the vision–language merger stay frozen |
| Metrics | Field-level exact match + ANLS | Reported on clean **and** degraded splits |
| Hardware | Colab T4 / Kaggle T4 (~30 GPU-hr/week) | L4 or A100 if you can get one — see the bf16 note below |
| Tracking | Weights & Biases | Free tier; gives you the loss curve screenshot for the README |

**Update from my earlier advice:** I'd said Qwen2-VL-2B. Qwen3-VL-2B-Instruct
supersedes it and is the better pick. Change the model name on your resume too.

**One consequence worth knowing up front:** Qwen3-VL's OCR is explicitly built to
be robust to low light, blur, and tilt. So your degraded split may show a
*smaller* gap than you expect. That is a legitimate finding, not a failed
experiment — "the base model's robustness training already covers most of my
corruptions; the remaining gap concentrates in X" is a genuinely interesting
sentence. Do not turn up the corruption severity until you get the drop you were
hoping for. That's fitting the experiment to the conclusion.

---

## 1. The two meanings of "fields"

### 1a. Data fields — what you're extracting

**SROIE** gives you four flat fields per receipt:

| Field | Content | Difficulty |
|---|---|---|
| `company` | Vendor / merchant name | Medium — often stylised logo text |
| `date` | Transaction date | Easy — but format normalisation matters |
| `address` | Full street address | **Hardest** — multi-line, wraps unpredictably |
| `total` | Final amount | Easy-ish — but confusable with subtotal / tax lines |

Your model's target output, one JSON object per image:

```json
{"company": "...", "date": "...", "address": "...", "total": "..."}
```

That four-field spread is genuinely useful: you get an easy field, a hard field,
and a numeric field, so your per-field results table will actually say something.
Predict where errors will concentrate before you measure (`address`, almost
certainly) and write the prediction down — being right is evidence you understand
the task; being wrong is more interesting still.

**CORD** as a stretch goal only: ~30 subclasses in a nested schema
(`menu.nm`, `menu.price`, `sub_total.subtotal_price`, `total.total_price`, …).
Better project, worse fit for seven days.

### 1b. Knowledge fields — what this touches

| Field | What you actually need from it |
|---|---|
| Transformer architecture | Attention, why √d scaling, what the KV projections are |
| Vision-language modelling | Patch embedding → connector → token sequence; why image tokens dominate your sequence length |
| Parameter-efficient fine-tuning | LoRA maths, rank/alpha, which modules to target |
| Quantization | NF4, double quant, what precision costs you and where |
| Supervised fine-tuning | Chat templates, label masking, loss on completion only |
| Evaluation methodology | Metric design, baseline strength, robustness testing |
| OCR / document AI | Just enough domain context to interpret your errors |

You do not need all seven deeply. You need one line of honest opinion about each.

---

## 2. What to learn, in three tiers

### Tier 0 — before you write code (half a day, non-negotiable)

**LoRA.** Frozen W, learned low-rank update: `W + (α/r)·BA`, where B is d×r and
A is r×k. Rank r controls capacity; alpha scales the update.
> *Check yourself:* on a 4096×4096 matrix at r=16, how many trainable
> parameters vs. the original? If you can't do this in your head, reread.

**QLoRA.** Base weights frozen in 4-bit NF4; adapters trained in higher
precision; gradients flow through the dequantized weights.
> *Check:* why does this save memory when you still dequantize during the
> forward pass?

**The VLM data path.** Image → patches → vision encoder → connector/merger →
tokens prepended into the LLM's sequence.
> *Check:* roughly how many tokens does one receipt image become? This number
> is why you'll hit OOM.

**Chat templates and label masking.** The single largest source of silent
failure. You must mask the prompt so loss is computed only on the completion.
> *Check:* what happens to your model if you don't mask? (It learns to generate
> your prompt. Loss looks fine. Outputs are garbage.)

### Tier 1 — learn while building (days 1–5)

Peak memory sources and how to cut each · gradient accumulation vs. batch size ·
gradient checkpointing's compute/memory trade · why greedy decoding for eval ·
Levenshtein distance and what ANLS forgives that exact match doesn't · fp16 vs.
bf16 numerics.

### Tier 2 — for the interview, not the build (day 7)

LLaVA's two-stage training · why LoRA is usually applied to attention
projections · when SFT is insufficient and preference alignment is needed ·
what full fine-tuning would buy you over LoRA here · how you'd extend this to
forgery detection.

---

## 3. Day-by-day

### Day 0 — Setup and reading (4–5 hrs)

- Read the LoRA and QLoRA papers. Skim; you want the method sections.
- Colab/Kaggle notebook: `pip install "transformers>=4.57" peft trl bitsandbytes accelerate qwen-vl-utils`
- Load `Qwen3-VL-2B-Instruct` in 4-bit. Run inference on **one** receipt.
- **Skip `flash_attention_2`.** Pre-built wheels don't cover every CUDA/torch
  combo and a source compile can eat 20+ minutes or just fail. You don't need it.
- Download SROIE. Look at 10 images and 10 label files with your own eyes.

**Done when:** the model describes a receipt image correctly. Nothing more.

---

### Day 1 — Baseline evaluation (6–7 hrs) ← *the day that removes the anxiety*

No training today.

- `src/data.py` — load images + ground-truth JSON, build the prompt.
- Prompt the base model zero-shot for the four-field JSON. **Make this baseline
  strong** — describe the schema clearly, give one format example. A weak
  baseline inflates your gain and it's the first thing a careful reader checks.
- `src/metrics.py` — exact match with explicit normalisation (lowercase,
  collapse whitespace, strip currency symbols) and ANLS.
- Parse failures count as total misses on every field. Never silently drop them;
  log the rate separately.
- Run over the test set. **Get a real number.**

**Done when:** you have clean-split zero-shot exact match, per field, plus a
JSON-parse-failure rate. Write them in the README immediately.

> Every `[X]` in your README is now filled. This is the psychological turning
> point of the week.

---

### Day 2 — The degraded split (5–6 hrs)

- `src/degrade.py` — JPEG q30, Gaussian blur σ≈1.5, ±15° rotation with edge
  padding, synthetic glare (elliptical overexposure), partial occlusion.
- **Fixed seed.** Same images, corrupted deterministically, so clean and
  degraded are directly comparable.
- Decide and record: corruptions applied independently, or composed? Composed is
  harder and more realistic. Say which — reviewers will ask.
- Re-run the Day 1 baseline on the degraded split.

**Done when:** all four `base × {clean, degraded}` cells in the results table are
real numbers. You now have a robustness gap for a model you haven't touched.

---

### Days 3–4 — Training (10–14 hrs across both)

Day 3 is the hard day. Budget for frustration.

- `src/train.py` with TRL's `SFTTrainer` (check the current TRL docs — this API
  moves).
- Format data into the chat template. Mask the prompt. **Print one fully
  tokenised example and read it manually before launching a run.** Ten minutes
  here saves six hours.
- LoRA config: r=16, alpha=32, dropout=0.05, target `q_proj, k_proj, v_proj,
  o_proj, gate_proj, up_proj, down_proj` on the decoder. ViT and merger frozen.
- Batch size 1 + gradient accumulation, gradient checkpointing on, cap image
  resolution, `paged_adamw_8bit`.
- **T4 users:** Turing has no bf16. Use fp16 — and if loss goes NaN, that's the
  fp16-plus-4-bit interaction, not your hyperparameters.
- First run: deliberately overfit **20 samples**. Loss should approach zero. If
  it doesn't, your data pipeline is broken, not your hyperparameters. Do not
  proceed until this passes.
- Then the real run. Then evaluate on both splits.

Day 4 — ablations, same data/seed/steps: rank 8, rank 16, rank 32.
Unfrozen vision encoder if time and memory permit.

**Done when:** at least two configurations trained and evaluated on both splits.

---

### Day 5 — Failure analysis (4–5 hrs) ← *the highest-value day*

- Pull 30–50 errors from the fine-tuned model on the degraded split.
- Categorise by hand: multi-line truncation, numeric transcription (5/S, 0/O),
  hallucinated fields, JSON malformation, and so on.
- Count each category. Which corruption drives which error type?
- Write 2–3 hypotheses, each with the experiment that would confirm or kill it.

**Done when:** you can say "X% of remaining errors are [category], probably
because [mechanism], which I'd test by [experiment]."

This is what separates your repo from a tutorial reproduction. Cut a training
run before you cut this.

---

### Day 6 — Write-up (4–5 hrs)

Fill the README. Push adapter to HF Hub. Include the loss curve. Write the
Limitations section honestly — single seed, small test set, hand-picked
corruption parameters. Update the resume with one real number: the
degraded-split comparison, not the clean one.

---

### Day 7 — Interview prep only. No code.

---

## 4. Go / no-go gates

| Gate | Condition | If failed |
|---|---|---|
| End Day 1 | Baseline eval produces real numbers | Stop building. Fix the eval — everything downstream depends on it |
| Mid Day 3 | 20-sample overfit reaches near-zero loss | Data pipeline bug. Do **not** tune hyperparameters to fix a plumbing problem |
| End Day 3 | A training run with decreasing loss | **Pivot:** drop vision, do text-only LoRA on a 1B Qwen/Llama for structured extraction. Keeps PEFT, SFT, HF, eval harness, ablations. Loses only the VLM bullet |
| End Day 5 | Failure analysis written | Ship anyway — an honest partial result beats nothing |

Decide the Day 3 pivot **now**, while calm. Not at 2am on day 5.

---

## 5. Risk register

| Risk | Symptom | Fix |
|---|---|---|
| CUDA OOM | Crash on first training step | Batch 1 + grad accum; grad checkpointing; cap image resolution; drop to SmolVLM-500M |
| Architecture unrecognised | `model type qwen3_vl but Transformers does not recognize` | transformers < 4.57 — upgrade, or install from git |
| flash-attn install hangs/fails | 20-min compile or hard failure | Remove `attn_implementation="flash_attention_2"` entirely |
| NaN loss | Loss → NaN early | fp16 + 4-bit on T4. Lower LR, add warmup, or move to a bf16-capable GPU |
| Chat template wrong | Trains fine, outputs garbage | Print a tokenised example and read it |
| No masking | Loss looks great, model echoes prompts | Mask prompt tokens; loss on completion only |
| JSON parse failures | Metrics implausibly low | Log parse-failure rate separately; it's a result, not a bug to hide |
| Colab disconnect | Hours lost | Checkpoint to Drive every N steps |

---

## 6. Deliverables checklist

- [ ] Public repo, README complete, zero unfilled placeholders
- [ ] Adapter on HuggingFace Hub
- [ ] Results: base vs. fine-tuned × clean vs. degraded
- [ ] Per-field breakdown
- [ ] Ablation table (2+ configurations, honestly labelled)
- [ ] Failure analysis with categories and hypotheses
- [ ] Limitations section
- [ ] Loss curve
- [ ] Reproduction commands that actually run from a clean clone
- [ ] Resume updated with one real number

---

## 7. Questions this sets you up for — and exposes you to

**You'll be able to answer:**
- Why rank 16? → your ablation
- Why freeze the vision encoder? → memory, stability, and you tested it
- How do you know it improved? → your harness, defined before you saw results
- What's still broken? → your failure analysis
- Why does the degraded gap matter? → their entire business

**You'll be exposed on** (prepare, don't bluff):
- "Walk me through the backward pass with quantized frozen weights"
- "Why those target modules and not the MLP only?"
- "How would you pick corruption parameters principled-ly?" → honest answer:
  you didn't; you'd calibrate against a measured distribution of real captures
- "What would full fine-tuning buy you?"

"I haven't worked with that — here's how I'd find out" is a fine answer from an
intern candidate. Bluffing is not, and experienced people spot it instantly.

---

## 8. Reading list, ordered by value per minute

1. **LoRA** — Hu et al., arXiv:2106.09685 — method section only
2. **QLoRA** — Dettmers et al., arXiv:2305.14314 — NF4 and paged optimizers
3. **Qwen3-VL technical report** — arXiv:2511.21631 — architecture, skim
4. HuggingFace PEFT docs — `LoraConfig` parameters, read properly
5. TRL `SFTTrainer` docs — **current version**, the API moves
6. **LLaVA** — arXiv:2304.08485 — day 7, for the two-stage training question

---

## 9. If you have four days, not seven

Keep: Day 1 (baseline), Day 2 (degraded), Day 3 (one training run), Day 5
(failure analysis).
Cut: ablations, Day 4 entirely, half the reading.

One configuration, two splits, honest failure analysis is a complete project.
Three configurations with no failure analysis is not.
