# Project Profile

Context for anyone picking this up — including you on day 4 when you have
forgotten why you chose something, and including an AI assistant in a fresh
session. Keep the Status section updated; it is the only part that goes stale.

---

## One-line

QLoRA fine-tune of Qwen3-VL-2B-Instruct for structured field extraction from
receipt images, evaluated on clean *and* deliberately degraded inputs.

## Why it exists

Portfolio artifact for a Deep Learning / ML Research Intern application
(LLMs & VLMs, identity verification / document intelligence / fraud detection).

The application-critical property: it must demonstrate **training and
evaluating** a model, not calling an API. The role description explicitly
screens out "API-first AI development" and "tutorial-based implementations".
Every design decision below serves that.

## Owner

Richu James — B.Tech CSE (Cyber Security), Amrita Vishwa Vidyapeetham.
Background: IoT/MQTT security, unsupervised anomaly detection, adversarial
evaluation of LLM guardrails (7 disclosed bypasses across frontier Gemini and
OpenAI models). Strong Python/C++. **First deep learning training project** —
no prior PyTorch training loop experience.

---

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` | Current small Qwen VLM; strong OCR; fits 4-bit on free-tier GPU |
| Model class | `Qwen3VLForConditionalGeneration` | **Requires transformers ≥ 4.57** |
| Dataset | SROIE (ICDAR 2019 Task 3) | 4 flat fields. CORD is richer but nested — parsing/scoring would cost a day |
| Fields | `company`, `date`, `address`, `total` | One easy, one hard (address), one numeric — the spread makes per-field results informative |
| Method | 4-bit QLoRA, NF4, double quant | Single-GPU feasibility |
| LoRA target | LLM decoder attention + MLP | ViT and merger frozen by default |
| Primary config | r=16, α=32, dropout 0.05 | α/r held at 2.0 across rank ablations |
| Metrics | Field-level exact match + ANLS | Reported on both splits |
| Decoding | Greedy, `max_new_tokens=128` | Identical for base and fine-tuned — non-negotiable |
| Tracking | Weights & Biases | Loss curve goes in the README |

**Non-decisions (deliberately open):** the baseline prompt wording (tune it on
~20 train images on Day 1, then freeze), and whether corruptions compose or
apply independently (record whichever you choose).

---

## Experimental design

**Claim under test:** does QLoRA fine-tuning on clean receipts improve
extraction accuracy, and does that improvement survive realistic input
degradation?

**Arms:** {base, fine-tuned} × {clean, degraded} = 4 evaluation runs.

**Controls:** same test images in both splits (corruption is deterministic,
seeded per filename); identical prompt and decoding across arms; same
data/seed/step count across ablation configs.

**Ablations:** LoRA rank 8 / 16 / 32 (α/r fixed); vision encoder frozen vs.
unfrozen at r=16.

**Known limitations — state these, do not wait to be asked:**
- Single seed per configuration; sub-~2pt differences are not meaningful
- Small test set; wide confidence intervals
- Corruption parameters chosen by eye, **not** calibrated against a measured
  distribution of real phone captures
- Single dataset, single document type, single language

**Expected complication:** Qwen3-VL's OCR is explicitly trained for robustness
to low light, blur, and tilt. The degraded-split gap may be *small*. That is a
finding, not a failure — "the base model's robustness training already absorbs
most of these corruptions; the residual gap concentrates in X" is a more
interesting result than a large drop. **Do not increase corruption severity
until you get the gap you were hoping for.** That is fitting the experiment to
the conclusion.

---

## Repository map

```
configs/          one YAML per training run
src/
  metrics.py      COMPLETE — exact match, ANLS, JSON parsing, aggregation
  degrade.py      COMPLETE — deterministic corruption pipeline
  data.py         COMPLETE except build_prompt() — yours to tune
  eval.py         COMPLETE — Day 1 deliverable, no training needed
  train.py        SCAFFOLD — 3 TODOs, deliberately left for you
results/          per-run summaries + per-sample records (jsonl)
notebooks/        failure analysis
```

**Why train.py is incomplete:** the collator, label masking, and TRL wiring are
exactly what an interviewer will probe. Code you did not write is code you
cannot defend, and the role screens out reproduced tutorials.

---

## Environment

```
transformers >= 4.57      # older: "model type qwen3_vl not recognized"
torch, peft, trl, bitsandbytes, accelerate, qwen-vl-utils
pillow, numpy, pyyaml
```

**No flash-attention.** Pre-built wheels don't cover every CUDA/torch combo;
pip falls back to a 20+ min source compile that often fails. Not needed at 2B.

**T4 (Colab/Kaggle free) is Turing — no bf16.** Use fp16. NaN loss early is
the fp16 + 4-bit interaction, not your learning rate.

---

## Gates

| Gate | Condition | If failed |
|---|---|---|
| End Day 1 | Baseline eval produces real numbers | Stop; fix eval before anything else |
| Mid Day 3 | 20-sample overfit → near-zero loss | Data pipeline bug. Do **not** tune hyperparameters |
| End Day 3 | Real run with decreasing loss | **Pivot** to text-only LoRA on a 1B model. Keeps PEFT/SFT/eval/ablations; loses the VLM bullet |
| End Day 5 | Failure analysis written | Ship anyway — honest partial beats nothing |

---

## Status

_Update this section as you go. It is the handoff._

- [ ] Day 0 — env, model loads, one inference works
- [ ] Day 1 — baseline eval, clean split → **first real numbers**
- [ ] Day 2 — degraded split built and baselined
- [ ] Day 3 — overfit gate passed; first real training run
- [ ] Day 4 — ablations
- [ ] Day 5 — failure analysis
- [ ] Day 6 — README complete, adapter pushed
- [ ] Day 7 — interview prep

**Current blocker:** _(none yet)_

**Numbers so far:** _(fill in as they land — do not leave these to the end)_

| Run | Split | Exact match | ANLS | Parse failures |
|---|---|---|---|---|
| base | clean | | | |
| base | degraded | | | |
| ft r16 | clean | | | |
| ft r16 | degraded | | | |
