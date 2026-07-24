# Document Field Extraction with a QLoRA Fine-Tuned Vision-Language Model

<!--
  ============================================================================
  TEMPLATE NOTES — DELETE THIS BLOCK BEFORE PUBLISHING
  ============================================================================
  Every `[FILL: ...]` is a gap you must close with a real number, name, or
  observation. Do not publish with any left in place.

  If an experiment didn't happen, DELETE the section rather than inventing a
  result. An empty README with three honest numbers beats a full one with
  three invented ones — and an interviewer will find the invented ones by
  asking "how did you measure that?"

  Fill order that saves the most time:
    1. Method + Setup (you know these before training finishes)
    2. Evaluation protocol (write this BEFORE you look at results — it stops
       you from quietly redefining the metric to flatter the outcome)
    3. Results tables
    4. Failure analysis + Limitations  <-- highest value per word. Do not skip.
  ============================================================================
-->

Fine-tuning `[FILL: Qwen3-VL-2B-Instruct]` with 4-bit QLoRA for structured field
extraction from `[FILL: receipt]` images, evaluated on both clean inputs and a
deliberately degraded split that simulates real-world capture conditions.

**Headline result:** field-level exact match improved from `[FILL: X]`% to
`[FILL: Y]`% on the clean split and from `[FILL: X]`% to `[FILL: Y]`% on the
degraded split, training `[FILL: N]`M parameters (`[FILL: P]`% of the base model)
in `[FILL: H]` GPU-hours on a single `[FILL: T4 / L4 / A100]`.

---

## Why this exists

Document extraction benchmarks are mostly clean scans. Deployed systems see
phone photos: motion blur, glare, 15-degree skew, aggressive JPEG compression
from a messaging app, a thumb over one corner.

A model that scores well on the clean split and collapses on the degraded one
is not a working system, and reporting only the first number hides that. So
this repo reports both, and treats the gap between them as the result that
actually matters.

---

## Results

### Base vs. fine-tuned

| Split | Model | Field-level exact match | ANLS | Notes |
|---|---|---|---|---|
| Clean | Base (zero-shot) | `[FILL]`% | `[FILL]` | |
| Clean | Fine-tuned | `[FILL]`% | `[FILL]` | |
| Degraded | Base (zero-shot) | `[FILL]`% | `[FILL]` | |
| Degraded | Fine-tuned | `[FILL]`% | `[FILL]` | |

**Robustness gap** (clean minus degraded, fine-tuned): `[FILL]` pt.
`[FILL: One sentence — did fine-tuning close the gap, widen it, or leave it flat?
This is the interesting question. A model that gains 20 pts on clean and 3 pts
on degraded has learned something narrower than it appears.]`

### Per-field breakdown

| Field | Base | Fine-tuned | Δ |
|---|---|---|---|
| `[FILL: total]` | `[FILL]`% | `[FILL]`% | `[FILL]` |
| `[FILL: date]` | `[FILL]`% | `[FILL]`% | `[FILL]` |
| `[FILL: vendor name]` | `[FILL]`% | `[FILL]`% | `[FILL]` |
| `[FILL: address]` | `[FILL]`% | `[FILL]`% | `[FILL]` |

### Ablations

All runs use identical data, seed, and step count unless noted.

| Run | LoRA rank | α | Vision encoder | Trainable params | Clean EM | Degraded EM |
|---|---|---|---|---|---|---|
| A | 8 | `[FILL]` | frozen | `[FILL]`M | `[FILL]`% | `[FILL]`% |
| B | 16 | `[FILL]` | frozen | `[FILL]`M | `[FILL]`% | `[FILL]`% |
| C | 32 | `[FILL]` | frozen | `[FILL]`M | `[FILL]`% | `[FILL]`% |
| D | 16 | `[FILL]` | unfrozen | `[FILL]`M | `[FILL]`% | `[FILL]`% |

**Read:** `[FILL: What did rank actually buy you? Where did it stop paying?
Did unfreezing the vision encoder help, or just cost memory? Say what you
concluded and how confident you are.]`

> Single seed per configuration. Differences under `[FILL: ~2]` pts should not
> be treated as real without repeated runs — noted here rather than quietly
> claimed as findings.

---

## Method

**Base model:** `[FILL: Qwen/Qwen3-VL-2B-Instruct]`

**Quantization:** 4-bit NF4 via bitsandbytes, double quantization `[FILL: on/off]`,
compute dtype `[FILL: fp16 — T4 is Turing and has no bf16 support]`.

**LoRA configuration:**

| Parameter | Value |
|---|---|
| Rank (r) | `[FILL: 16]` |
| Alpha | `[FILL: 32]` |
| Dropout | `[FILL: 0.05]` |
| Target modules | `[FILL: q_proj, k_proj, v_proj, o_proj]` |
| Vision encoder | `[FILL: frozen]` |

`[FILL: Why these target modules and not others? One sentence. "Attention
projections only, because the MLP adapters roughly doubled trainable params
for no measurable gain in run X" is a real answer. So is "this is the common
default and I didn't have budget to test alternatives" — that's honest and
still fine.]`

**Training:**

| Parameter | Value |
|---|---|
| Epochs / steps | `[FILL]` |
| Effective batch size | `[FILL]` (batch `[FILL]` × grad accum `[FILL]`) |
| Learning rate | `[FILL]` |
| Scheduler | `[FILL: cosine, warmup ratio 0.03]` |
| Optimizer | `[FILL: paged_adamw_8bit]` |
| Max image resolution | `[FILL]` |
| Gradient checkpointing | `[FILL: on]` |
| Seed | `[FILL: 42]` |

**Loss curve:** `[FILL: link to W&B run or drop results/loss_curve.png here]`

---

## Data

**Dataset:** `[FILL: CORD / SROIE]` — `[FILL: N]` train / `[FILL: N]` val /
`[FILL: N]` test.

**Target format:** JSON with keys `[FILL: list them]`. Prediction is parsed as
JSON; unparseable output counts as a total miss on every field rather than
being silently dropped.

### The degraded split

The degraded test set is the *same* test images with corruptions applied at a
fixed seed, so clean and degraded results are directly comparable — no
distribution shift beyond the corruption itself.

| Corruption | Parameter |
|---|---|
| JPEG recompression | quality `[FILL: 30]` |
| Gaussian blur | σ = `[FILL: 1.5]` |
| Rotation | ±`[FILL: 15]`°, bilinear, edge-padded |
| Synthetic glare | `[FILL: elliptical overexposure, N% of frame area]` |
| Partial occlusion | `[FILL: N% of frame, random position]` |

Implemented in `src/degrade.py`. Run with `[FILL: --severity]` to reproduce.

`[FILL: Were corruptions applied independently or composed? Say which — it
changes how hard the split is, and reviewers will ask.]`

---

## Evaluation protocol

Written before results were inspected.

**Field-level exact match** — normalized string equality per field, averaged
over fields then over documents. Normalization: `[FILL: lowercase, collapse
whitespace, strip currency symbols]`. Missing field predicted as missing counts
as correct; missing field hallucinated counts as wrong.

**ANLS** (Average Normalized Levenshtein Similarity) — per field,
`1 − NL(pred, gt)` where `NL` is Levenshtein distance normalized by the longer
string, thresholded to 0 below `[FILL: 0.5]`. Chosen because exact match
punishes a single-character OCR slip as harshly as a completely wrong answer,
which overstates failure on long fields like addresses.

**Decoding:** `[FILL: greedy, max_new_tokens=N]`. Identical for base and
fine-tuned — no per-model prompt tuning, since that would confound the
comparison.

**Baseline prompt:** the zero-shot base model gets `[FILL: a prompt describing
the target JSON schema]`. `[FILL: Note whether you tried to make this baseline
strong. A weak baseline inflates your gain and is the first thing a careful
reader checks.]`

---

## Failure analysis

Manual inspection of `[FILL: N]` errors from the fine-tuned model on the
degraded split.

| # | Input | Predicted | Ground truth | Category |
|---|---|---|---|---|
| 1 | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| 2 | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| 3 | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

**Categories observed:**

| Category | Count | Share |
|---|---|---|
| `[FILL: multi-line field truncation]` | `[FILL]` | `[FILL]`% |
| `[FILL: numeric transcription — 5/S, 0/O]` | `[FILL]` | `[FILL]`% |
| `[FILL: hallucinated field on absent data]` | `[FILL]` | `[FILL]`% |
| `[FILL: JSON malformation]` | `[FILL]` | `[FILL]`% |

**Hypotheses and how I'd test them:**

1. `[FILL: Hypothesis.]` → `[FILL: The experiment that would confirm or kill it.]`
2. `[FILL: Hypothesis.]` → `[FILL: The experiment that would confirm or kill it.]`

> This section is the point of the repo. Anyone can post an accuracy number;
> the reason to read further is knowing *which* `[FILL: N]`% is still broken
> and having a theory about why.

---

## Limitations

- Single seed per configuration; small differences are not significant.
- Test set is `[FILL: N]` documents — confidence intervals are wide.
- `[FILL: Corruption parameters were chosen by eye, not calibrated against a
  measured distribution of real phone captures.]`
- `[FILL: Single dataset and document type — no evidence this transfers to
  other layouts or languages.]`
- `[FILL: Anything else you know is weak. Listing it yourself is strictly
  better than having it found.]`

---

## Reproducing

```bash
git clone https://github.com/[FILL: richujames/repo-name]
cd [FILL: repo-name]
pip install -r requirements.txt

# 1. Build clean + degraded test splits (fixed seed)
python src/degrade.py --input data/test --output data/test_degraded --seed 42

# 2. Zero-shot baseline
python src/eval.py --model [FILL: base-model-id] --split clean
python src/eval.py --model [FILL: base-model-id] --split degraded

# 3. Train
python src/train.py --config configs/[FILL: rank16_frozen].yaml

# 4. Evaluate the adapter
python src/eval.py --adapter outputs/[FILL: run-name] --split clean
python src/eval.py --adapter outputs/[FILL: run-name] --split degraded
```

**Environment:** `[FILL: Python 3.x, torch x.x, transformers x.x, peft x.x,
trl x.x, bitsandbytes x.x]`. Pinned in `requirements.txt`.

**Hardware:** `[FILL: single T4 (16GB), ~H hours per run]`.

**Adapter weights:** `[FILL: HuggingFace Hub link]`

---

## Repository layout

```
├── configs/           # one YAML per training run
├── src/
│   ├── data.py        # dataset loading, prompt/target formatting
│   ├── degrade.py     # corruption pipeline for the degraded split
│   ├── train.py       # QLoRA fine-tuning
│   ├── eval.py        # inference + scoring
│   └── metrics.py     # exact match, ANLS, normalization
├── results/           # per-run metrics, loss curves, failure samples
└── notebooks/
    └── failure_analysis.ipynb
```

---

## References

- `[FILL: Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (2021)]`
- `[FILL: Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs (2023)]`
- `[FILL: Qwen2-VL technical report]`
- `[FILL: CORD / SROIE dataset paper]`

## License

`[FILL: MIT]` — note that the base model and dataset carry their own licenses.
