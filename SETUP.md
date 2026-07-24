# Setup

## 1. Merge into your repo

You already have `github.com/richujames/vlm-doc-extraction`. I could not read
its contents (GitHub API rate limit), so **check for filename collisions before
copying** — particularly `README.md` and `.gitignore`.

```bash
git clone https://github.com/richujames/vlm-doc-extraction.git
cd vlm-doc-extraction

# copy the scaffold in, then review before committing
git status
git diff

git add -A
git commit -m "Add project scaffold: eval harness, degradation pipeline, configs"
git push
```

If your repo already has a README you want to keep, rename this one to
`README_TEMPLATE.md` and merge by hand.

---

## 2. Local environment (light work only)

Your machine is fine for reading data, running `metrics.py`, and building the
degraded split — none of that needs a GPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pillow numpy pyyaml

python src/metrics.py              # should print "all sanity checks passed"
```

Training and evaluation need a GPU. Use Colab or Kaggle.

---

## 3. Get SROIE

SROIE = ICDAR 2019 Robust Reading Challenge, Task 3 (key information
extraction). Available on Kaggle and several HuggingFace dataset mirrors.

Arrange it as:

```
data/
  train/img/X0001.jpg        train/entities/X0001.txt
  test/img/X9001.jpg         test/entities/X9001.txt
```

Each `entities/*.txt` is a JSON object with keys `company`, `date`, `address`,
`total`.

Verify before anything else:

```bash
python src/data.py --root data --split test --n 3
```

This prints three samples with their gold fields and the exact training target
string. **Read the output.** If the labels look wrong here, every number
downstream is wrong.

---

## 4. Colab / Kaggle quickstart

```python
!pip install -q "transformers>=4.57" peft trl bitsandbytes accelerate qwen-vl-utils

!git clone https://github.com/richujames/vlm-doc-extraction.git
%cd vlm-doc-extraction

# upload / mount your data/ directory here

import sys; sys.path.insert(0, "src")
```

Check what you were allocated — it determines your dtype:

```python
import torch
print(torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
```

**T4** → `bf16: false, fp16: true` in your config (Turing has no bf16).
**L4 / A100** → set `bf16: true, fp16: false`. Fewer numerical problems.

Kaggle gives roughly 30 GPU-hours/week, more generous than Colab free.

---

## 5. Run order

```bash
# Day 1 — baseline. No training. This is the day that produces your first
# real numbers, and the day the placeholders in the README start disappearing.
python src/eval.py --split test --tag base_clean --limit 20   # smoke test first
python src/eval.py --split test --tag base_clean

# Day 2 — degraded split
python src/degrade.py --input data/test/img --output data/test_degraded/img \
                      --limit 5 --save-comparison
#   ^ LOOK AT data/test_degraded/../comparison/ WITH YOUR OWN EYES before
#     generating the full split. If the corruptions look absurd or invisible,
#     fix them now, not after you have results built on them.

python src/degrade.py --input data/test/img --output data/test_degraded/img
python src/eval.py --split test_degraded --tag base_degraded

# Day 3 — training. Gate first.
python src/train.py --config configs/rank16_frozen.yaml --overfit-test
#   Loss must approach ~0 on 20 samples. If it plateaus, your data pipeline is
#   broken. Do not proceed. Do not tune hyperparameters to fix a plumbing bug.

python src/train.py --config configs/rank16_frozen.yaml

python src/eval.py --split test --adapter outputs/rank16_frozen --tag ft_clean
python src/eval.py --split test_degraded --adapter outputs/rank16_frozen --tag ft_degraded

# Day 4 — ablations
python src/train.py --config configs/rank8_frozen.yaml
python src/train.py --config configs/rank32_frozen.yaml
python src/train.py --config configs/rank16_unfrozen.yaml
```

---

## 6. Day 5 — failure analysis

Everything you need is already on disk. `results/*_records.jsonl` has one line
per sample with predictions, gold values, normalised forms, and per-field
scores. You do not need to re-run inference.

```python
import json, collections
recs = [json.loads(l) for l in open("results/ft_degraded_records.jsonl")]

# where are the errors concentrated?
by_field = collections.Counter()
for r in recs:
    for f, v in r["fields"].items():
        if not v["em"]:
            by_field[f] += 1
print(by_field)

# pull 30 actual failures and read them
fails = [r for r in recs if any(not v["em"] for v in r["fields"].values())][:30]
```

Then categorise **by hand**. Multi-line truncation, numeric transcription
(5/S, 0/O), hallucinated fields, JSON malformation. Count each. Cross-reference
against `data/degradation_manifest.json` to see which corruption drove which
error type.

This is the highest-value work in the project. If you are behind schedule, cut
a training run before you cut this.

---

## 7. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `model type qwen3_vl but Transformers does not recognize this architecture` | transformers < 4.57 | `pip install -U transformers`, or from git |
| flash-attn install hangs 20+ min | no wheel for your CUDA/torch | Don't install it. Not needed. |
| CUDA OOM on first step | image tokens | Lower `max_pixels` first, then batch size, then switch to SmolVLM-500M |
| Loss → NaN in first ~50 steps | fp16 + 4-bit on Turing | Lower LR, raise warmup, or move to a bf16 GPU |
| Loss looks great, outputs are garbage | prompt not masked | Print tokenised labels; prompt positions must be −100 |
| Exact match implausibly low | JSON parse failures | Check `parse_failure_rate` in the summary — it is reported separately for exactly this reason |
| Colab disconnects mid-run | idle timeout | `save_steps: 100`; checkpoint to Drive |
