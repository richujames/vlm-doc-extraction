"""
SROIE loading and prompt construction.

Expected layout after you download SROIE (ICDAR 2019 Task 3):

    data/
      train/img/X0001.jpg          data/train/entities/X0001.txt
      test/img/X9001.jpg           data/test/entities/X9001.txt
      test_degraded/img/X9001.jpg  (produced by src/degrade.py)

Each entities file is a JSON object with keys: company, date, address, total.

WHAT IS DONE FOR YOU: file pairing, JSON loading, the split abstraction.
WHAT IS YOURS: the prompt. See build_prompt() — that is a real design decision
and the single cheapest lever on your baseline number. Do not accept the
placeholder without thinking about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

FIELDS = ["company", "date", "address", "total"]


@dataclass
class Sample:
    doc_id: str
    image_path: Path
    gold: dict[str, str]

    def image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


def load_split(root: Path, split: str) -> list[Sample]:
    """
    Load a split. `split` is a directory name under `root`: 'train', 'test',
    'test_degraded'.

    For degraded splits, labels are read from the CLEAN split of the same name
    (test_degraded -> test), because corruption changes pixels, not ground truth.
    """
    split_dir = root / split
    img_dir = split_dir / "img"
    label_split = split.replace("_degraded", "")
    label_dir = root / label_split / "entities"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"missing image dir: {img_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"missing label dir: {label_dir}")

    samples: list[Sample] = []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue  # unlabelled image; skip silently, SROIE has a few
        try:
            raw = json.loads(label_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            print(f"  warn: unparseable label {label_path.name}, skipping")
            continue
        samples.append(
            Sample(
                doc_id=img_path.stem,
                image_path=img_path,
                gold={f: str(raw.get(f, "")).strip() for f in FIELDS},
            )
        )

    if not samples:
        raise RuntimeError(f"no labelled samples found for split '{split}'")
    return samples


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SCHEMA_HINT = """{"company": "", "date": "", "address": "", "total": ""}"""


def build_prompt() -> str:
    """
    YOUR DESIGN DECISION. Read this before you run anything.

    A weak baseline prompt inflates your fine-tuning gain and is the first thing
    a careful reader checks. If your zero-shot model scores near zero because
    you asked vaguely and it wrote a paragraph, your headline improvement is
    measuring your prompt, not your training.

    A strong baseline should at minimum:
      - state the exact JSON keys required
      - demand JSON only, no prose, no markdown fences
      - say what to do when a field is absent (empty string, not omitted)
      - specify format expectations where they matter (e.g. total as a bare
        number, date verbatim as printed)

    Try 3-4 variants on ~20 training images on Day 1, keep the best, and RECORD
    in your README that you did this and what you picked. "I tuned the baseline
    prompt before measuring" is a sentence that earns trust.

    Do NOT then tune the prompt again for the fine-tuned model — identical
    prompts on both sides, or the comparison is confounded.
    """
    return (
        "Extract the following fields from this receipt image and return ONLY a "
        "JSON object, with no explanation and no markdown code fences.\n\n"
        f"Required keys: {SCHEMA_HINT}\n\n"
        "Rules:\n"
        "- company: the merchant or vendor name as printed.\n"
        "- date: the transaction date exactly as printed on the receipt.\n"
        "- address: the full merchant address, joined into one line with single spaces.\n"
        "- total: the final total amount as a bare number, without a currency symbol.\n"
        "- If a field is not present, use an empty string. Never omit a key.\n"
    )


def build_target(gold: dict[str, str]) -> str:
    """Training target: compact JSON, key order fixed. Consistency matters more than prettiness."""
    return json.dumps({f: gold.get(f, "") for f in FIELDS}, ensure_ascii=False)


def to_chat_messages(sample: Sample, *, include_answer: bool) -> list[dict]:
    """
    Build the message list for Qwen3-VL's chat template.

    include_answer=True  -> training (prompt + target)
    include_answer=False -> inference (prompt only)

    ---------------------------------------------------------------------------
    CRITICAL — LABEL MASKING (this is where most people silently fail)

    When training, loss must be computed ONLY on the assistant turn. If you
    leave the prompt unmasked, the model learns to reproduce your instructions.
    Loss will look completely healthy. Outputs will be garbage.

    Before you launch a real run:
      1. Build one training example
      2. Print the tokenised input_ids and the labels side by side
      3. Confirm every prompt position has label == -100
      4. Confirm the answer positions have real token ids

    Ten minutes here saves six hours of debugging a loss curve that looks fine.
    ---------------------------------------------------------------------------
    """
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(sample.image_path)},
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": build_target(sample.gold)}]}
        )
    return messages


def iter_batches(samples: list[Sample], batch_size: int) -> Iterator[list[Sample]]:
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a split. Run this before anything else.")
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    samples = load_split(args.root, args.split)
    print(f"split '{args.split}': {len(samples)} labelled samples\n")
    for s in samples[: args.n]:
        img = s.image()
        print(f"{s.doc_id}  {img.size[0]}x{img.size[1]}px")
        for f in FIELDS:
            print(f"    {f:<9} {s.gold[f]!r}")
        print(f"    target -> {build_target(s.gold)}\n")

    print("=" * 70)
    print("PROMPT:\n")
    print(build_prompt())
