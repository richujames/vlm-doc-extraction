"""
Evaluation harness. This is your Day 1 deliverable — no training involved.

    # zero-shot baseline, clean split
    python src/eval.py --split test --tag base_clean

    # zero-shot baseline, degraded split
    python src/eval.py --split test_degraded --tag base_degraded

    # after training
    python src/eval.py --split test --adapter outputs/rank16_frozen --tag ft_clean
    python src/eval.py --split test_degraded --adapter outputs/rank16_frozen --tag ft_degraded

Writes results/{tag}.json (summary) and results/{tag}_records.jsonl (per-sample).
Keep the records file. On Day 5 you will want to grep it, not regenerate it.

DECODING IS FIXED AND IDENTICAL FOR BASE AND FINE-TUNED. Greedy, same
max_new_tokens, same prompt. Changing decoding between the two arms would
confound the comparison and invalidate every number you report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data import FIELDS, build_prompt, load_split
from metrics import Scores, score_sample

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def load_model(adapter: Path | None, four_bit: bool = True):
    """
    Load Qwen3-VL, optionally with a trained LoRA adapter on top.

    Requires transformers >= 4.57. Older versions raise
    'model type qwen3_vl but Transformers does not recognize this architecture'.

    NOTE: no flash_attention_2. Pre-built wheels do not cover every CUDA/torch
    combination and pip falls back to a 20+ minute source compile that often
    fails outright. You do not need it for a 2B model.
    """
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant_config = None
    if four_bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # T4 is Turing and has NO bf16 support -> float16.
            # On L4/A100 switch this to torch.bfloat16 for better numerical stability.
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype="auto",
        device_map="auto",
    )

    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"loaded adapter: {adapter}")

    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


@torch.no_grad()
def generate(model, processor, image, prompt: str, max_new_tokens: int = 128) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,   # greedy. deterministic. do not change between arms.
    )
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--split", required=True, help="test | test_degraded")
    ap.add_argument("--adapter", type=Path, default=None, help="omit for zero-shot baseline")
    ap.add_argument("--tag", required=True, help="name for the results files")
    ap.add_argument("--limit", type=int, default=None, help="subset, for a quick smoke run")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    samples = load_split(args.root, args.split)
    if args.limit:
        samples = samples[: args.limit]
    print(f"split={args.split}  n={len(samples)}  adapter={args.adapter or 'NONE (zero-shot)'}")

    model, processor = load_model(args.adapter, four_bit=not args.no_4bit)
    prompt = build_prompt()

    scores = Scores()
    records = []
    started = time.time()

    for i, sample in enumerate(samples, start=1):
        raw = generate(model, processor, sample.image(), prompt, args.max_new_tokens)
        rec = score_sample(scores, raw, sample.gold)
        rec["doc_id"] = sample.doc_id
        records.append(rec)

        if i % 25 == 0 or i == len(samples):
            elapsed = time.time() - started
            print(f"  {i}/{len(samples)}  ({elapsed/i:.2f}s/sample)")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    summary = scores.summary()
    summary["meta"] = {
        "tag": args.tag,
        "split": args.split,
        "adapter": str(args.adapter) if args.adapter else None,
        "model_id": MODEL_ID,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "n_samples": len(samples),
        "seconds_total": round(time.time() - started, 1),
    }
    (results_dir / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))
    with (results_dir / f"{args.tag}_records.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + scores.pretty())
    print(f"\nwrote results/{args.tag}.json and results/{args.tag}_records.jsonl")
    print("\n>>> Put these numbers in the README now, while they are in front of you.")


if __name__ == "__main__":
    main()
