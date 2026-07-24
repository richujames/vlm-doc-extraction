"""
QLoRA fine-tuning for Qwen3-VL-2B-Instruct.

    python src/train.py --config configs/rank16_frozen.yaml

---------------------------------------------------------------------------
THIS FILE IS DELIBERATELY INCOMPLETE.

Everything structural is done: config loading, quantisation, LoRA setup,
trainer wiring, the overfit gate. Three things are marked TODO, and they are
the three things you will actually be asked about in an interview:

  TODO-1  the collator: turning samples into batched tensors
  TODO-2  label masking: loss on the completion only
  TODO-3  reading the TRL SFTTrainer API for your installed version

If I write those for you, you cannot explain them, and the JD explicitly
screens out portfolios of reproduced tutorials. The comments below tell you
exactly what each must do and how to verify it.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from data import load_split, to_chat_messages

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    print(json.dumps(cfg, indent=2))
    return cfg


def build_model(cfg: dict):
    """Load in 4-bit and attach LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    compute_dtype = torch.bfloat16 if cfg["training"].get("bf16") else torch.float16

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=compute_dtype, device_map="auto"
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )

    # Freeze the vision tower and the vision-language merger.
    #
    # WHY (be ready to say this out loud): the ViT is already strong, and
    # fine-tuning it on a few hundred receipts tends to destabilise visual
    # features rather than adapt them. It also costs memory you do not have.
    # The rank16_unfrozen config exists so you can TEST this claim rather than
    # just repeat it — that test is worth more than the assertion.
    if cfg["lora"]["freeze_vision"]:
        frozen = 0
        for name, param in model.named_parameters():
            if any(k in name for k in ("visual", "vision_tower", "merger")):
                param.requires_grad = False
                frozen += 1
        print(f"froze {frozen} vision-side parameter tensors")

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # <- this number goes straight in your README

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


# ---------------------------------------------------------------------------
# TODO-1 and TODO-2 live here
# ---------------------------------------------------------------------------

def make_collator(processor, cfg: dict):
    """
    Return a function: list[Sample] -> dict of batched tensors.

    TODO-1 — BUILD THE BATCH
      For each sample:
        msgs = to_chat_messages(sample, include_answer=True)
        text = processor.apply_chat_template(msgs, tokenize=False)
      Then pass texts + images to `processor(...)` with padding=True,
      return_tensors="pt".

      Cap image resolution here. Image tokens dominate your sequence length and
      are the main reason you will hit OOM — a large receipt can become many
      hundreds of tokens. Qwen processors accept min_pixels/max_pixels; start
      restrictive and relax only if memory allows.

    TODO-2 — MASK THE LABELS  (the highest-risk line in this project)
      labels = input_ids.clone()
      labels[padding positions] = -100
      labels[image token positions] = -100
      labels[prompt positions] = -100      # <-- the one people forget

      Loss must be computed ONLY on the assistant turn. If you skip the prompt
      mask, the model learns to reproduce your instructions. The loss curve
      will look completely healthy. The outputs will be garbage.

      Find the assistant turn by locating the generation-prompt boundary in the
      tokenised sequence — tokenise the prompt alone, and mask that many
      leading positions.

      VERIFY BEFORE TRAINING (do not skip):
        batch = collate([samples[0]])
        ids, labs = batch["input_ids"][0], batch["labels"][0]
        for i in range(len(ids)):
            tok = processor.tokenizer.decode([ids[i]])
            print(f"{i:4d} {tok!r:>20}  label={labs[i].item()}")
      Every prompt position must show -100. Every answer position must show a
      real token id. If that is not what you see, stop and fix it.
    """
    raise NotImplementedError(
        "TODO-1 / TODO-2: implement the collator and label masking. "
        "See the docstring — and print one tokenised example before training."
    )


def overfit_gate(model, processor, cfg, samples) -> None:
    """
    Deliberately overfit ~20 samples before the real run.

    Loss should fall close to zero within a few hundred steps. A 2B model can
    trivially memorise 20 examples; if it cannot, your DATA PIPELINE is broken,
    not your hyperparameters.

    This is the single highest-value 10 minutes in the whole week. It catches
    chat-template errors, missing label masks, and image-token misalignment —
    all of which otherwise present as "training ran fine but the model outputs
    nonsense" two days later.

    DO NOT PROCEED TO THE REAL RUN UNTIL THIS PASSES.
    """
    print("\n" + "=" * 70)
    print("OVERFIT GATE: 20 samples, loss must approach ~0")
    print("If it plateaus above ~0.5, your data pipeline is broken. Stop and debug.")
    print("=" * 70 + "\n")
    # Run the trainer below with samples[:20], high LR, many epochs, no eval.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--overfit-test", action="store_true", help="run the 20-sample gate and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["training"]["seed"])

    train_samples = load_split(args.root, "train")
    print(f"train samples: {len(train_samples)}")

    model, processor = build_model(cfg)
    collate = make_collator(processor, cfg)

    if args.overfit_test:
        overfit_gate(model, processor, cfg, train_samples[:20])
        return

    # TODO-3 — WIRE UP THE TRAINER
    #
    # The TRL SFTTrainer API changes between releases. Read the docs for the
    # version you actually installed (`pip show trl`) rather than copying a
    # blog post — mismatched argument names are the single most common source
    # of "why won't this run" on this stack.
    #
    # Settings that matter, and why:
    #   per_device_train_batch_size=1     memory. accumulate instead.
    #   gradient_accumulation_steps=8     effective batch 8 without the VRAM
    #   gradient_checkpointing=True       trades compute for memory
    #   optim="paged_adamw_8bit"          survives fragmentation spikes
    #   lr_scheduler_type="cosine"        warmup_ratio ~0.03
    #   fp16=True on T4 / bf16=True on L4+
    #   save_steps                        Colab WILL disconnect. Checkpoint.
    #
    # If loss goes NaN early on a T4: that is the fp16 + 4-bit interaction, not
    # your learning rate. Lower the LR and add warmup, or move to a bf16 GPU.
    raise NotImplementedError(
        "TODO-3: wire up SFTTrainer for your installed TRL version, then remove this."
    )


if __name__ == "__main__":
    main()
