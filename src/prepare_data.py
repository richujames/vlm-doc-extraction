"""
Download SROIE from HuggingFace and write it into the layout this project expects.

    pip install datasets
    python src/prepare_data.py --output data

Source: jsdnrs/ICDAR2019-SROIE (CC-BY-4.0) — the ICDAR 2019 Task 3 dataset with
14 missing test annotations restored and image metadata stripped. 987 receipts:
626 train / 361 test.

Produces:

    data/
      train/img/X00016469612.jpg      train/entities/X00016469612.txt
      test/img/X51005757343.jpg       test/entities/X51005757343.txt

Each entities/*.txt is a JSON object: {"company", "date", "address", "total"}.

This is plumbing, so it is written for you. What is NOT written for you is
looking at the output — run src/data.py afterwards and read some samples. There
are real quirks in this data (see LABEL QUIRKS below) that will shape how you
interpret every number you produce.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FIELDS = ["company", "date", "address", "total"]
HF_DATASET = "jsdnrs/ICDAR2019-SROIE"


def write_split(ds, split_name: str, out_root: Path) -> dict:
    img_dir = out_root / split_name / "img"
    ent_dir = out_root / split_name / "entities"
    img_dir.mkdir(parents=True, exist_ok=True)
    ent_dir.mkdir(parents=True, exist_ok=True)

    stats = {"n": 0, "empty_fields": Counter(), "currency_in_total": 0}

    for row in ds:
        doc_id = row["key"]
        row["image"].convert("RGB").save(img_dir / f"{doc_id}.jpg", quality=95)

        entities = row["entities"] or {}
        record = {f: str(entities.get(f, "") or "").strip() for f in FIELDS}
        (ent_dir / f"{doc_id}.txt").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )

        stats["n"] += 1
        for f in FIELDS:
            if not record[f]:
                stats["empty_fields"][f] += 1
        if any(sym in record["total"] for sym in ("$", "RM", "£", "€")):
            stats["currency_in_total"] += 1

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data"))
    ap.add_argument("--dataset", default=HF_DATASET)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")

    print(f"downloading {args.dataset} ...")
    ds = load_dataset(args.dataset)
    print({k: len(v) for k, v in ds.items()})

    all_stats = {}
    for split in ("train", "test"):
        if split not in ds:
            print(f"  warn: no '{split}' split in this dataset")
            continue
        print(f"writing {split} ...")
        all_stats[split] = write_split(ds[split], split, args.output)

    print("\n" + "=" * 68)
    for split, s in all_stats.items():
        print(f"{split}: {s['n']} samples")
        if s["empty_fields"]:
            print(f"    empty ground-truth fields: {dict(s['empty_fields'])}")
        print(f"    totals containing a currency symbol: {s['currency_in_total']}")


if __name__ == "__main__":
    main()