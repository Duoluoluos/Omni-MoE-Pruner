#!/usr/bin/env python3
"""Materialize a fixed, leakage-free OmniVideo LoRA train/validation split.

The selected validation members are taken from a prior clean seed-42 ARC run,
whose automatically generated validation rows preserve the source example in a
slightly normalized form.  This tool matches those rows back to the original
JSONL, then writes original-format train and validation files.  Both arms of a
new paired run can therefore consume identical, explicit files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalized_key(record: dict) -> str:
    """Normalize Swift's saved validation-row format to the source JSONL form."""
    row = dict(record)
    row.pop("dataset", None)
    messages = []
    for message in row["messages"]:
        messages.append(json.loads(message) if isinstance(message, str) else message)
    row["messages"] = messages
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "ft_data" / "omnivideo_100k_lora.jsonl",
    )
    parser.add_argument(
        "--reference-val",
        type=Path,
        default=root
        / "models"
        / "omnivideo_100k_qwen3_omni_lora_arc_sceb"
        / "v0-20260730-162842"
        / "val_dataset.jsonl",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=root / "ft_data" / "omnivideo_100k_lora_seed42_train.jsonl",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=root / "ft_data" / "omnivideo_100k_lora_seed42_val.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {
        normalized_key(json.loads(line))
        for line in args.reference_val.read_text(encoding="utf-8").splitlines()
        if line
    }
    if not selected:
        raise RuntimeError("Reference validation file is empty.")

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    source_count = 0
    with (
        args.source.open("r", encoding="utf-8") as source,
        args.train_output.open("w", encoding="utf-8") as train_out,
        args.val_output.open("w", encoding="utf-8") as val_out,
    ):
        for line in source:
            if not line.strip():
                continue
            source_count += 1
            record = json.loads(line)
            if normalized_key(record) in selected:
                val_out.write(line if line.endswith("\n") else f"{line}\n")
                matched += 1
            else:
                train_out.write(line if line.endswith("\n") else f"{line}\n")

    if matched != len(selected):
        raise RuntimeError(
            f"Matched {matched}/{len(selected)} selected validation examples; refusing partial split."
        )
    print(
        json.dumps(
            {
                "source_examples": source_count,
                "validation_examples": matched,
                "training_examples": source_count - matched,
                "train_output": str(args.train_output),
                "val_output": str(args.val_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
