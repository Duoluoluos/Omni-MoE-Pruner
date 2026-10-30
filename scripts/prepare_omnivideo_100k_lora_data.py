#!/usr/bin/env python3
"""Convert OmniVideo-100K QA JSONL files into ms-swift's multimodal SFT format.

The output uses absolute video paths so it is valid from a worker shell, while
the source annotations remain unchanged.  Answers deliberately exclude the
dataset's ``analysis`` field, which is rationale metadata rather than a target
response.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


SOURCE_FILES = (
    "train_oe_70k_formatted.jsonl",
    "train_mcq_30k_formatted.jsonl",
)
SYSTEM_PROMPT = (
    "You are a precise audio-visual assistant. Analyze both the visual content "
    "and audio of the supplied video, then answer the user's question directly."
)


def iter_records(source: Path) -> Iterable[dict]:
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {source}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {source}:{line_number}")
            yield record


def to_sft_record(record: dict, dataset_dir: Path) -> dict:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    video_path = str(record.get("video_path", "")).strip()
    if not question or not answer or not video_path:
        question_id = record.get("question_id", "<unknown>")
        raise ValueError(f"Incomplete sample {question_id!r}")

    absolute_video_path = (dataset_dir / video_path).resolve()
    if not absolute_video_path.is_file():
        raise FileNotFoundError(
            f"Video for {record.get('question_id', '<unknown>')} is missing: "
            f"{absolute_video_path}"
        )

    user_content = (
        "<video>\n"
        "Answer the following question about the video. Use both its visual "
        "content and audio when relevant.\n\n"
        f"{question}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "videos": [str(absolute_video_path)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("ft_data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/omnivideo_100k_lora.jsonl",
    )
    parser.add_argument("--smoke-count", type=int, default=8)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output = args.output or dataset_dir / "omnivideo_100k_lora.jsonl"
    smoke_output = dataset_dir / "omnivideo_100k_lora_smoke.jsonl"
    if args.smoke_count < 1:
        raise ValueError("--smoke-count must be positive")

    sources = [dataset_dir / source for source in SOURCE_FILES]
    missing_sources = [str(source) for source in sources if not source.is_file()]
    if missing_sources:
        raise FileNotFoundError("Missing annotation files: " + ", ".join(missing_sources))

    output.parent.mkdir(parents=True, exist_ok=True)
    smoke_records: list[dict] = []
    counts: dict[str, int] = {}
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False, suffix=".tmp"
    ) as temporary:
        temporary_path = Path(temporary.name)
        for source in sources:
            source_count = 0
            for record in iter_records(source):
                sft_record = to_sft_record(record, dataset_dir)
                json.dump(sft_record, temporary, ensure_ascii=False)
                temporary.write("\n")
                if len(smoke_records) < args.smoke_count:
                    smoke_records.append(sft_record)
                source_count += 1
            counts[source.name] = source_count

    os.replace(temporary_path, output)
    with smoke_output.open("w", encoding="utf-8") as smoke_handle:
        for record in smoke_records:
            json.dump(record, smoke_handle, ensure_ascii=False)
            smoke_handle.write("\n")

    manifest = {
        "output": str(output),
        "smoke_output": str(smoke_output),
        "records": sum(counts.values()),
        "records_by_source": counts,
        "video_root": str((dataset_dir / "videos").resolve()),
    }
    manifest_path = output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        json.dump(manifest, manifest_handle, ensure_ascii=False, indent=2)
        manifest_handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
