#!/usr/bin/env python3
"""Run P-M1 on Uni-MoE-2.0-Omni: modality workload decomposition.

The protocol is frozen in
research_paper/iclr2027/evidence/pm1_unimoe2_workload_decomposition_protocol.json.

For every OmniVideoBench example this runner rebuilds the omni evaluation
conversation (the same mp4 supplied as both an audio and a video element,
matching the official examples/cookbook_video.py omni path), applies the
Uni-MoE-2 processor chat template, expands the multimodal placeholders, and
classifies every non-padding input token into exactly one of
video/audio/image/control/text.  It counts baseline token-expert pairs under
the model's data-independent shared-expert budget across the 28 MoE layers,
split into three depth groups.  No model generation is performed; this is a
template-expansion and routing-structure audit, so it only needs the processor
and the model config, not a forward pass.  It runs on the GPU worker so the
exact evaluation processor path (video/audio decoding) matches the PM-2 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO
    / "research_paper/iclr2027/evidence/pm1_unimoe2_workload_decomposition_protocol.json"
)
sys.path.insert(0, "/home/tiger/Uni-MoE/Uni-MoE-2")
MODALITY_ORDER = ["video", "audio", "image", "control", "text"]


def log(*parts: Any) -> None:
    print("[P-M1-UniMoE2]", *parts, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def build_conversation(
    video_path: Path, record: dict[str, Any]
) -> list[dict[str, Any]]:
    options = "\n".join(str(option) for option in record["options"])
    prompt = (
        f"{record['question']}\nOptions:\n{options}\n"
        "Answer with only the option letter."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(video_path)},
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": f"<video>\n<audio>\n{prompt}"},
            ],
        }
    ]


def expand_placeholders(texts: str) -> str:
    return (
        texts.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        .replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")
        .replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
    )


def classify_tokens(
    token_ids: np.ndarray, modality_token_ids: dict[str, list[int]]
) -> dict[str, int]:
    counts = {name: 0 for name in MODALITY_ORDER}
    id_to_class: dict[int, str] = {}
    for name in ["video", "audio", "image", "control"]:
        for tid in modality_token_ids[name]:
            id_to_class[int(tid)] = name
    for tid in token_ids.tolist():
        counts[id_to_class.get(int(tid), "text")] += 1
    return counts


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return "unavailable"


def environment_record(
    protocol: dict[str, Any], dataset_path: Path, model_path: Path
) -> dict[str, Any]:
    import torch

    config_path = model_path / "config.json"
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "requested_worker": os.environ.get("PM1_WORKER_ID", "unknown"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "model_config_sha256": sha256_file(config_path)
        if config_path.exists()
        else None,
        "protocol": protocol,
    }


def depth_group_order(protocol: dict[str, Any]) -> list[str]:
    return list(protocol["moe_depth_groups"].keys())


def run_sample(
    *,
    sample_index: int,
    record: dict[str, Any],
    processor: Any,
    process_mm_info: Callable[..., Any],
    protocol: dict[str, Any],
    pad_token_id: int,
) -> dict[str, Any]:
    video_path = (REPO / "eval_data/OmniVideoBench" / str(record["video"])).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    conversation = build_conversation(video_path, record)
    texts = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    texts = expand_placeholders(texts)
    image_inputs, video_inputs, audio_inputs = process_mm_info(conversation)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    flat = input_ids.detach().reshape(-1).to("cpu")
    attention = inputs.get("attention_mask")
    if attention is not None:
        mask = attention.detach().reshape(-1).to("cpu").bool()
    else:
        mask = flat.ne(int(pad_token_id))
    non_pad = flat[mask].numpy().astype(np.int64)
    total_with_pad = int(flat.numel())

    counts = classify_tokens(non_pad, protocol["modality_token_ids"])
    total_non_pad = int(non_pad.size)
    partition_sum = sum(counts.values())
    if partition_sum != total_non_pad:
        raise AssertionError(
            f"Partition mismatch: sum {partition_sum} != non-pad {total_non_pad}"
        )

    expert_count = int(protocol["moe_structure"]["baseline_expert_count_per_token"])
    total_layers = int(protocol["moe_structure"]["total_moe_layers"])
    depth_groups = protocol["moe_depth_groups"]
    pairs_by_modality = {
        name: counts[name] * expert_count * total_layers for name in MODALITY_ORDER
    }
    pairs_by_depth: dict[str, dict[str, int]] = {}
    for group_name in depth_group_order(protocol):
        n_layers = len(depth_groups[group_name])
        pairs_by_depth[group_name] = {
            name: counts[name] * expert_count * n_layers for name in MODALITY_ORDER
        }

    return {
        "sample_index": sample_index,
        "video": record["video"],
        "question_type": record.get("question_type"),
        "audio_type": record.get("audio_type"),
        "duration": record.get("duration"),
        "input_sequence_length": total_with_pad,
        "non_padding_tokens": total_non_pad,
        "token_counts_by_modality": counts,
        "baseline_token_expert_pairs_by_modality": pairs_by_modality,
        "baseline_token_expert_pairs_by_depth_group": pairs_by_depth,
        "baseline_expert_count_per_token": expert_count,
        "total_moe_layers": total_layers,
    }


def summarize(
    rows: list[dict[str, Any]], errors: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    n = len(rows)
    groups = depth_group_order(protocol)
    total_tokens_by_mod = {name: 0 for name in MODALITY_ORDER}
    total_pairs_by_mod = {name: 0 for name in MODALITY_ORDER}
    total_pairs_by_depth = {g: {name: 0 for name in MODALITY_ORDER} for g in groups}
    seq_lengths: list[int] = []
    non_pad_total = 0
    for row in rows:
        for name in MODALITY_ORDER:
            total_tokens_by_mod[name] += int(row["token_counts_by_modality"][name])
            total_pairs_by_mod[name] += int(
                row["baseline_token_expert_pairs_by_modality"][name]
            )
            for g in groups:
                total_pairs_by_depth[g][name] += int(
                    row["baseline_token_expert_pairs_by_depth_group"][g][name]
                )
        seq_lengths.append(int(row["input_sequence_length"]))
        non_pad_total += int(row["non_padding_tokens"])

    grand_tokens = sum(total_tokens_by_mod.values())
    grand_pairs = sum(total_pairs_by_mod.values())
    token_fraction = {
        name: (total_tokens_by_mod[name] / grand_tokens if grand_tokens else None)
        for name in MODALITY_ORDER
    }
    pair_fraction = {
        name: (total_pairs_by_mod[name] / grand_pairs if grand_pairs else None)
        for name in MODALITY_ORDER
    }
    seq_arr = (
        np.asarray(seq_lengths, dtype=np.float64) if seq_lengths else np.asarray([0.0])
    )

    minimum = int(protocol["gate"]["minimum_completed_samples"])
    partition_ok = all(
        sum(row["token_counts_by_modality"].values()) == int(row["non_padding_tokens"])
        for row in rows
    )
    verified = n >= minimum and partition_ok

    return {
        "schema_version": "unimoe2/pm1-results-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": "P-M1-UniMoE2",
        "backbone": protocol["backbone"],
        "evidence_status": "verified" if verified else "placeholder",
        "completed_samples": n,
        "failed_samples": len(errors),
        "requested_samples": int(protocol["sample_selection"]["target_samples"]),
        "grand_total_non_padding_tokens": non_pad_total,
        "token_counts_by_modality": total_tokens_by_mod,
        "token_fraction_by_modality": token_fraction,
        "baseline_token_expert_pairs_by_modality": total_pairs_by_mod,
        "pair_fraction_by_modality": pair_fraction,
        "baseline_token_expert_pairs_by_depth_group": total_pairs_by_depth,
        "baseline_expert_count_per_token": int(
            protocol["moe_structure"]["baseline_expert_count_per_token"]
        ),
        "total_moe_layers": int(protocol["moe_structure"]["total_moe_layers"]),
        "sequence_length": {
            "mean": float(seq_arr.mean()),
            "min": float(seq_arr.min()),
            "p25": float(np.quantile(seq_arr, 0.25)),
            "median": float(np.quantile(seq_arr, 0.5)),
            "p75": float(np.quantile(seq_arr, 0.75)),
            "max": float(seq_arr.max()),
        },
        "gates": {
            "minimum_completed_samples": minimum,
            "partition_exact_every_sample": partition_ok,
            "passed": verified,
        },
        "scope": protocol["scope"],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug-only override; changes the frozen run.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    modality_ids = {
        "video": [151656],
        "audio": [151667],
        "image": [151655],
        "control": [151652, 151653],
    }
    ids = np.asarray(
        [151656, 151656, 151667, 151655, 151652, 100, 200, 151653], dtype=np.int64
    )
    counts = classify_tokens(ids, modality_ids)
    assert counts == {"video": 2, "audio": 1, "image": 1, "control": 2, "text": 2}, (
        counts
    )
    assert sum(counts.values()) == len(ids)
    print("P-M1-UniMoE2 classification self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --self-test is used")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    dataset_path = REPO / protocol["dataset"]
    model_path = REPO / protocol["model"]
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    target = int(protocol["sample_selection"]["target_samples"])
    n_records = len(records)
    if args.limit is not None:
        n_records = min(n_records, int(args.limit))
    selected = list(range(min(n_records, target if args.limit is None else n_records)))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.jsonl"
    error_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"

    log("loading Uni-MoE-2 processor")
    from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
    from uni_moe.qwen_vl_utils import process_mm_info

    processor = Qwen2VLProcessor.from_pretrained(str(model_path))
    # data_args is normally set from the model config; load config directly to
    # avoid instantiating the full model for this template-only audit.
    import types

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    processor.data_args = types.SimpleNamespace(**config)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    atomic_json(
        output_dir / "environment.json",
        environment_record(protocol, dataset_path, model_path),
    )

    completed_rows = load_jsonl(raw_path)
    completed = {int(row["sample_index"]) for row in completed_rows}
    errors = load_jsonl(error_path)

    for ordinal, sample_index in enumerate(selected, 1):
        if sample_index in completed:
            continue
        try:
            row = run_sample(
                sample_index=sample_index,
                record=records[sample_index],
                processor=processor,
                process_mm_info=process_mm_info,
                protocol=protocol,
                pad_token_id=int(pad_token_id),
            )
            append_jsonl(raw_path, row)
            completed_rows.append(row)
            completed.add(sample_index)
            if ordinal % 25 == 0 or ordinal == 1:
                log(
                    f"[{ordinal}/{len(selected)}] sample {sample_index} done; "
                    f"video_tok={row['token_counts_by_modality']['video']} "
                    f"audio_tok={row['token_counts_by_modality']['audio']} "
                    f"text_tok={row['token_counts_by_modality']['text']}"
                )
            atomic_json(
                output_dir / "progress.json",
                {
                    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                    "completed_samples": len(completed_rows),
                    "requested_samples": len(selected),
                    "error_records": len(errors),
                    "last_completed_sample_index": sample_index,
                },
            )
        except Exception as exc:
            error = {
                "sample_index": sample_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(error_path, error)
            errors.append(error)
            log(f"sample {sample_index} failed: {exc!r}")

    final_summary = summarize(completed_rows, errors, protocol)
    atomic_json(summary_path, final_summary)
    log(
        f"finished: status={final_summary['evidence_status']} "
        f"completed={final_summary['completed_samples']} "
        f"failed={final_summary['failed_samples']}"
    )
    return 0 if final_summary["evidence_status"] == "verified" else 3


if __name__ == "__main__":
    raise SystemExit(main())
