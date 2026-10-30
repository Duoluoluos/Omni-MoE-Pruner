#!/usr/bin/env python3
"""Audit and summarize a completed E7 cutoff-gap matched-control run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    try:
        name = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        name = str(path.resolve())
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def stable_rank(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def proportional_stratified_indices(
    records: list[dict[str, Any]], target: int, seed: int
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[str(record.get("question_type", "unknown"))].append(idx)
    exact = {name: target * len(indices) / len(records) for name, indices in groups.items()}
    allocation = {name: int(math.floor(value)) for name, value in exact.items()}
    remainder = target - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda name: (-(exact[name] - allocation[name]), stable_rank(seed, "stratum", name)),
    )
    for name in order[:remainder]:
        allocation[name] += 1
    selected: list[int] = []
    for name, indices in sorted(groups.items()):
        ranked = sorted(indices, key=lambda idx: stable_rank(seed, "sample", idx))
        selected.extend(ranked[: allocation[name]])
    return sorted(selected)


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    clusters: list[str],
    seed: int,
    draws: int = 10000,
) -> dict[str, float | int]:
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for idx, cluster in enumerate(clusters):
        by_cluster[cluster].append(idx)
    names = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = [idx for name in sampled for idx in by_cluster[str(name)]]
        estimates[draw] = float(values[indices].mean())
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "clusters": len(names),
        "samples": len(values),
    }


def sample_bootstrap_mean_ci(
    values: np.ndarray, seed: int, draws: int = 10000
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    chunk = 1000
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def mean_bool(rows: list[dict[str, Any]], arm: str, field: str) -> float:
    return float(np.mean([bool(row["arms"][arm][field]) for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = repo_root / "research_paper/iclr2027/evidence/e7_cutoff_gap_protocol.json"
    runner_path = repo_root / "scripts/run_e7_cutoff_gap_control.py"
    analyzer_path = Path(__file__).resolve()
    worker_log_path = repo_root / "logs/e7_cutoff_gap_worker4177575_20260824/run.log"
    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "raw_results.jsonl"
    error_path = run_dir / "errors.jsonl"
    summary_path = run_dir / "summary.json"
    selection_path = run_dir / "selected_samples.json"
    environment_path = run_dir / "environment.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    dataset_path = repo_root / str(protocol["dataset"])
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = load_jsonl(raw_path)
    errors = load_jsonl(error_path) if error_path.is_file() else []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if environment.get("requested_worker") != str(protocol["execution_worker"]):
        raise SystemExit("Formal run did not execute on the frozen worker")
    if environment.get("protocol") != protocol:
        raise SystemExit("Formal-run embedded protocol differs from the live frozen protocol")
    provenance_hashes = {
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(runner_path),
        "dataset_sha256": sha256_file(dataset_path),
    }
    for key, expected in provenance_hashes.items():
        if environment.get(key) != expected:
            raise SystemExit(f"Formal-run provenance hash mismatch: {key}")
    target = int(protocol["sample_selection"]["target_samples"])
    seed = int(protocol["sample_selection"]["seed"])
    recomputed_selection = proportional_stratified_indices(dataset, target, seed)
    if [int(value) for value in selection["indices"]] != recomputed_selection:
        raise SystemExit("Selected samples do not match the frozen stratified procedure")
    if selection.get("dataset_sha256") != provenance_hashes["dataset_sha256"]:
        raise SystemExit("Selection record carries a different dataset hash")
    if summary.get("evidence_status") != "verified":
        raise SystemExit("Refusing final analysis: E7 summary is not verified")
    if len(rows) != int(summary["paired_samples"]):
        raise SystemExit("raw_results.jsonl and summary.json sample counts differ")
    if len(errors) != int(summary["failed_samples"]):
        raise SystemExit("errors.jsonl and summary.json failure counts differ")
    if len(rows) + len(errors) != target:
        raise SystemExit("The frozen 128-sample target was not fully accounted for")
    if (
        int(progress["paired_samples"]) != len(rows)
        or int(progress["requested_samples"]) != target
    ):
        raise SystemExit("progress.json is inconsistent with final run artifacts")
    if int(progress["error_records"]) != len(errors):
        raise SystemExit("progress.json error count differs from errors.jsonl")
    minimum_n = int(protocol["sample_selection"]["minimum_paired_samples"])
    if len(rows) < minimum_n:
        raise SystemExit("Fewer paired samples than the frozen verification minimum")
    if int(summary["requested_samples"]) != target:
        raise SystemExit("Summary target differs from the frozen sample target")
    if not bool(summary["implementation_gates"]["passed"]):
        raise SystemExit("Runner implementation gates are not marked passed")
    completed_indices = [int(row["sample_index"]) for row in rows]
    completed_set = set(completed_indices)
    error_indices = [int(row["sample_index"]) for row in errors]
    if (
        len(set(error_indices)) != len(error_indices)
        or completed_set.intersection(error_indices)
        or not set(error_indices).issubset({int(value) for value in selection["indices"]})
    ):
        raise SystemExit("Error rows are duplicated, completed, or outside the frozen selection")
    frozen_completed_order = [
        int(value) for value in selection["indices"] if int(value) in completed_set
    ]
    if completed_indices != frozen_completed_order or len(completed_set) != len(completed_indices):
        raise SystemExit("Completed rows are duplicated or do not follow the frozen selection order")
    expected_arm_names = set(protocol["arms"]) - {"native_top8"}
    for row in rows:
        if set(row["arms"]) != expected_arm_names:
            raise SystemExit(f"Sample {row['sample_index']} has an incomplete intervention set")
        sample_index = int(row["sample_index"])
        record = dataset[sample_index]
        if (
            row["video"] != record["video"]
            or row["question_type"] != record.get("question_type")
            or row["correct_option"] != str(record["correct_option"]).upper()
        ):
            raise SystemExit(f"Sample {sample_index} metadata differs from the frozen dataset")
    gap_dir = run_dir / "routing_gaps"
    expected_gap_names = {f"sample_{index:04d}.npz" for index in completed_indices}
    selected_gap_names = {f"sample_{int(index):04d}.npz" for index in selection["indices"]}
    actual_gap_paths = sorted(gap_dir.glob("sample_*.npz"))
    actual_gap_names = {path.name for path in actual_gap_paths}
    missing = sorted(expected_gap_names - actual_gap_names)
    unexpected = sorted(actual_gap_names - selected_gap_names)
    if missing or unexpected:
        raise SystemExit(
            f"Routing-gap archive mismatch: missing_completed={missing}, "
            f"outside_frozen_selection={unexpected}"
        )

    # Reconstruct every intervention allocation from the retained native-routing
    # gaps.  This deliberately does not trust the runner's count or allocation
    # hashes: the archive is the independent source for this audit.
    threshold = float(protocol["policy"]["cutoff_gap_threshold"])
    random_arm_names = sorted(
        name for name in protocol["arms"] if name.startswith("random_matched_")
    )
    expected_gap_keys = {
        f"layer_{layer:02d}_{gap_name}"
        for layer in protocol["policy"]["eligible_moe_layers"]
        for gap_name in ("g4", "g6")
    }
    reconstructed_archives = 0
    for row in rows:
        sample_index = int(row["sample_index"])
        gap_path = gap_dir / f"sample_{sample_index:04d}.npz"
        with np.load(gap_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_gap_keys:
                raise SystemExit(
                    f"Sample {sample_index} routing-gap keys differ from the frozen layers"
                )
            for layer in protocol["policy"]["eligible_moe_layers"]:
                layer_key = str(layer)
                layer_record = row["budget_audit"]["layers"].get(layer_key)
                if layer_record is None:
                    raise SystemExit(
                        f"Sample {sample_index} has no budget audit for layer {layer}"
                    )
                g4 = np.asarray(archive[f"layer_{layer:02d}_g4"])
                g6 = np.asarray(archive[f"layer_{layer:02d}_g6"])
                expected_tokens = int(row["video_tokens"])
                if g4.shape != (expected_tokens,) or g6.shape != (expected_tokens,):
                    raise SystemExit(
                        f"Sample {sample_index} layer {layer} gap shape is inconsistent"
                    )
                if not np.all(np.isfinite(g4)) or not np.all(np.isfinite(g6)):
                    raise SystemExit(
                        f"Sample {sample_index} layer {layer} contains non-finite gaps"
                    )
                if np.any(g4 < 0.0) or np.any(g6 < 0.0):
                    raise SystemExit(
                        f"Sample {sample_index} layer {layer} contains a negative cutoff gap"
                    )

                high = np.full(expected_tokens, 8, dtype=np.int16)
                choose4 = g4 >= threshold
                choose6 = (~choose4) & (g6 >= threshold)
                high[choose4] = 4
                high[choose6] = 6
                counts = {
                    str(q): int(np.count_nonzero(high == q)) for q in (4, 6, 8)
                }
                if counts != {
                    str(q): int(value)
                    for q, value in layer_record["reference_counts"].items()
                }:
                    raise SystemExit(
                        f"Sample {sample_index} layer {layer} reference counts "
                        "cannot be reconstructed from the gap archive"
                    )

                n4, n6 = counts["4"], counts["6"]
                low = np.full(expected_tokens, 8, dtype=np.int16)
                low4 = np.argsort(g4, kind="mergesort")[:n4]
                low[low4] = 4
                remaining = np.flatnonzero(low == 8)
                low6 = remaining[np.argsort(g6[remaining], kind="mergesort")[:n6]]
                low[low6] = 6
                reconstructed = {
                    "high_gap": high,
                    "low_gap_matched": low,
                }
                for arm_name in random_arm_names:
                    arm_seed = int(arm_name.rsplit("_", 1)[1])
                    rng = np.random.default_rng(stable_rank(arm_seed, sample_index, layer))
                    order = rng.permutation(expected_tokens)
                    random_budget = np.full(expected_tokens, 8, dtype=np.int16)
                    random_budget[order[:n4]] = 4
                    random_budget[order[n4 : n4 + n6]] = 6
                    reconstructed[arm_name] = random_budget

                for arm_name, budget in reconstructed.items():
                    reconstructed_counts = {
                        str(q): int(np.count_nonzero(budget == q)) for q in (4, 6, 8)
                    }
                    declared_counts = {
                        str(q): int(value)
                        for q, value in layer_record["arm_counts"][arm_name].items()
                    }
                    if reconstructed_counts != declared_counts:
                        raise SystemExit(
                            f"Sample {sample_index} layer {layer} {arm_name} counts differ"
                        )
                    reconstructed_hash = hashlib.sha256(budget.tobytes()).hexdigest()
                    if reconstructed_hash != layer_record["allocation_sha256"][arm_name]:
                        raise SystemExit(
                            f"Sample {sample_index} layer {layer} {arm_name} allocation hash differs"
                        )

                for gap_name, values in (("g4", g4), ("g6", g6)):
                    declared_quantiles = np.asarray(
                        layer_record[f"{gap_name}_quantiles"], dtype=np.float64
                    )
                    reconstructed_quantiles = np.quantile(
                        values, [0.0, 0.25, 0.5, 0.75, 1.0]
                    )
                    if not np.allclose(
                        declared_quantiles,
                        reconstructed_quantiles,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise SystemExit(
                            f"Sample {sample_index} layer {layer} {gap_name} quantiles differ"
                        )
        reconstructed_archives += 1

    random_arms = sorted(name for name in rows[0]["arms"] if name.startswith("random_matched_"))
    high = np.asarray([row["arms"]["high_gap"]["js_divergence_nats"] for row in rows])
    low = np.asarray([row["arms"]["low_gap_matched"]["js_divergence_nats"] for row in rows])
    random_js_matrix = np.asarray(
        [[row["arms"][arm]["js_divergence_nats"] for arm in random_arms] for row in rows]
    )
    random_mean = random_js_matrix.mean(axis=1)
    videos = [str(row["video"]) for row in rows]
    recomputed_primary = {
        "high_gap": sample_bootstrap_mean_ci(high, seed + 1),
        "low_gap_matched": sample_bootstrap_mean_ci(low, seed + 2),
        "random_matched_within_sample_mean": sample_bootstrap_mean_ci(random_mean, seed + 3),
        "effect_low_minus_high": sample_bootstrap_mean_ci(low - high, seed + 4),
        "effect_random_minus_high": sample_bootstrap_mean_ci(random_mean - high, seed + 5),
    }
    for endpoint, values in recomputed_primary.items():
        declared = summary["primary_endpoint"][endpoint]
        for field, expected in values.items():
            if not np.isclose(float(declared[field]), expected, rtol=0.0, atol=1e-15):
                raise SystemExit(f"Primary endpoint mismatch: {endpoint}.{field}")
    recomputed_support = (
        recomputed_primary["effect_low_minus_high"]["ci95_low"] > 0.0
        and recomputed_primary["effect_random_minus_high"]["ci95_low"] > 0.0
    )
    if recomputed_support:
        hypothesis_outcome = "supported"
    elif (
        recomputed_primary["effect_low_minus_high"]["ci95_high"] < 0.0
        and recomputed_primary["effect_random_minus_high"]["ci95_high"] < 0.0
    ):
        hypothesis_outcome = "opposite_direction"
    else:
        hypothesis_outcome = "not_supported"
    if bool(summary["predictive_validity_supported"]) != bool(recomputed_support):
        raise SystemExit("Predictive-validity gate differs from the raw paired results")

    eligible_layers = {str(value) for value in protocol["policy"]["eligible_moe_layers"]}
    max_budget_diff = 0
    for row in rows:
        row_max_budget_diff = 0
        layer_records = row["budget_audit"]["layers"]
        if set(layer_records) != eligible_layers:
            raise SystemExit(f"Sample {row['sample_index']} has an incomplete layer audit")
        recomputed_counts: Counter[int] = Counter()
        for layer_record in layer_records.values():
            reference = {str(q): int(v) for q, v in layer_record["reference_counts"].items()}
            recomputed_counts.update({int(q): count for q, count in reference.items()})
            for arm_name in expected_arm_names:
                arm_counts = {
                    str(q): int(v) for q, v in layer_record["arm_counts"][arm_name].items()
                }
                row_max_budget_diff = max(
                    row_max_budget_diff,
                    max(abs(arm_counts[str(q)] - reference[str(q)]) for q in (4, 6, 8)),
                )
        declared_counts = {int(q): int(v) for q, v in row["budget"]["top_m_counts"].items()}
        if declared_counts != dict(recomputed_counts):
            raise SystemExit(f"Sample {row['sample_index']} pooled budget counts are inconsistent")
        if int(row["budget_audit"]["maximum_abs_count_difference"]) != row_max_budget_diff:
            raise SystemExit(f"Sample {row['sample_index']} budget-audit maximum is inconsistent")
        max_budget_diff = max(max_budget_diff, row_max_budget_diff)
    identities = [row["identity"] for row in rows if row.get("identity") is not None]
    if max_budget_diff != 0:
        raise SystemExit("Budget audit is not exact")
    if len(identities) < 4:
        raise SystemExit("Fewer than four identity validations are present")
    identity_indices = [
        int(row["sample_index"]) for row in rows if row.get("identity") is not None
    ]
    if identity_indices != completed_indices[: len(identity_indices)]:
        raise SystemExit("Identity validations are not attached to the first completed samples")
    identity_limit = float(
        protocol["implementation_validation"]["maximum_identity_js_divergence_nats"]
    )
    if not np.isclose(
        float(summary["implementation_gates"]["identity_limit"]),
        identity_limit,
        rtol=0.0,
        atol=0.0,
    ):
        raise SystemExit("Summary identity limit differs from the frozen protocol")
    max_identity_js = max(float(item["js_divergence_nats"]) for item in identities)
    all_identity_top1 = all(bool(item["native_top1_retained"]) for item in identities)
    all_identity_text = all(bool(item["output_text_equal"]) for item in identities)
    if max_identity_js > identity_limit:
        raise SystemExit("Identity JS-divergence gate failed")
    if not all_identity_top1:
        raise SystemExit("Identity native-top1 gate failed")
    if not all_identity_text:
        raise SystemExit("Identity output-text gate failed")

    pooled_counts: Counter[int] = Counter()
    for row in rows:
        pooled_counts.update({int(q): int(v) for q, v in row["budget"]["top_m_counts"].items()})
    pooled_decisions = sum(pooled_counts.values())
    pooled_budget = sum(q * count for q, count in pooled_counts.items()) / pooled_decisions

    cluster_effects = {
        "low_minus_high": cluster_bootstrap_mean_ci(low - high, videos, 3707),
        "random_minus_high": cluster_bootstrap_mean_ci(random_mean - high, videos, 4707),
    }
    cluster_support = (
        cluster_effects["low_minus_high"]["ci95_low"] > 0.0
        and cluster_effects["random_minus_high"]["ci95_low"] > 0.0
    )

    parsed_native = [row["native"]["answer"] is not None for row in rows]
    parsed_arms = [
        row["arms"][arm]["answer"] is not None
        for row in rows
        for arm in ["high_gap", "low_gap_matched", *random_arms]
    ]
    analysis = {
        "schema_version": "arc-sceb/e7-analysis-v1",
        "experiment_id": "E7",
        "evidence_status": "verified",
        "provenance_audit": {
            "execution_worker": str(environment["requested_worker"]),
            "frozen_at": protocol["frozen_at"],
            "requested_samples": target,
            "minimum_paired_samples": minimum_n,
            "paired_samples": len(rows),
            "failed_samples": int(summary["failed_samples"]),
            "unique_videos": len(set(videos)),
            "sample_selection_recomputed_exactly": True,
            "paired_order_matches_frozen_selection": True,
            "all_intervention_arms_present": True,
            "routing_gap_archives_reconstructed": reconstructed_archives,
            "allocation_hashes_recomputed": True,
            "progress_error_count_matches_final_errors": (
                int(progress["error_records"]) == len(errors)
            ),
            **provenance_hashes,
            "analysis_script_sha256": sha256_file(analyzer_path),
        },
        "primary_analysis": {
            "endpoint": "first_generated_token_js_divergence_nats",
            "paired_unit": "question",
            "bootstrap_draws": 10000,
            "confidence_level": 0.95,
            "bootstrap_base_seed": seed,
            "endpoint_seeds": {
                "high_gap": seed + 1,
                "low_gap_matched": seed + 2,
                "random_matched_within_sample_mean": seed + 3,
                "effect_low_minus_high": seed + 4,
                "effect_random_minus_high": seed + 5,
            },
            "seed_note": "The frozen runner uses the protocol seed as a base and fixed endpoint offsets.",
            "random_arm_count": len(random_arms),
            "direction": "lower_is_less_output_sensitive",
        },
        "primary_estimates": recomputed_primary,
        "prespecified_predictive_validity_supported": bool(
            summary["predictive_validity_supported"]
        ),
        "prespecified_hypothesis_outcome": hypothesis_outcome,
        "cluster_bootstrap_sensitivity": {
            "unit": "video",
            "confidence_level": 0.95,
            "note": "Post-registered sensitivity analysis; the frozen primary analysis remains the paired-sample bootstrap.",
            "effects": cluster_effects,
            "predictive_validity_supported": bool(cluster_support),
        },
        "descriptive_robustness": {
            "high_gap_lower_js_than_low_gap_fraction": float(np.mean(high < low)),
            "high_gap_lower_js_than_random_mean_fraction": float(np.mean(high < random_mean)),
            "high_gap_js_median": float(np.median(high)),
            "low_gap_js_median": float(np.median(low)),
            "random_mean_js_median": float(np.median(random_mean)),
        },
        "budget_audit": {
            "maximum_abs_count_difference": max_budget_diff,
            "pooled_top_m_counts": {str(q): int(pooled_counts[q]) for q in (4, 6, 8)},
            "pooled_mean_expert_budget": float(pooled_budget),
            "pooled_video_token_layer_decisions": int(pooled_decisions),
        },
        "identity_audit": {
            "samples": len(identities),
            "maximum_js_divergence_nats": max_identity_js,
            "identity_limit": identity_limit,
            "all_native_top1_retained": all_identity_top1,
            "all_output_text_equal": all_identity_text,
        },
        "answer_parse_coverage": {
            "native": float(np.mean(parsed_native)),
            "intervention_arms": float(np.mean(parsed_arms)),
        },
        "artifact_sha256": {
            "raw_results_jsonl": sha256_file(raw_path),
            "summary_json": sha256_file(summary_path),
            "selected_samples_json": sha256_file(selection_path),
            "environment_json": sha256_file(environment_path),
            "routing_gap_archives": {
                "paired_sample_count": len(expected_gap_names),
                "total_archive_count": len(actual_gap_paths),
                "combined_sha256": hashlib.sha256(
                    "".join(
                        f"{path.name}\t{sha256_file(path)}\n" for path in actual_gap_paths
                    ).encode("utf-8")
                ).hexdigest(),
            },
        },
    }
    atomic_json(run_dir / "analysis.json", analysis)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    random_tv = np.asarray(
        [[row["arms"][arm]["total_variation"] for arm in random_arms] for row in rows]
    ).mean(axis=1)
    table_rows = [
        {
            "arm": "native_top8",
            "paired_samples": len(rows),
            "mean_js_nats": 0.0,
            "js_ci95_low": 0.0,
            "js_ci95_high": 0.0,
            "mean_total_variation": 0.0,
            "native_top1_retention": 1.0,
            "answer_flip_rate": 0.0,
            "accuracy": float(summary["native"]["accuracy"]),
        },
        {
            "arm": "high_gap",
            "paired_samples": len(rows),
            "mean_js_nats": summary["primary_endpoint"]["high_gap"]["mean"],
            "js_ci95_low": summary["primary_endpoint"]["high_gap"]["ci95_low"],
            "js_ci95_high": summary["primary_endpoint"]["high_gap"]["ci95_high"],
            "mean_total_variation": float(np.mean([row["arms"]["high_gap"]["total_variation"] for row in rows])),
            "native_top1_retention": mean_bool(rows, "high_gap", "native_top1_retained"),
            "answer_flip_rate": mean_bool(rows, "high_gap", "answer_flip_vs_native"),
            "accuracy": float(np.mean([row["arms"]["high_gap"]["is_correct"] for row in rows])),
        },
        {
            "arm": "low_gap_matched",
            "paired_samples": len(rows),
            "mean_js_nats": summary["primary_endpoint"]["low_gap_matched"]["mean"],
            "js_ci95_low": summary["primary_endpoint"]["low_gap_matched"]["ci95_low"],
            "js_ci95_high": summary["primary_endpoint"]["low_gap_matched"]["ci95_high"],
            "mean_total_variation": float(np.mean([row["arms"]["low_gap_matched"]["total_variation"] for row in rows])),
            "native_top1_retention": mean_bool(rows, "low_gap_matched", "native_top1_retained"),
            "answer_flip_rate": mean_bool(rows, "low_gap_matched", "answer_flip_vs_native"),
            "accuracy": float(np.mean([row["arms"]["low_gap_matched"]["is_correct"] for row in rows])),
        },
        {
            "arm": "random_matched_mean",
            "paired_samples": len(rows),
            "mean_js_nats": summary["primary_endpoint"]["random_matched_within_sample_mean"]["mean"],
            "js_ci95_low": summary["primary_endpoint"]["random_matched_within_sample_mean"]["ci95_low"],
            "js_ci95_high": summary["primary_endpoint"]["random_matched_within_sample_mean"]["ci95_high"],
            "mean_total_variation": float(random_tv.mean()),
            "native_top1_retention": float(np.mean([
                np.mean([row["arms"][arm]["native_top1_retained"] for arm in random_arms])
                for row in rows
            ])),
            "answer_flip_rate": float(np.mean([
                np.mean([row["arms"][arm]["answer_flip_vs_native"] for arm in random_arms])
                for row in rows
            ])),
            "accuracy": float(np.mean([
                np.mean([row["arms"][arm]["is_correct"] for arm in random_arms])
                for row in rows
            ])),
        },
    ]
    effects_by_arm = {
        "low_gap_matched": summary["primary_endpoint"]["effect_low_minus_high"],
        "random_matched_mean": summary["primary_endpoint"]["effect_random_minus_high"],
    }
    for row in table_rows:
        row["mean_js_times_1e3"] = 1000.0 * float(row["mean_js_nats"])
        row["js_ci95_low_times_1e3"] = 1000.0 * float(row["js_ci95_low"])
        row["js_ci95_high_times_1e3"] = 1000.0 * float(row["js_ci95_high"])
        effect = effects_by_arm.get(str(row["arm"]))
        row["effect_vs_high_times_1e3"] = (
            "" if effect is None else 1000.0 * float(effect["mean"])
        )
        row["effect_ci95_low_times_1e3"] = (
            "" if effect is None else 1000.0 * float(effect["ci95_low"])
        )
        row["effect_ci95_high_times_1e3"] = (
            "" if effect is None else 1000.0 * float(effect["ci95_high"])
        )
        row["native_top1_retention_percent"] = 100.0 * float(row["native_top1_retention"])
        row["answer_flip_rate_percent"] = 100.0 * float(row["answer_flip_rate"])
        row["accuracy_percent"] = 100.0 * float(row["accuracy"])
    columns = list(table_rows[0])
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table_rows)

    core_paths = [
        protocol_path,
        runner_path,
        analyzer_path,
        worker_log_path,
        raw_path,
        summary_path,
        selection_path,
        environment_path,
        run_dir / "progress.json",
        run_dir / "analysis.json",
        args.csv.resolve(),
    ]
    optional_paths = [run_dir / "errors.jsonl", run_dir / "RUN.md", run_dir / "launcher.pid"]
    manifest = {
        "schema_version": "arc-sceb/e7-artifact-manifest-v1",
        "experiment_id": "E7",
        "evidence_status": "verified",
        "files": [
            manifest_entry(path, repo_root)
            for path in [*core_paths, *optional_paths, *actual_gap_paths]
            if path.is_file()
        ],
    }
    atomic_json(run_dir / "artifact_manifest.json", manifest)
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
