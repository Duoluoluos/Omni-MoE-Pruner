#!/usr/bin/env python3
import argparse
import datetime as _dt
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text "
    "and speech. Please analyze the video carefully and select the most appropriate "
    "answer from the given options."
)


COLON_TIME_RE = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):([0-5]\d)(?![\d:])")
EN_TIME_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hr|h)\s*)?"
    r"(?:(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|min|m)\s*)?"
    r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)\b",
    re.IGNORECASE,
)
ZH_TIME_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*小时)?\s*"
    r"(?:(\d+(?:\.\d+)?)\s*分(?:钟)?)?\s*"
    r"(\d+(?:\.\d+)?)\s*秒"
)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_jsonl(path: Path, obj: dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=json_default) + "\n")
        f.flush()


def normalize_sample(sample):
    sample = dict(sample)
    if "video_path" not in sample and sample.get("video"):
        sample["video_path"] = sample["video"]
    if "task_type" not in sample:
        sample["task_type"] = sample.get("question_type") or sample.get("problem_type") or sample.get("video_type") or "Unknown"
    return sample


def load_json(path: Path):
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError(
                f"Reading {path} requires pyarrow. Use eval_data/OmniVideoBench/data.json "
                "or install pyarrow in this Python environment."
            ) from exc
        data = pq.read_table(path).to_pylist()
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return [normalize_sample(sample) for sample in data]


def get_candidates_and_answer(sample):
    if sample.get("options") is not None:
        candidates = []
        for i, option in enumerate(sample.get("options") or []):
            letter = chr(ord("A") + i)
            text = str(option)
            if len(text) >= 2 and text[0].upper() == letter and text[1] in (".", ")", " "):
                candidates.append(f"{letter}. {text[2:].strip()}")
            else:
                candidates.append(f"{letter}. {text}")
        correct = str(sample.get("correct_option") or sample.get("answer", "")).strip().upper()
        return candidates, correct[:1] if correct else ""
    candidates = []
    for opt in ["A", "B", "C", "D", "E", "F"]:
        key = f"option_{opt}"
        if key in sample and sample[key] is not None:
            candidates.append(f"{opt}. {sample[key]}")
    correct = str(sample.get("correct_option") or sample.get("answer", "")).strip().upper()
    return candidates, correct[:1] if correct else ""


def extract_answer(resp_text):
    resp_text = (resp_text or "").strip()
    for opt in ["A", "B", "C", "D", "E", "F"]:
        if resp_text.upper().startswith(opt):
            return opt
    return resp_text[:1].upper() if resp_text else None


def check_video_has_audio(video_path: str) -> bool:
    try:
        import av

        with av.open(video_path) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return False


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def move_inputs_to_model(inputs, model, device_map):
    if device_map == "auto":
        return inputs.to("cuda").to(model.dtype)
    return inputs.to(model.device).to(model.dtype)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni on OmniVideoBench with two-pass evidence retrieval.")
    parser.add_argument("--model-path", default="models/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--data-path", default="eval_data/OmniVideoBench/data.json")
    parser.add_argument("--video-dir", default="eval_data/OmniVideoBench")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-audio-in-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--segment-max-frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--segment-context-seconds", type=float, default=8.0)
    parser.add_argument("--top-segments", type=int, default=2)
    parser.add_argument("--merge-gap-seconds", type=float, default=2.0)
    parser.add_argument("--min-coverage-segments", type=int, default=0)
    parser.add_argument("--time-sigma-seconds", type=float, default=18.0)
    parser.add_argument("--time-cue-weight", type=float, default=4.0)
    parser.add_argument("--question-time-cue-weight", type=float, default=2.0)
    parser.add_argument("--audio-energy-weight", type=float, default=1.0)
    parser.add_argument("--uniform-anchor-weight", type=float, default=0.03)
    parser.add_argument("--audio-energy-sr", type=int, default=8000)
    parser.add_argument("--audio-energy", dest="audio_energy", action="store_true", default=True)
    parser.add_argument("--no-audio-energy", dest="audio_energy", action="store_false")
    parser.add_argument("--oracle-reasoning-times", dest="oracle_reasoning_times", action="store_true", default=True)
    parser.add_argument("--no-oracle-reasoning-times", dest="oracle_reasoning_times", action="store_false")
    parser.add_argument("--question-time-cues", dest="question_time_cues", action="store_true", default=True)
    parser.add_argument("--no-question-time-cues", dest="question_time_cues", action="store_false")
    parser.add_argument("--include-segment-times-in-prompt", action="store_true", default=True)
    parser.add_argument("--no-segment-times-in-prompt", dest="include_segment_times_in_prompt", action="store_false")
    parser.add_argument("--summary-every", type=int, default=10)
    return parser.parse_args()


def parse_duration_seconds(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if all(part.strip().isdigit() for part in parts) and len(parts) in (2, 3):
        nums = [int(part) for part in parts]
        if len(nums) == 2:
            return float(nums[0] * 60 + nums[1])
        return float(nums[0] * 3600 + nums[1] * 60 + nums[2])
    return None


def format_time(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def extract_times_from_text(text: str, source: str, source_weight: float) -> list[dict[str, Any]]:
    cues = []
    if not text:
        return cues
    for match in COLON_TIME_RE.finditer(text):
        h, m, s = match.groups()
        if h is None:
            seconds = int(m) * 60 + int(s)
        else:
            seconds = int(h) * 3600 + int(m) * 60 + int(s)
        cues.append(
            {
                "seconds": float(seconds),
                "source": source,
                "weight": float(source_weight),
                "text": match.group(0),
                "span": [int(match.start()), int(match.end())],
            }
        )
    for regex in (EN_TIME_RE, ZH_TIME_RE):
        for match in regex.finditer(text):
            h, m, s = match.groups()
            seconds = (float(h or 0) * 3600.0) + (float(m or 0) * 60.0) + float(s or 0)
            if seconds <= 0:
                continue
            cues.append(
                {
                    "seconds": float(seconds),
                    "source": source,
                    "weight": float(source_weight),
                    "text": match.group(0),
                    "span": [int(match.start()), int(match.end())],
                }
            )
    return cues


def dedupe_time_cues(cues: list[dict[str, Any]], duration_seconds: float | None) -> list[dict[str, Any]]:
    deduped = []
    for cue in sorted(cues, key=lambda x: (float(x["seconds"]), str(x.get("source")))):
        seconds = float(cue["seconds"])
        if duration_seconds is not None and seconds > duration_seconds + 3.0:
            continue
        if duration_seconds is not None:
            cue = dict(cue)
            cue["seconds"] = float(max(0.0, min(seconds, duration_seconds)))
        merged = False
        for existing in deduped:
            if abs(existing["seconds"] - cue["seconds"]) <= 0.5 and existing.get("source") == cue.get("source"):
                existing["weight"] = max(float(existing.get("weight", 1.0)), float(cue.get("weight", 1.0)))
                merged = True
                break
        if not merged:
            deduped.append(cue)
    return deduped


def extract_time_cues(sample: dict[str, Any], args, duration_seconds: float | None) -> list[dict[str, Any]]:
    cues = []
    if args.question_time_cues:
        cues.extend(extract_times_from_text(str(sample.get("question") or ""), "question", args.question_time_cue_weight))
    if args.oracle_reasoning_times:
        steps = sample.get("reasoning_steps") or []
        if isinstance(steps, str):
            steps = [steps]
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                for key in ("evidence", "inference"):
                    cues.extend(
                        extract_times_from_text(
                            str(step.get(key) or ""),
                            f"reasoning_steps[{i}].{key}",
                            args.time_cue_weight,
                        )
                    )
            else:
                cues.extend(extract_times_from_text(str(step), f"reasoning_steps[{i}]", args.time_cue_weight))
    return dedupe_time_cues(cues, duration_seconds)


def resolve_video_path(video_dir: str, sample_video_path: str) -> str:
    rel = str(sample_video_path).replace("data/", "")
    path = Path(rel)
    if path.is_absolute():
        return str(path)
    return str(Path(video_dir) / rel)


def probe_video_duration(video_path: str, fallback: float | None) -> tuple[float, str]:
    if fallback is not None and fallback > 0:
        return float(fallback), "sample_duration"
    try:
        import av

        with av.open(video_path) as container:
            if container.duration is not None:
                return float(container.duration / 1_000_000.0), "av_container_duration"
            stream_durations = []
            for stream in list(container.streams.video) + list(container.streams.audio):
                if stream.duration is not None and stream.time_base is not None:
                    stream_durations.append(float(stream.duration * stream.time_base))
            if stream_durations:
                return float(max(stream_durations)), "av_stream_duration"
    except Exception:
        pass
    return 0.0, "unknown_duration"


def make_uniform_segments(duration_seconds: float, segment_seconds: float) -> list[dict[str, Any]]:
    duration_seconds = max(float(duration_seconds), 0.001)
    segment_seconds = max(float(segment_seconds), 1.0)
    count = max(1, int(math.ceil(duration_seconds / segment_seconds)))
    segments = []
    for idx in range(count):
        start = min(duration_seconds, idx * segment_seconds)
        end = min(duration_seconds, (idx + 1) * segment_seconds)
        if end <= start:
            end = min(duration_seconds, start + 1.0)
        segments.append(
            {
                "segment_index": idx,
                "start": float(start),
                "end": float(end),
                "mid": float((start + end) / 2.0),
                "duration": float(max(0.0, end - start)),
            }
        )
    return segments


class RetrievalCache:
    def __init__(self):
        self.duration = {}
        self.audio = {}
        self.has_audio = {}

    def duration_for(self, sample: dict[str, Any], video_path: str) -> tuple[float, str]:
        if video_path in self.duration:
            return self.duration[video_path]
        parsed = parse_duration_seconds(sample.get("duration"))
        value = probe_video_duration(video_path, parsed)
        self.duration[video_path] = value
        return value

    def has_audio_for(self, video_path: str) -> bool:
        if video_path not in self.has_audio:
            self.has_audio[video_path] = check_video_has_audio(video_path)
        return bool(self.has_audio[video_path])

    def audio_energy_for(
        self,
        video_path: str,
        segments: list[dict[str, Any]],
        sample_rate: int,
        enabled: bool,
        has_audio: bool,
    ) -> tuple[list[float], list[float], str | None]:
        key = (video_path, tuple((round(s["start"], 3), round(s["end"], 3)) for s in segments), int(sample_rate), bool(enabled))
        if key in self.audio:
            return self.audio[key]
        if not enabled or not has_audio:
            result = ([0.0 for _ in segments], [0.0 for _ in segments], None if has_audio else "no_audio_stream")
            self.audio[key] = result
            return result
        try:
            raw = self._audio_energy_with_pyav(video_path, segments)
            norm = normalize_scores(raw)
            result = (raw, norm, None)
        except Exception as av_exc:
            result = self._audio_energy_with_librosa(video_path, segments, sample_rate, av_exc)
        self.audio[key] = result
        return result

    @staticmethod
    def _audio_energy_with_pyav(video_path: str, segments: list[dict[str, Any]]) -> list[float]:
        import av

        sumsq = np.zeros(len(segments), dtype=np.float64)
        counts = np.zeros(len(segments), dtype=np.int64)
        with av.open(video_path) as container:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if not audio_streams:
                return [0.0 for _ in segments]
            stream = audio_streams[0]
            cursor = 0.0
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.ndim == 2:
                    if arr.shape[0] <= 16 and arr.shape[0] <= arr.shape[1]:
                        arr = arr.mean(axis=0)
                    else:
                        arr = arr.mean(axis=1)
                arr = np.asarray(arr, dtype=np.float64).reshape(-1)
                if arr.size == 0:
                    continue
                sr = float(getattr(frame, "sample_rate", None) or getattr(stream, "sample_rate", None) or 1.0)
                if frame.pts is not None and frame.time_base is not None:
                    frame_start = float(frame.pts * frame.time_base)
                elif frame.time is not None:
                    frame_start = float(frame.time)
                else:
                    frame_start = cursor
                frame_end = frame_start + float(arr.size / max(sr, 1.0))
                cursor = frame_end
                for idx, segment in enumerate(segments):
                    overlap_start = max(float(segment["start"]), frame_start)
                    overlap_end = min(float(segment["end"]), frame_end)
                    if overlap_end <= overlap_start:
                        continue
                    local_start = int(max(0, math.floor((overlap_start - frame_start) * sr)))
                    local_end = int(min(arr.size, math.ceil((overlap_end - frame_start) * sr)))
                    if local_end <= local_start:
                        continue
                    chunk = arr[local_start:local_end]
                    sumsq[idx] += float(np.sum(chunk * chunk))
                    counts[idx] += int(chunk.size)
        raw = []
        for idx in range(len(segments)):
            if counts[idx] <= 0:
                raw.append(0.0)
            else:
                raw.append(float(math.sqrt(sumsq[idx] / counts[idx])))
        return raw

    @staticmethod
    def _audio_energy_with_librosa(
        video_path: str,
        segments: list[dict[str, Any]],
        sample_rate: int,
        av_exc: Exception,
    ) -> tuple[list[float], list[float], str | None]:
        try:
            import librosa

            audio, sr = librosa.load(video_path, sr=int(sample_rate), mono=True)
            raw = []
            for segment in segments:
                start = int(max(0.0, float(segment["start"])) * sr)
                end = int(max(float(segment["start"]), float(segment["end"])) * sr)
                chunk = audio[start:end]
                if chunk.size == 0:
                    raw.append(0.0)
                else:
                    chunk64 = np.asarray(chunk, dtype=np.float64)
                    raw.append(float(np.sqrt(np.mean(chunk64 * chunk64))))
            norm = normalize_scores(raw)
            result = (raw, norm, None)
        except Exception as exc:
            result = ([0.0 for _ in segments], [0.0 for _ in segments], f"pyav={av_exc!r}; librosa={exc!r}")
        return result


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return [0.0 for _ in values]
    return ((arr - lo) / (hi - lo)).astype(float).tolist()


def time_score_for_segment(segment: dict[str, Any], cues: list[dict[str, Any]], sigma: float) -> tuple[float, list[dict[str, Any]]]:
    sigma = max(float(sigma), 1.0)
    best = 0.0
    matched = []
    for cue in cues:
        t = float(cue["seconds"])
        start = float(segment["start"])
        end = float(segment["end"])
        if start <= t <= end:
            distance = 0.0
        elif t < start:
            distance = start - t
        else:
            distance = t - end
        proximity = math.exp(-distance / sigma)
        weighted = float(cue.get("weight", 1.0)) * proximity
        if weighted > best:
            best = weighted
        if distance <= sigma:
            matched.append(
                {
                    "seconds": t,
                    "time": format_time(t),
                    "distance_seconds": float(distance),
                    "source": cue.get("source"),
                    "weight": cue.get("weight"),
                    "text": cue.get("text"),
                }
            )
    return float(best), matched


def uniform_anchor_score(segment: dict[str, Any], duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    midpoint = duration_seconds / 2.0
    distance = abs(float(segment["mid"]) - midpoint)
    return float(max(0.0, 1.0 - distance / max(midpoint, 1.0)))


def select_evidence_segments(
    sample: dict[str, Any],
    sample_index: int,
    video_path: str,
    args,
    cache: RetrievalCache,
) -> dict[str, Any]:
    start_t = time.perf_counter()
    duration_seconds, duration_source = cache.duration_for(sample, video_path)
    base_segments = make_uniform_segments(duration_seconds, args.segment_seconds)
    has_audio = cache.has_audio_for(video_path)
    raw_energy, norm_energy, audio_error = cache.audio_energy_for(
        video_path,
        base_segments,
        args.audio_energy_sr,
        args.audio_energy,
        has_audio,
    )
    time_cues = extract_time_cues(sample, args, duration_seconds)

    scored = []
    for idx, segment in enumerate(base_segments):
        time_score, matched_cues = time_score_for_segment(segment, time_cues, args.time_sigma_seconds)
        audio_score = float(norm_energy[idx]) if idx < len(norm_energy) else 0.0
        anchor_score = uniform_anchor_score(segment, duration_seconds)
        total = (
            time_score
            + float(args.audio_energy_weight) * audio_score
            + float(args.uniform_anchor_weight) * anchor_score
        )
        row = {
            **segment,
            "time_score": float(time_score),
            "audio_energy_raw": float(raw_energy[idx]) if idx < len(raw_energy) else 0.0,
            "audio_energy_score": audio_score,
            "uniform_anchor_score": anchor_score,
            "score": float(total),
            "matched_time_cues": matched_cues,
        }
        scored.append(row)

    ranked = sorted(scored, key=lambda x: (-float(x["score"]), int(x["segment_index"])))
    selected_indices = []
    selected_intervals = []
    covered_cues = set()
    for row in ranked:
        if len(selected_indices) >= max(1, int(args.top_segments)):
            break
        cue_keys = {
            (round(float(cue.get("seconds", 0.0)), 1), cue.get("source"))
            for cue in row.get("matched_time_cues", [])
        }
        candidate_start = max(0.0, float(row["start"]) - max(0.0, float(args.segment_context_seconds)))
        candidate_end = min(duration_seconds, float(row["end"]) + max(0.0, float(args.segment_context_seconds)))
        redundant = False
        for existing_start, existing_end in selected_intervals:
            overlap = max(0.0, min(candidate_end, existing_end) - max(candidate_start, existing_start))
            shorter = max(1e-6, min(candidate_end - candidate_start, existing_end - existing_start))
            if overlap > 0.0 and not (cue_keys - covered_cues):
                redundant = True
                break
        if redundant:
            continue
        selected_indices.append(int(row["segment_index"]))
        selected_intervals.append((candidate_start, candidate_end))
        covered_cues.update(cue_keys)

    if args.min_coverage_segments > 0:
        coverage = min(int(args.min_coverage_segments), len(scored))
        if coverage > 0:
            for pos in np.linspace(0, len(scored) - 1, coverage).round().astype(int).tolist():
                if int(pos) not in selected_indices:
                    selected_indices.append(int(pos))

    selected = [scored[i] for i in sorted(set(selected_indices))]
    expanded = []
    context = max(0.0, float(args.segment_context_seconds))
    for segment in selected:
        expanded.append(
            {
                "source_segment_indices": [int(segment["segment_index"])],
                "start": float(max(0.0, segment["start"] - context)),
                "end": float(min(duration_seconds, segment["end"] + context)),
                "score": float(segment["score"]),
                "time_score": float(segment["time_score"]),
                "audio_energy_score": float(segment["audio_energy_score"]),
                "matched_time_cues": segment.get("matched_time_cues", []),
            }
        )
    expanded = merge_segments(expanded, duration_seconds, args.merge_gap_seconds)
    for idx, segment in enumerate(expanded):
        segment["segment_id"] = idx + 1
        segment["duration"] = float(max(0.0, segment["end"] - segment["start"]))
        segment["start_time"] = format_time(segment["start"])
        segment["end_time"] = format_time(segment["end"])

    top_debug = []
    for row in ranked[: min(12, len(ranked))]:
        top_debug.append(
            {
                "segment_index": int(row["segment_index"]),
                "start": row["start"],
                "end": row["end"],
                "start_time": format_time(row["start"]),
                "end_time": format_time(row["end"]),
                "score": row["score"],
                "time_score": row["time_score"],
                "audio_energy_score": row["audio_energy_score"],
                "matched_time_cues": row.get("matched_time_cues", []),
            }
        )

    selected_total_seconds = float(sum(max(0.0, s["end"] - s["start"]) for s in expanded))
    return {
        "sample_index": sample_index,
        "retrieval_version": "twopass_v0_oracle_time_audio_energy",
        "retrieval_mode": {
            "oracle_reasoning_times": bool(args.oracle_reasoning_times),
            "question_time_cues": bool(args.question_time_cues),
            "audio_energy": bool(args.audio_energy),
            "low_res_frame_features": "not_enabled_v0",
            "moe_route_signature": "not_enabled_v0",
            "cheap_partial_forward": "not_enabled_v0",
        },
        "duration_seconds": float(duration_seconds),
        "duration_source": duration_source,
        "segment_seconds": float(args.segment_seconds),
        "segment_context_seconds": float(args.segment_context_seconds),
        "top_segments": int(args.top_segments),
        "has_audio": bool(has_audio),
        "audio_energy_error": audio_error,
        "time_cues": time_cues,
        "selected_segments": expanded,
        "selected_total_seconds": selected_total_seconds,
        "coverage_fraction": float(selected_total_seconds / duration_seconds) if duration_seconds > 0 else None,
        "top_scored_segments": top_debug,
        "skim_latency_ms": float((time.perf_counter() - start_t) * 1000.0),
        "score_weights": {
            "time_cue_weight": float(args.time_cue_weight),
            "question_time_cue_weight": float(args.question_time_cue_weight),
            "audio_energy_weight": float(args.audio_energy_weight),
            "uniform_anchor_weight": float(args.uniform_anchor_weight),
            "time_sigma_seconds": float(args.time_sigma_seconds),
        },
    }


def merge_segments(segments: list[dict[str, Any]], duration_seconds: float, merge_gap_seconds: float) -> list[dict[str, Any]]:
    if not segments:
        return [{"source_segment_indices": [], "start": 0.0, "end": min(duration_seconds, 30.0), "score": 0.0}]
    ordered = sorted(segments, key=lambda x: (float(x["start"]), float(x["end"])))
    merged = []
    gap = max(0.0, float(merge_gap_seconds))
    for segment in ordered:
        if not merged or float(segment["start"]) > float(merged[-1]["end"]) + gap:
            merged.append(dict(segment))
            continue
        current = merged[-1]
        current["end"] = float(max(current["end"], segment["end"]))
        current["score"] = float(max(current.get("score", 0.0), segment.get("score", 0.0)))
        current["time_score"] = float(max(current.get("time_score", 0.0), segment.get("time_score", 0.0)))
        current["audio_energy_score"] = float(max(current.get("audio_energy_score", 0.0), segment.get("audio_energy_score", 0.0)))
        current.setdefault("source_segment_indices", []).extend(segment.get("source_segment_indices", []))
        current.setdefault("matched_time_cues", []).extend(segment.get("matched_time_cues", []))
    for segment in merged:
        segment["source_segment_indices"] = sorted(set(int(i) for i in segment.get("source_segment_indices", [])))
        seen = set()
        deduped_cues = []
        for cue in segment.get("matched_time_cues", []):
            key = (round(float(cue.get("seconds", 0.0)), 2), cue.get("source"), cue.get("text"))
            if key in seen:
                continue
            seen.add(key)
            deduped_cues.append(cue)
        segment["matched_time_cues"] = deduped_cues
    return merged


def build_twopass_prompt(sample: dict[str, Any], retrieval: dict[str, Any], include_times: bool) -> str:
    candidates, _ = get_candidates_and_answer(sample)
    candidates_text = "\n".join(candidates)
    prefix = ""
    if include_times:
        lines = [
            "The original video has been skimmed first. The following evidence windows are provided in order:",
        ]
        for segment in retrieval.get("selected_segments", []):
            lines.append(f"- Segment {segment['segment_id']}: original time {segment['start_time']} to {segment['end_time']}")
        prefix = "\n".join(lines) + "\n\n"
    return (
        f"{prefix}{sample['question']}\nOptions:\n{candidates_text}\n"
        "Answer with the option's letter from the given choices directly."
    )


def build_twopass_conversation(video_path: str, prompt: str, retrieval: dict[str, Any], args) -> list[dict[str, Any]]:
    content = []
    for segment in retrieval.get("selected_segments", []):
        video_item = {
            "type": "video",
            "video": video_path,
            "video_start": float(segment["start"]),
            "video_end": float(segment["end"]),
        }
        if args.video_fps is not None:
            video_item["fps"] = float(args.video_fps)
        if args.segment_max_frames is not None:
            video_item["max_frames"] = int(args.segment_max_frames)
        content.append(video_item)
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def tensor_shape(value):
    if value is None:
        return None
    try:
        return list(value.shape)
    except Exception:
        return None


def twopass_input_metadata(inputs, videos, audios, use_audio: bool, retrieval: dict[str, Any]) -> dict[str, Any]:
    decoded_video_frames = [int(video.shape[0]) for video in videos] if videos else []
    decoded_video_shapes = [tensor_shape(video) for video in videos] if videos else []
    ids = inputs.get("input_ids")
    input_ids = ids.detach().cpu().numpy()[0] if ids is not None else np.asarray([], dtype=np.int64)
    token_counts = {}
    # Token IDs are model-specific; fill exact modality counts later in evaluate_one when config is available.
    return {
        "use_audio_in_video": bool(use_audio),
        "input_ids_shape": tensor_shape(inputs.get("input_ids")),
        "attention_mask_shape": tensor_shape(inputs.get("attention_mask")),
        "pixel_values_videos_shape": tensor_shape(inputs.get("pixel_values_videos")),
        "input_features_shape": tensor_shape(inputs.get("input_features")),
        "video_grid_thw": inputs.get("video_grid_thw").detach().cpu().tolist() if inputs.get("video_grid_thw") is not None else None,
        "audio_feature_lengths": inputs.get("audio_feature_lengths").detach().cpu().tolist()
        if inputs.get("audio_feature_lengths") is not None
        else None,
        "decoded_video_frames": decoded_video_frames,
        "decoded_video_frames_total": int(sum(decoded_video_frames)),
        "decoded_video_shapes": decoded_video_shapes,
        "audio_items": len(audios) if audios else 0,
        "selected_segment_count": len(retrieval.get("selected_segments", [])),
        "selected_total_seconds": retrieval.get("selected_total_seconds"),
        "coverage_fraction": retrieval.get("coverage_fraction"),
        "input_token_count": int(input_ids.shape[0]) if input_ids.size else None,
        "token_modality_counts": token_counts,
    }


def add_token_modality_counts(meta: dict[str, Any], inputs, model):
    try:
        ids = inputs.input_ids[0].detach().cpu().numpy()
        token_ids = {
            "image": int(model.thinker.config.image_token_id),
            "video": int(model.thinker.config.video_token_id),
            "audio": int(model.thinker.config.audio_token_id),
        }
        counts = {"total": int(ids.shape[0])}
        for name, token_id in token_ids.items():
            counts[name] = int(np.sum(ids == token_id))
        counts["text_like"] = int(ids.shape[0] - counts.get("image", 0) - counts.get("video", 0) - counts.get("audio", 0))
        meta["token_modality_counts"] = counts
    except Exception as exc:
        meta["token_modality_count_error"] = repr(exc)


def evaluate_one(sample_index, sample, model, processor, args, output_dir: Path, cache: RetrievalCache):
    import torch

    from qwen_omni_utils import process_mm_info

    sample_video_path = sample.get("video_path") or sample.get("video")
    video_path = resolve_video_path(args.video_dir, sample_video_path)
    candidates, correct_answer = get_candidates_and_answer(sample)
    retrieval = select_evidence_segments(sample, sample_index, video_path, args, cache)
    prompt = build_twopass_prompt(sample, retrieval, args.include_segment_times_in_prompt)
    use_audio = (not args.no_audio_in_video) and bool(retrieval.get("has_audio"))
    conversation = build_twopass_conversation(video_path, prompt, retrieval, args)

    prep_start = time.perf_counter()
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    if videos:
        try:
            model.thinker.nframes = int(sum(video.shape[0] for video in videos))
        except Exception:
            pass
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio,
    )
    meta = twopass_input_metadata(inputs, videos, audios, use_audio, retrieval)
    add_token_modality_counts(meta, inputs, model)
    prepare_latency_ms = float((time.perf_counter() - prep_start) * 1000.0)
    inputs = move_inputs_to_model(inputs, model, args.device_map)

    gen_start = time.perf_counter()
    cont = None
    try:
        with torch.inference_mode():
            cont, _ = model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=args.max_new_tokens,
                thinker_eos_token_id=processor.tokenizer.eos_token_id,
                thinker_pad_token_id=processor.tokenizer.pad_token_id,
                thinker_do_sample=False,
                thinker_temperature=0,
                thinker_top_p=None,
                thinker_num_beams=1,
                use_cache=True,
                use_audio_in_video=use_audio,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generate_latency_ms = float((time.perf_counter() - gen_start) * 1000.0)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        resp_text = answers[0].strip() if answers and answers[0] else ""
        predicted_answer = extract_answer(resp_text)
        error = None
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generate_latency_ms = float((time.perf_counter() - gen_start) * 1000.0)
        resp_text = f"Error: {exc}"
        predicted_answer = None
        error = repr(exc)

    result = {
        "sample_index": sample_index,
        "video_id": sample.get("video_id"),
        "QA_id": sample.get("QA_id"),
        "video_path": sample_video_path,
        "question": sample["question"],
        "candidates": candidates,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": predicted_answer == correct_answer,
        "task_type": sample.get("task_type", "Unknown"),
        "question_type": sample.get("question_type"),
        "audio_type": sample.get("audio_type"),
        "video_type": sample.get("video_type"),
        "duration": sample.get("duration"),
        "model_response": resp_text,
        "error": error,
        "two_pass": {
            "selected_segment_count": len(retrieval.get("selected_segments", [])),
            "selected_total_seconds": retrieval.get("selected_total_seconds"),
            "coverage_fraction": retrieval.get("coverage_fraction"),
            "skim_latency_ms": retrieval.get("skim_latency_ms"),
            "prepare_latency_ms": prepare_latency_ms,
            "generate_latency_ms": generate_latency_ms,
            "input_meta": meta,
        },
    }
    retrieval_record = {
        **retrieval,
        "video_path": sample_video_path,
        "task_type": sample.get("task_type", "Unknown"),
        "question_type": sample.get("question_type"),
        "audio_type": sample.get("audio_type"),
        "video_type": sample.get("video_type"),
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": predicted_answer == correct_answer,
        "input_meta": meta,
        "prepare_latency_ms": prepare_latency_ms,
        "generate_latency_ms": generate_latency_ms,
        "error": error,
    }
    del inputs
    if cont is not None:
        del cont
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, retrieval_record


def read_done_indices(path: Path) -> set[int]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["sample_index"]))
            except Exception:
                continue
    return done


def aggregate_summary(output_dir: Path, args_dict: dict[str, Any]):
    predictions_path = output_dir / "predictions.jsonl"
    retrieval_path = output_dir / "retrieval.jsonl"
    predictions = []
    retrievals = []
    if predictions_path.exists():
        predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if retrieval_path.exists():
        retrievals = [json.loads(line) for line in retrieval_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    total = len(predictions)
    correct = sum(1 for r in predictions if r.get("is_correct"))
    by_task = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in predictions:
        task = r.get("question_type") or r.get("task_type") or "Unknown"
        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(bool(r.get("is_correct")))
    task_accuracy = {
        k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0, **v}
        for k, v in sorted(by_task.items())
    }

    def collect_from_retrievals(key):
        values = []
        for row in retrievals:
            value = row.get(key)
            if value is not None:
                values.append(float(value))
        return values

    selected_seconds = collect_from_retrievals("selected_total_seconds")
    coverage = collect_from_retrievals("coverage_fraction")
    skim_latency = collect_from_retrievals("skim_latency_ms")
    prepare_latency = collect_from_retrievals("prepare_latency_ms")
    generate_latency = collect_from_retrievals("generate_latency_ms")
    input_tokens = []
    video_tokens = []
    audio_tokens = []
    decoded_frames = []
    for row in retrievals:
        meta = row.get("input_meta") or {}
        counts = meta.get("token_modality_counts") or {}
        if meta.get("input_token_count") is not None:
            input_tokens.append(float(meta["input_token_count"]))
        if counts.get("video") is not None:
            video_tokens.append(float(counts["video"]))
        if counts.get("audio") is not None:
            audio_tokens.append(float(counts["audio"]))
        if meta.get("decoded_video_frames_total") is not None:
            decoded_frames.append(float(meta["decoded_video_frames_total"]))

    def stats(values):
        if not values:
            return {"mean": None, "median": None, "p95": None}
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": percentile(values, 95),
        }

    summary = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "args": args_dict,
        "files": {
            "predictions": str(predictions_path),
            "retrieval": str(retrieval_path),
        },
        "accuracy": {
            "overall_accuracy": correct / total if total else 0.0,
            "total_correct": correct,
            "total_samples": total,
            "question_type_accuracy": task_accuracy,
        },
        "retrieval": {
            "total_records": len(retrievals),
            "selected_seconds": stats(selected_seconds),
            "coverage_fraction": stats(coverage),
            "skim_latency_ms": stats(skim_latency),
            "oracle_reasoning_times": bool(args_dict.get("oracle_reasoning_times")),
            "audio_energy": bool(args_dict.get("audio_energy")),
        },
        "model_input": {
            "prepare_latency_ms": stats(prepare_latency),
            "generate_latency_ms": stats(generate_latency),
            "input_tokens": stats(input_tokens),
            "video_tokens": stats(video_tokens),
            "audio_tokens": stats(audio_tokens),
            "decoded_video_frames": stats(decoded_frames),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return summary


def retrieval_only_loop(selected, args, output_dir: Path, retrieval_path: Path):
    cache = RetrievalCache()
    for sample_index, sample in tqdm(selected, desc="OmniVideoBench two-pass retrieval"):
        sample_video_path = sample.get("video_path") or sample.get("video")
        video_path = resolve_video_path(args.video_dir, sample_video_path)
        retrieval = select_evidence_segments(sample, sample_index, video_path, args, cache)
        retrieval.update(
            {
                "video_path": sample_video_path,
                "question_type": sample.get("question_type"),
                "audio_type": sample.get("audio_type"),
                "video_type": sample.get("video_type"),
            }
        )
        append_jsonl(retrieval_path, retrieval)
        print(
            f"[retrieval {sample_index}] selected={len(retrieval['selected_segments'])} "
            f"coverage={retrieval.get('coverage_fraction')} "
            f"segments={[s['start_time'] + '-' + s['end_time'] for s in retrieval['selected_segments']]}",
            flush=True,
        )
    summary = aggregate_summary(output_dir, vars(args))
    print(f"[done] retrieval_only summary={output_dir / 'summary.json'}", flush=True)
    return summary


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.output_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"logs/omnivideobench_qwen3_twopass/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    predictions_path = output_dir / "predictions.jsonl"
    retrieval_path = output_dir / "retrieval.jsonl"
    errors_path = output_dir / "errors.jsonl"
    done = read_done_indices(predictions_path) if args.resume else set()

    data = load_json(Path(args.data_path))
    selected = list(enumerate(data))
    if args.start_index:
        selected = [(i, s) for i, s in selected if i >= args.start_index]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"[setup] output_dir={output_dir}", flush=True)
    print(
        f"[setup] two_pass top_segments={args.top_segments} segment_seconds={args.segment_seconds} "
        f"context={args.segment_context_seconds} segment_max_frames={args.segment_max_frames} "
        f"oracle_reasoning_times={args.oracle_reasoning_times} audio_energy={args.audio_energy}",
        flush=True,
    )
    print(f"[setup] loaded {len(data)} OmniVideoBench samples; scheduled {len(selected)}", flush=True)

    if args.retrieval_only:
        retrieval_only_loop(selected, args, output_dir, retrieval_path)
        return

    print("[setup] loading Qwen3-Omni without OmniZip/token-pruning wrapper", flush=True)
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    Qwen3OmniMoeForConditionalGeneration._tp_plan = (
        [] if Qwen3OmniMoeForConditionalGeneration._tp_plan is None else Qwen3OmniMoeForConditionalGeneration._tp_plan
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    ).eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    cache = RetrievalCache()
    for sample_index, sample in tqdm(selected, desc="OmniVideoBench Qwen3-Omni two-pass"):
        if sample_index in done:
            continue
        try:
            result, retrieval = evaluate_one(sample_index, sample, model, processor, args, output_dir, cache)
            append_jsonl(predictions_path, result)
            append_jsonl(retrieval_path, retrieval)
            if result.get("error"):
                append_jsonl(errors_path, result)
            print(
                f"[sample {sample_index}] qtype={result.get('question_type')} gt={result['correct_answer']} "
                f"pred={result['predicted_answer']} correct={result['is_correct']} "
                f"coverage={result['two_pass'].get('coverage_fraction')} "
                f"tokens={result['two_pass'].get('input_meta', {}).get('input_token_count')}",
                flush=True,
            )
        except Exception as exc:
            err = {
                "sample_index": sample_index,
                "video_path": sample.get("video_path") or sample.get("video"),
                "question_type": sample.get("question_type"),
                "error": repr(exc),
            }
            append_jsonl(errors_path, err)
            print(f"[sample {sample_index}] fatal_error={exc!r}", flush=True)
        if args.summary_every > 0 and (sample_index + 1) % args.summary_every == 0:
            aggregate_summary(output_dir, vars(args))

    summary = aggregate_summary(output_dir, vars(args))
    print(json.dumps(summary["accuracy"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] summary={output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
