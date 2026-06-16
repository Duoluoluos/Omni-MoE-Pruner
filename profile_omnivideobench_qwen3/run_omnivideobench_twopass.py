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
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni on OmniVideoBench with GT-free MoE route-signature retrieval.")
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

    parser.add_argument("--retrieval-mode", choices=("route_signature", "oracle_time_audio"), default="route_signature")
    parser.add_argument("--segment-seconds", type=float, default=6.0)
    parser.add_argument("--segment-context-seconds", type=float, default=2.0)
    parser.add_argument("--top-segments", type=int, default=4)
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
    parser.add_argument("--oracle-reasoning-times", dest="oracle_reasoning_times", action="store_true", default=False)
    parser.add_argument("--no-oracle-reasoning-times", dest="oracle_reasoning_times", action="store_false")
    parser.add_argument("--question-time-cues", dest="question_time_cues", action="store_true", default=True)
    parser.add_argument("--no-question-time-cues", dest="question_time_cues", action="store_false")
    parser.add_argument("--include-segment-times-in-prompt", action="store_true", default=True)
    parser.add_argument("--no-segment-times-in-prompt", dest="include_segment_times_in_prompt", action="store_false")

    parser.add_argument("--skim-video-fps", type=float, default=0.5)
    parser.add_argument("--skim-segment-max-frames", type=int, default=12)
    parser.add_argument("--skim-max-new-tokens", type=int, default=1)
    parser.add_argument("--route-layer-start", type=int, default=None)
    parser.add_argument("--route-layer-end", type=int, default=None)
    parser.add_argument("--route-question-weight", type=float, default=1.5)
    parser.add_argument("--route-audio-weight", type=float, default=0.8)
    parser.add_argument("--route-margin-weight", type=float, default=0.9)
    parser.add_argument("--route-confidence-weight", type=float, default=0.5)
    parser.add_argument("--route-novelty-weight", type=float, default=1.0)
    parser.add_argument("--route-change-weight", type=float, default=0.7)
    parser.add_argument("--scene-change-weight", type=float, default=0.3)
    parser.add_argument("--redundancy-penalty-weight", type=float, default=0.7)
    parser.add_argument("--adaptive-compute-policy", choices=("metadata_only", "off"), default="metadata_only")
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
        self.visual = {}

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

    def visual_change_for(self, video_path: str, segments: list[dict[str, Any]]) -> tuple[list[float], str | None]:
        key = (video_path, tuple((round(s["start"], 3), round(s["end"], 3)) for s in segments))
        if key in self.visual:
            return self.visual[key]
        try:
            raw = self._visual_change_with_pyav(video_path, segments)
            result = (normalize_scores(raw), None)
        except Exception as exc:
            result = ([0.0 for _ in segments], repr(exc))
        self.visual[key] = result
        return result

    @staticmethod
    def _frame_feature(frame) -> np.ndarray:
        arr = frame.to_ndarray(format="rgb24")
        if arr.size == 0:
            return np.zeros(8 * 8 * 3, dtype=np.float32)
        h, w = arr.shape[:2]
        ys = np.linspace(0, max(0, h - 1), 8).round().astype(np.int64)
        xs = np.linspace(0, max(0, w - 1), 8).round().astype(np.int64)
        thumb = arr[ys][:, xs].astype(np.float32) / 255.0
        return thumb.reshape(-1)

    @classmethod
    def _visual_change_with_pyav(cls, video_path: str, segments: list[dict[str, Any]]) -> list[float]:
        import av

        target_times = [float(s["mid"]) for s in segments]
        features = [None for _ in segments]
        next_target = 0
        with av.open(video_path) as container:
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            if not video_streams:
                return [0.0 for _ in segments]
            stream = video_streams[0]
            for frame in container.decode(stream):
                if next_target >= len(target_times):
                    break
                t = frame.time
                if t is None:
                    if frame.pts is not None and frame.time_base is not None:
                        t = float(frame.pts * frame.time_base)
                    else:
                        continue
                while next_target < len(target_times) and float(t) >= target_times[next_target]:
                    features[next_target] = cls._frame_feature(frame)
                    next_target += 1
        last = None
        for idx, feature in enumerate(features):
            if feature is None:
                feature = last if last is not None else np.zeros(8 * 8 * 3, dtype=np.float32)
                features[idx] = feature
            else:
                last = feature
        raw = []
        for idx, feature in enumerate(features):
            prev_feature = features[idx - 1] if idx > 0 else feature
            next_feature = features[idx + 1] if idx + 1 < len(features) else feature
            raw.append(float(max(np.linalg.norm(feature - prev_feature), np.linalg.norm(feature - next_feature))))
        return raw

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


MULTIHOP_QUESTION_TYPES = {
    "temporal understanding",
    "reference reasoning",
    "spatial understanding",
    "counting",
    "causal reasoning",
    "ego reasoning",
}


AGGRESSIVE_QUESTION_TYPES = {
    "sentiment analysis",
    "relationship reasoning",
    "hypothetical reasoning",
    "background&music understanding",
}


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def vector_cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.size == 0 or bv.size == 0 or av.shape != bv.shape:
        return 0.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(av, bv) / denom, 0.0, 1.0))


def normalize_field(rows: list[dict[str, Any]], field: str, out_field: str):
    values = [safe_float(row.get(field)) for row in rows]
    norm = normalize_scores(values)
    for row, value in zip(rows, norm):
        row[out_field] = float(value)


def top_experts_from_vector(vector, limit: int = 8) -> list[dict[str, Any]]:
    if vector is None:
        return []
    arr = np.asarray(vector, dtype=np.float64)
    if arr.size == 0:
        return []
    total = float(arr.sum())
    top = []
    for idx in arr.argsort()[::-1][:limit]:
        value = float(arr[idx])
        if value <= 0:
            continue
        top.append({"expert_id": int(idx), "weight": value, "frequency": float(value / total) if total > 0 else 0.0})
    return top


def signature_from_vector(vector, limit: int = 8) -> str:
    return "-".join(str(item["expert_id"]) for item in top_experts_from_vector(vector, limit))


def task_compute_policy(question_type: str | None) -> dict[str, Any]:
    qtype = (question_type or "").strip().lower()
    if qtype in MULTIHOP_QUESTION_TYPES:
        aggressiveness = "conservative"
        middle_layers = "route_signature_selection_only"
        late_layers = "preserve_full_budget"
    elif qtype in AGGRESSIVE_QUESTION_TYPES:
        aggressiveness = "aggressive_allowed"
        middle_layers = "eligible_for_route_signature_token_merge_or_expert_budget_reduction"
        late_layers = "moderate_preserve"
    else:
        aggressiveness = "balanced"
        middle_layers = "eligible_after_quality_check"
        late_layers = "preserve_full_budget"
    return {
        "question_type": question_type,
        "aggressiveness": aggressiveness,
        "middle_layers": middle_layers,
        "late_layers": late_layers,
        "status": "metadata_only",
    }


class RouteSignatureCollector:
    def __init__(self, model, args):
        self.model = model
        self.current_record = None
        self.original_forwards = []
        text_config = model.thinker.config.text_config
        self.num_layers = int(text_config.num_hidden_layers)
        self.num_experts = int(text_config.num_experts)
        self.top_k = int(text_config.num_experts_per_tok)
        start = args.route_layer_start
        end = args.route_layer_end
        if start is None:
            start = self.num_layers // 3
        if end is None:
            end = max(start + 1, (self.num_layers * 2) // 3)
        start = max(0, min(int(start), self.num_layers - 1))
        end = max(start + 1, min(int(end), self.num_layers))
        self.target_layers = set(range(start, end))
        self.layer_range = [start, end]
        self.token_ids = {
            "image": int(model.thinker.config.image_token_id),
            "video": int(model.thinker.config.video_token_id),
            "audio": int(model.thinker.config.audio_token_id),
        }
        self.modality_codes = {"text": 0, "video": 1, "audio": 2, "image": 3}
        self.modality_names = {v: k for k, v in self.modality_codes.items()}
        self._install_hooks()

    def _install_hooks(self):
        for name, module in self.model.thinker.named_modules():
            if self._is_sparse_moe(module):
                layer_idx = self._parse_layer_idx(name)
                self._wrap_moe_forward(name, module, layer_idx)

    @staticmethod
    def _is_sparse_moe(module):
        cls_name = module.__class__.__name__.lower()
        return "sparsemoeblock" in cls_name and hasattr(module, "gate") and hasattr(module, "experts")

    @staticmethod
    def _parse_layer_idx(name):
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _wrap_moe_forward(self, name, module, layer_idx):
        original_forward = module.forward
        self.original_forwards.append((module, original_forward))

        def wrapped_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            record = self.current_record
            if record is None or layer_idx not in self.target_layers:
                return result
            if not (isinstance(result, tuple) and len(result) > 1):
                return result
            try:
                self._append_layer_summary(record, name, layer_idx, result[1])
            except Exception as exc:
                record.setdefault("collector_errors", []).append({"layer_idx": layer_idx, "error": repr(exc)})
            return result

        module.forward = wrapped_forward

    def begin_record(self, label: str, input_ids, meta: dict[str, Any] | None = None):
        ids = input_ids[0].detach().cpu().numpy()
        modalities = np.full(ids.shape[0], self.modality_codes["text"], dtype=np.int8)
        modalities[ids == self.token_ids["video"]] = self.modality_codes["video"]
        modalities[ids == self.token_ids["audio"]] = self.modality_codes["audio"]
        modalities[ids == self.token_ids["image"]] = self.modality_codes["image"]
        self.current_record = {
            "label": label,
            "input_token_count": int(ids.shape[0]),
            "token_modality_counts": {
                name: int(np.sum(modalities == code)) for name, code in self.modality_codes.items()
            },
            "target_layer_range": self.layer_range,
            "target_layers": sorted(self.target_layers),
            "meta": meta or {},
            "layers": [],
            "_modalities": modalities,
        }

    def end_record(self) -> dict[str, Any]:
        record = self.current_record or {}
        self.current_record = None
        if record:
            record["summary"] = self._summarize_record(record.get("layers", []))
            record.pop("_modalities", None)
        return record

    def _append_layer_summary(self, record: dict[str, Any], module_name: str, layer_idx: int, router_logits):
        import torch

        logits = router_logits.reshape(-1, router_logits.shape[-1])
        modalities = self.current_record_modalities(record)
        if modalities is None or logits.shape[0] != modalities.shape[0]:
            return
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            top_weights, selected = torch.topk(probs, self.top_k, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1)
            selected_cpu = selected.detach().cpu().to(torch.int16).numpy()
            top_weights_cpu = top_weights.detach().cpu().float().numpy()
            entropy_cpu = entropy.detach().cpu().float().numpy()
        if top_weights_cpu.shape[1] >= 2:
            margin_cpu = top_weights_cpu[:, 0] - top_weights_cpu[:, 1]
        else:
            margin_cpu = np.zeros(top_weights_cpu.shape[0], dtype=np.float32)

        modalities_summary = {}
        for code, name in self.modality_names.items():
            mask = modalities == code
            if not np.any(mask):
                continue
            counts = np.bincount(selected_cpu[mask].reshape(-1), minlength=self.num_experts).astype(np.float64)
            weight_sums = np.zeros(self.num_experts, dtype=np.float64)
            np.add.at(weight_sums, selected_cpu[mask].reshape(-1), top_weights_cpu[mask].reshape(-1))
            count_total = float(counts.sum())
            weight_total = float(weight_sums.sum())
            activation_frequency = (counts / count_total).tolist() if count_total > 0 else [0.0 for _ in range(self.num_experts)]
            weight_frequency = (
                (weight_sums / weight_total).tolist() if weight_total > 0 else [0.0 for _ in range(self.num_experts)]
            )
            modalities_summary[name] = {
                "tokens": int(mask.sum()),
                "activation_frequency": activation_frequency,
                "routing_weight_frequency": weight_frequency,
                "route_signature": signature_from_vector(weight_frequency),
                "top_experts": top_experts_from_vector(weight_frequency),
                "top1_top2_margin_mean": float(np.mean(margin_cpu[mask])),
                "routing_entropy_mean": float(np.mean(entropy_cpu[mask])),
                "routing_entropy_normalized": float(np.mean(entropy_cpu[mask]) / math.log(self.num_experts)),
            }
        record.setdefault("layers", []).append(
            {
                "layer_idx": int(layer_idx),
                "module_name": module_name,
                "modalities": modalities_summary,
            }
        )

    def current_record_modalities(self, record: dict[str, Any]):
        return record.get("_modalities")

    def _summarize_record(self, layers: list[dict[str, Any]]) -> dict[str, Any]:
        out = {"modalities": {}, "layer_count": len(layers)}
        for modality in ("text", "video", "audio", "image"):
            vectors = []
            margins = []
            entropies = []
            tokens = []
            for layer in layers:
                md = (layer.get("modalities") or {}).get(modality)
                if not md:
                    continue
                vectors.append(np.asarray(md.get("routing_weight_frequency") or [], dtype=np.float64))
                margins.append(safe_float(md.get("top1_top2_margin_mean")))
                entropies.append(safe_float(md.get("routing_entropy_normalized"), default=1.0))
                tokens.append(int(md.get("tokens") or 0))
            if not vectors:
                continue
            avg = np.mean(np.stack(vectors, axis=0), axis=0)
            out["modalities"][modality] = {
                "tokens_mean": float(statistics.mean(tokens)) if tokens else 0.0,
                "routing_weight_frequency": avg.astype(float).tolist(),
                "route_signature": signature_from_vector(avg),
                "top_experts": top_experts_from_vector(avg),
                "top1_top2_margin_mean": float(statistics.mean(margins)) if margins else 0.0,
                "routing_entropy_normalized": float(statistics.mean(entropies)) if entropies else 1.0,
            }
        return out


def record_vector(record: dict[str, Any] | None, modality: str):
    if not record:
        return None
    return (
        ((record.get("summary") or {}).get("modalities") or {})
        .get(modality, {})
        .get("routing_weight_frequency")
    )


def record_metric(record: dict[str, Any] | None, modality: str, key: str, default: float = 0.0) -> float:
    if not record:
        return default
    md = ((record.get("summary") or {}).get("modalities") or {}).get(modality, {})
    return safe_float(md.get(key), default=default)


def select_time_audio_segments(
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
        "retrieval_version": "twopass_v0_time_audio_optional_oracle",
        "retrieval_mode": {
            "mode": "oracle_time_audio",
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


def build_route_question_prompt(sample: dict[str, Any]) -> str:
    candidates, _ = get_candidates_and_answer(sample)
    candidates_text = "\n".join(candidates)
    return f"{sample['question']}\nOptions:\n{candidates_text}"


def build_route_question_conversation(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": build_route_question_prompt(sample)}]},
    ]


def build_route_segment_conversation(
    sample: dict[str, Any],
    video_path: str,
    segment: dict[str, Any],
    args,
) -> list[dict[str, Any]]:
    video_item = {
        "type": "video",
        "video": video_path,
        "video_start": float(segment["start"]),
        "video_end": float(segment["end"]),
    }
    if args.skim_video_fps is not None:
        video_item["fps"] = float(args.skim_video_fps)
    if args.skim_segment_max_frames is not None:
        video_item["max_frames"] = int(args.skim_segment_max_frames)
    prompt = (
        "Skim this short video segment for model-internal route signatures. "
        "Use the question only as routing context.\n"
        f"{build_route_question_prompt(sample)}"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [video_item, {"type": "text", "text": prompt}]},
    ]


def collect_route_record(
    conversation: list[dict[str, Any]],
    label: str,
    model,
    processor,
    route_collector: RouteSignatureCollector,
    args,
    use_audio: bool,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio,
    )
    record_meta = {
        "use_audio_in_video": bool(use_audio),
        "decoded_video_frames": [int(video.shape[0]) for video in videos] if videos else [],
        "decoded_video_shapes": [tensor_shape(video) for video in videos] if videos else [],
        "audio_items": len(audios) if audios else 0,
    }
    if meta:
        record_meta.update(meta)
    route_collector.begin_record(label, inputs.input_ids, record_meta)
    try:
        inputs = move_inputs_to_model(inputs, model, args.device_map)
        with torch.inference_mode():
            model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=max(1, int(args.skim_max_new_tokens)),
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
    except Exception as exc:
        record = route_collector.end_record()
        record["error"] = repr(exc)
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return record
    record = route_collector.end_record()
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def combined_route_vector(record: dict[str, Any], modalities=("video", "audio")):
    vectors = []
    weights = []
    for modality in modalities:
        vec = record_vector(record, modality)
        if vec is None:
            continue
        token_weight = record_metric(record, modality, "tokens_mean", default=0.0)
        vectors.append(np.asarray(vec, dtype=np.float64))
        weights.append(max(token_weight, 1.0))
    if not vectors:
        return None
    arr = np.average(np.stack(vectors, axis=0), axis=0, weights=np.asarray(weights, dtype=np.float64))
    total = float(arr.sum())
    return (arr / total).astype(float).tolist() if total > 0 else arr.astype(float).tolist()


def route_record_debug(record: dict[str, Any], keep_layers: bool = False) -> dict[str, Any]:
    if not record:
        return {}
    out = {
        "label": record.get("label"),
        "input_token_count": record.get("input_token_count"),
        "token_modality_counts": record.get("token_modality_counts"),
        "target_layer_range": record.get("target_layer_range"),
        "summary": record.get("summary"),
        "meta": record.get("meta"),
    }
    if record.get("collector_errors"):
        out["collector_errors"] = record.get("collector_errors")
    if record.get("error"):
        out["error"] = record.get("error")
    if keep_layers:
        out["layers"] = record.get("layers")
    return out


def build_route_scored_segments(
    base_segments: list[dict[str, Any]],
    segment_records: list[dict[str, Any]],
    question_record: dict[str, Any],
    raw_energy: list[float],
    norm_energy: list[float],
    visual_change: list[float],
    args,
) -> list[dict[str, Any]]:
    q_text = record_vector(question_record, "text")
    rows = []
    route_vectors = []
    for idx, segment in enumerate(base_segments):
        record = segment_records[idx]
        video_vec = record_vector(record, "video")
        audio_vec = record_vector(record, "audio")
        combined_vec = combined_route_vector(record)
        route_vectors.append(combined_vec)
        question_video = vector_cosine(q_text, video_vec)
        question_audio = vector_cosine(q_text, audio_vec)
        video_margin = record_metric(record, "video", "top1_top2_margin_mean")
        audio_margin = record_metric(record, "audio", "top1_top2_margin_mean")
        video_entropy = record_metric(record, "video", "routing_entropy_normalized", default=1.0)
        audio_entropy = record_metric(record, "audio", "routing_entropy_normalized", default=1.0)
        audio_score = float(norm_energy[idx]) if idx < len(norm_energy) else 0.0
        row = {
            **segment,
            "route_question_relevance_raw": float(max(question_video, 0.75 * question_audio)),
            "route_audio_relevance_raw": float(max(question_audio, 0.0)),
            "route_margin_raw": float(max(video_margin, audio_margin)),
            "route_confidence_raw": float(max(0.0, 1.0 - min(video_entropy, audio_entropy))),
            "audio_energy_raw": float(raw_energy[idx]) if idx < len(raw_energy) else 0.0,
            "audio_energy_score": audio_score,
            "visual_scene_change_score": float(visual_change[idx]) if idx < len(visual_change) else 0.0,
            "route_vector": combined_vec,
            "route_signature": signature_from_vector(combined_vec),
            "route_top_experts": top_experts_from_vector(combined_vec),
            "route_record": route_record_debug(record),
        }
        rows.append(row)

    for idx, row in enumerate(rows):
        sims = [
            vector_cosine(row.get("route_vector"), other)
            for j, other in enumerate(route_vectors)
            if j != idx and other is not None
        ]
        row["route_novelty_raw"] = float(1.0 - statistics.mean(sims)) if sims else 0.0
        neighbor_distances = []
        if idx > 0:
            neighbor_distances.append(1.0 - vector_cosine(row.get("route_vector"), route_vectors[idx - 1]))
        if idx + 1 < len(rows):
            neighbor_distances.append(1.0 - vector_cosine(row.get("route_vector"), route_vectors[idx + 1]))
        row["route_change_raw"] = float(max(neighbor_distances)) if neighbor_distances else 0.0

    normalize_field(rows, "route_margin_raw", "route_margin_score")
    normalize_field(rows, "route_confidence_raw", "route_confidence_score")
    normalize_field(rows, "route_novelty_raw", "route_novelty_score")
    normalize_field(rows, "route_change_raw", "route_change_score")
    for row in rows:
        row["route_question_relevance_score"] = safe_float(row.get("route_question_relevance_raw"))
        row["route_audio_relevance_score"] = safe_float(row.get("route_audio_relevance_raw"))
        row["score"] = float(
            args.route_question_weight * row["route_question_relevance_score"]
            + args.route_audio_weight * row["route_audio_relevance_score"]
            + args.audio_energy_weight * row["audio_energy_score"]
            + args.route_margin_weight * row["route_margin_score"]
            + args.route_confidence_weight * row["route_confidence_score"]
            + args.route_novelty_weight * row["route_novelty_score"]
            + args.route_change_weight * row["route_change_score"]
            + args.scene_change_weight * row["visual_scene_change_score"]
        )
    return rows


def temporal_overlap_fraction(row: dict[str, Any], selected: dict[str, Any], context: float, duration_seconds: float) -> float:
    a0 = max(0.0, float(row["start"]) - context)
    a1 = min(duration_seconds, float(row["end"]) + context)
    b0 = max(0.0, float(selected["start"]) - context)
    b1 = min(duration_seconds, float(selected["end"]) + context)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = max(1e-6, min(a1 - a0, b1 - b0))
    return float(overlap / shorter)


def choose_route_signature_rows(
    rows: list[dict[str, Any]],
    sample: dict[str, Any],
    args,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    target = max(1, int(args.top_segments))
    context = max(0.0, float(args.segment_context_seconds))
    selected: list[dict[str, Any]] = []
    qtype = (sample.get("question_type") or sample.get("task_type") or "").strip().lower()
    is_multihop = qtype in MULTIHOP_QUESTION_TYPES

    def redundancy_penalty(row: dict[str, Any]) -> float:
        if not selected:
            return 0.0
        route_sim = max(vector_cosine(row.get("route_vector"), s.get("route_vector")) for s in selected)
        temporal = max(temporal_overlap_fraction(row, s, context, duration_seconds) for s in selected)
        return float(args.redundancy_penalty_weight) * max(route_sim, temporal)

    def add_best(field: str, reason: str):
        if len(selected) >= target:
            return
        candidates = [row for row in rows if int(row["segment_index"]) not in {int(s["segment_index"]) for s in selected}]
        if not candidates:
            return
        best = max(candidates, key=lambda row: (safe_float(row.get(field)) - redundancy_penalty(row), safe_float(row.get("score"))))
        chosen = dict(best)
        chosen.setdefault("selection_reasons", []).append(reason)
        selected.append(chosen)

    add_best("route_question_relevance_score", "top_route_question_similarity")
    add_best("route_audio_relevance_score", "top_audio_route_similarity")
    if is_multihop:
        add_best("route_novelty_score", "top_route_novelty")
        if len(selected) < target:
            bridge_candidates = []
            selected_indices = {int(s["segment_index"]) for s in selected}
            for s in selected:
                for neighbor_idx in (int(s["segment_index"]) - 1, int(s["segment_index"]) + 1):
                    if 0 <= neighbor_idx < len(rows) and neighbor_idx not in selected_indices:
                        bridge_candidates.append(rows[neighbor_idx])
            if bridge_candidates:
                best = max(
                    bridge_candidates,
                    key=lambda row: (
                        safe_float(row.get("route_change_score"))
                        + safe_float(row.get("visual_scene_change_score"))
                        + safe_float(row.get("score"))
                    ),
                )
                chosen = dict(best)
                chosen.setdefault("selection_reasons", []).append("neighboring_bridge_segment")
                selected.append(chosen)
        if len(selected) < target:
            add_best("route_margin_score", "top_route_margin")
    else:
        add_best("route_novelty_score", "top_route_novelty")
        add_best("route_margin_score", "top_route_margin")

    while len(selected) < target:
        selected_indices = {int(s["segment_index"]) for s in selected}
        candidates = [row for row in rows if int(row["segment_index"]) not in selected_indices]
        if not candidates:
            break
        best = max(candidates, key=lambda row: safe_float(row.get("score")) - redundancy_penalty(row))
        chosen = dict(best)
        chosen.setdefault("selection_reasons", []).append("greedy_score_minus_redundancy")
        selected.append(chosen)
    return sorted(selected, key=lambda row: int(row["segment_index"]))


def select_route_signature_segments(
    sample: dict[str, Any],
    sample_index: int,
    video_path: str,
    args,
    cache: RetrievalCache,
    model,
    processor,
    route_collector: RouteSignatureCollector,
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
    visual_change, visual_error = cache.visual_change_for(video_path, base_segments)
    use_audio = (not args.no_audio_in_video) and bool(has_audio)

    question_record = collect_route_record(
        build_route_question_conversation(sample),
        "question_text",
        model,
        processor,
        route_collector,
        args,
        use_audio=False,
        meta={"sample_index": sample_index},
    )
    segment_records = []
    for segment in base_segments:
        record = collect_route_record(
            build_route_segment_conversation(sample, video_path, segment, args),
            f"segment_{int(segment['segment_index'])}",
            model,
            processor,
            route_collector,
            args,
            use_audio=use_audio,
            meta={
                "sample_index": sample_index,
                "segment_index": int(segment["segment_index"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "start_time": format_time(segment["start"]),
                "end_time": format_time(segment["end"]),
            },
        )
        segment_records.append(record)

    scored = build_route_scored_segments(
        base_segments,
        segment_records,
        question_record,
        raw_energy,
        norm_energy,
        visual_change,
        args,
    )
    selected = choose_route_signature_rows(scored, sample, args, duration_seconds)
    expanded = []
    context = max(0.0, float(args.segment_context_seconds))
    for segment in selected:
        expanded.append(
            {
                "source_segment_indices": [int(segment["segment_index"])],
                "start": float(max(0.0, segment["start"] - context)),
                "end": float(min(duration_seconds, segment["end"] + context)),
                "score": float(segment["score"]),
                "route_question_relevance_score": segment.get("route_question_relevance_score"),
                "route_audio_relevance_score": segment.get("route_audio_relevance_score"),
                "route_margin_score": segment.get("route_margin_score"),
                "route_confidence_score": segment.get("route_confidence_score"),
                "route_novelty_score": segment.get("route_novelty_score"),
                "route_change_score": segment.get("route_change_score"),
                "audio_energy_score": segment.get("audio_energy_score"),
                "visual_scene_change_score": segment.get("visual_scene_change_score"),
                "route_signature": segment.get("route_signature"),
                "route_top_experts": segment.get("route_top_experts"),
                "selection_reasons": segment.get("selection_reasons", []),
            }
        )
    expanded = merge_segments(expanded, duration_seconds, args.merge_gap_seconds)
    for idx, segment in enumerate(expanded):
        segment["segment_id"] = idx + 1
        segment["duration"] = float(max(0.0, segment["end"] - segment["start"]))
        segment["start_time"] = format_time(segment["start"])
        segment["end_time"] = format_time(segment["end"])

    top_debug = []
    ranked = sorted(scored, key=lambda x: (-float(x["score"]), int(x["segment_index"])))
    for row in ranked[: min(12, len(ranked))]:
        debug = {
            key: row.get(key)
            for key in (
                "segment_index",
                "start",
                "end",
                "score",
                "route_question_relevance_score",
                "route_audio_relevance_score",
                "route_margin_score",
                "route_confidence_score",
                "route_novelty_score",
                "route_change_score",
                "audio_energy_score",
                "visual_scene_change_score",
                "route_signature",
                "route_top_experts",
            )
        }
        debug["start_time"] = format_time(row["start"])
        debug["end_time"] = format_time(row["end"])
        top_debug.append(debug)

    selected_total_seconds = float(sum(max(0.0, s["end"] - s["start"]) for s in expanded))
    return {
        "sample_index": sample_index,
        "retrieval_version": "twopass_v1_moe_route_signature",
        "retrieval_mode": {
            "mode": "route_signature",
            "oracle_reasoning_times": False,
            "question_time_cues": False,
            "audio_energy": bool(args.audio_energy),
            "visual_scene_change": "pyav_sparse_midframe",
            "moe_route_signature": "enabled_live_low_cost_skim",
            "query_bm25_or_caption_retrieval": "disabled",
            "cheap_partial_forward": "full_thinker_generate_one_token_on_low_fps_segments",
        },
        "duration_seconds": float(duration_seconds),
        "duration_source": duration_source,
        "segment_seconds": float(args.segment_seconds),
        "segment_context_seconds": float(args.segment_context_seconds),
        "top_segments": int(args.top_segments),
        "has_audio": bool(has_audio),
        "audio_energy_error": audio_error,
        "visual_scene_change_error": visual_error,
        "route_question_record": route_record_debug(question_record),
        "selected_segments": expanded,
        "selected_total_seconds": selected_total_seconds,
        "coverage_fraction": float(selected_total_seconds / duration_seconds) if duration_seconds > 0 else None,
        "top_scored_segments": top_debug,
        "skim_latency_ms": float((time.perf_counter() - start_t) * 1000.0),
        "score_weights": {
            "route_question_weight": float(args.route_question_weight),
            "route_audio_weight": float(args.route_audio_weight),
            "audio_energy_weight": float(args.audio_energy_weight),
            "route_margin_weight": float(args.route_margin_weight),
            "route_confidence_weight": float(args.route_confidence_weight),
            "route_novelty_weight": float(args.route_novelty_weight),
            "route_change_weight": float(args.route_change_weight),
            "scene_change_weight": float(args.scene_change_weight),
            "redundancy_penalty_weight": float(args.redundancy_penalty_weight),
        },
        "adaptive_compute_policy": (
            task_compute_policy(sample.get("question_type") or sample.get("task_type"))
            if args.adaptive_compute_policy != "off"
            else {"status": "off"}
        ),
    }


def select_evidence_segments(
    sample: dict[str, Any],
    sample_index: int,
    video_path: str,
    args,
    cache: RetrievalCache,
    model=None,
    processor=None,
    route_collector: RouteSignatureCollector | None = None,
) -> dict[str, Any]:
    if args.retrieval_mode == "route_signature":
        if model is None or processor is None or route_collector is None:
            raise RuntimeError("route_signature retrieval requires model, processor, and RouteSignatureCollector")
        return select_route_signature_segments(sample, sample_index, video_path, args, cache, model, processor, route_collector)
    return select_time_audio_segments(sample, sample_index, video_path, args, cache)


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
        for key in (
            "route_question_relevance_score",
            "route_audio_relevance_score",
            "route_margin_score",
            "route_confidence_score",
            "route_novelty_score",
            "route_change_score",
            "visual_scene_change_score",
        ):
            if key in current or key in segment:
                current[key] = float(max(current.get(key, 0.0) or 0.0, segment.get(key, 0.0) or 0.0))
        current.setdefault("source_segment_indices", []).extend(segment.get("source_segment_indices", []))
        current.setdefault("matched_time_cues", []).extend(segment.get("matched_time_cues", []))
        current.setdefault("selection_reasons", []).extend(segment.get("selection_reasons", []))
        current.setdefault("merged_route_signatures", [])
        for sig in (current.get("route_signature"), segment.get("route_signature")):
            if sig:
                current["merged_route_signatures"].append(sig)
    for segment in merged:
        segment["source_segment_indices"] = sorted(set(int(i) for i in segment.get("source_segment_indices", [])))
        if segment.get("selection_reasons"):
            segment["selection_reasons"] = sorted(set(str(x) for x in segment.get("selection_reasons", [])))
        if segment.get("merged_route_signatures"):
            segment["merged_route_signatures"] = sorted(set(str(x) for x in segment.get("merged_route_signatures", [])))
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


def evaluate_one(
    sample_index,
    sample,
    model,
    processor,
    args,
    output_dir: Path,
    cache: RetrievalCache,
    route_collector: RouteSignatureCollector | None = None,
):
    import torch

    from qwen_omni_utils import process_mm_info

    sample_video_path = sample.get("video_path") or sample.get("video")
    video_path = resolve_video_path(args.video_dir, sample_video_path)
    candidates, correct_answer = get_candidates_and_answer(sample)
    retrieval = select_evidence_segments(
        sample,
        sample_index,
        video_path,
        args,
        cache,
        model=model,
        processor=processor,
        route_collector=route_collector,
    )
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
            "mode": args_dict.get("retrieval_mode"),
            "oracle_reasoning_times": bool(args_dict.get("oracle_reasoning_times")),
            "audio_energy": bool(args_dict.get("audio_energy")),
            "route_signature": args_dict.get("retrieval_mode") == "route_signature",
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


def retrieval_only_loop(
    selected,
    args,
    output_dir: Path,
    retrieval_path: Path,
    model=None,
    processor=None,
    route_collector: RouteSignatureCollector | None = None,
):
    cache = RetrievalCache()
    for sample_index, sample in tqdm(selected, desc="OmniVideoBench two-pass retrieval"):
        sample_video_path = sample.get("video_path") or sample.get("video")
        video_path = resolve_video_path(args.video_dir, sample_video_path)
        retrieval = select_evidence_segments(
            sample,
            sample_index,
            video_path,
            args,
            cache,
            model=model,
            processor=processor,
            route_collector=route_collector,
        )
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


def load_qwen3_omni(args):
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
    return model, processor


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
        f"[setup] retrieval_mode={args.retrieval_mode} top_segments={args.top_segments} segment_seconds={args.segment_seconds} "
        f"context={args.segment_context_seconds} segment_max_frames={args.segment_max_frames} "
        f"skim_fps={args.skim_video_fps} skim_max_frames={args.skim_segment_max_frames} "
        f"oracle_reasoning_times={args.oracle_reasoning_times} audio_energy={args.audio_energy}",
        flush=True,
    )
    print(f"[setup] loaded {len(data)} OmniVideoBench samples; scheduled {len(selected)}", flush=True)

    if args.retrieval_only and args.retrieval_mode != "route_signature":
        retrieval_only_loop(selected, args, output_dir, retrieval_path)
        return

    model, processor = load_qwen3_omni(args)
    route_collector = RouteSignatureCollector(model, args) if args.retrieval_mode == "route_signature" else None

    if args.retrieval_only:
        retrieval_only_loop(selected, args, output_dir, retrieval_path, model, processor, route_collector)
        return

    cache = RetrievalCache()
    for sample_index, sample in tqdm(selected, desc="OmniVideoBench Qwen3-Omni two-pass"):
        if sample_index in done:
            continue
        try:
            result, retrieval = evaluate_one(
                sample_index,
                sample,
                model,
                processor,
                args,
                output_dir,
                cache,
                route_collector,
            )
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
