#!/usr/bin/env python3
import argparse
import datetime as _dt
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text "
    "and speech. Please analyze the video carefully and select the most appropriate "
    "answer from the given options."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni on AVUTBenchmark with prefill/MoE profiling.")
    parser.add_argument("--model-path", default="models/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--data-path", default="eval_data/AVUTBenchmark/AV_Human_data.json")
    parser.add_argument("--video-dir", default="eval_data/AVUTBenchmark")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--route-dump", choices=("none", "head", "full"), default="head")
    parser.add_argument("--route-head-tokens", type=int, default=64)
    parser.add_argument("--torch-profile-samples", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-audio-in-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_jsonl(path: Path, obj: dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=json_default) + "\n")
        f.flush()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_candidates_and_answer(sample):
    candidates = []
    for opt in ["A", "B", "C", "D", "E", "F"]:
        key = f"option_{opt}"
        if key in sample and sample[key] is not None:
            candidates.append(f"{opt}. {sample[key]}")
    return candidates, str(sample.get("answer", "")).strip().upper()


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


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


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


class Qwen3ProfileCollector:
    def __init__(self, model, output_dir: Path, route_dump: str, route_head_tokens: int):
        self.model = model
        self.output_dir = output_dir
        self.route_dump = route_dump
        self.route_head_tokens = route_head_tokens
        self.current_sample = None
        self.current_call = None
        self.call_index = 0
        self.handles = []
        self.original_forwards = []

        text_config = model.thinker.config.text_config
        self.text_config = text_config
        self.num_layers = int(text_config.num_hidden_layers)
        self.hidden_size = int(text_config.hidden_size)
        self.num_heads = int(text_config.num_attention_heads)
        self.num_kv_heads = int(text_config.num_key_value_heads)
        self.head_dim = int(getattr(text_config, "head_dim", self.hidden_size // self.num_heads))
        self.num_experts = int(text_config.num_experts)
        self.top_k = int(text_config.num_experts_per_tok)
        self.moe_intermediate_size = int(text_config.moe_intermediate_size)
        self.shared_expert_intermediate_size = int(getattr(text_config, "shared_expert_intermediate_size", 0) or 0)
        self.dtype_bytes = 2
        if str(getattr(model, "dtype", "")).endswith("float32"):
            self.dtype_bytes = 4
        self.token_ids = {
            "image": int(model.thinker.config.image_token_id),
            "video": int(model.thinker.config.video_token_id),
            "audio": int(model.thinker.config.audio_token_id),
        }
        self.modality_codes = {"text": 0, "video": 1, "audio": 2, "image": 3, "decode": 4}
        self.modality_names = {v: k for k, v in self.modality_codes.items()}
        self._install_hooks()

    def _install_hooks(self):
        self._wrap_thinker_forward()
        for name, module in self.model.thinker.named_modules():
            if self._is_sparse_moe(module):
                layer_idx = self._parse_layer_idx(name)
                self._wrap_moe_forward(name, module, layer_idx)

    def _is_sparse_moe(self, module):
        cls_name = module.__class__.__name__.lower()
        return "sparsemoeblock" in cls_name and hasattr(module, "gate") and hasattr(module, "experts")

    @staticmethod
    def _parse_layer_idx(name):
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    def _cuda_elapsed_ms(self, fn):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            return result, float(start.elapsed_time(end))
        start_t = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start_t) * 1000.0

    def _wrap_thinker_forward(self):
        module = self.model.thinker
        original_forward = module.forward
        self.original_forwards.append((module, original_forward))

        def wrapped_forward(*args, **kwargs):
            self.call_index += 1
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            cache_position = kwargs.get("cache_position")
            past_key_values = kwargs.get("past_key_values")
            seq_len = None
            batch_size = None
            if input_ids is not None:
                batch_size, seq_len = list(input_ids.shape[:2])
            elif inputs_embeds is not None:
                batch_size, seq_len = list(inputs_embeds.shape[:2])
            cache_first = self._tensor_first(cache_position)
            cache_last = self._tensor_last(cache_position)
            phase = "decode"
            if seq_len is not None and seq_len > 1 and (cache_first in (None, 0) or past_key_values is None):
                phase = "prefill"
            call = {
                "call_index": self.call_index,
                "phase": phase,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "cache_position_first": cache_first,
                "cache_position_last": cache_last,
                "has_past_key_values": past_key_values is not None,
                "has_video_pixels": kwargs.get("pixel_values_videos") is not None,
                "has_audio_features": kwargs.get("input_features") is not None,
                "attention_impl": getattr(self.text_config, "_attn_implementation", None),
                "moe_pending": [],
                "moe_forward_time_ms_by_layer": {},
            }
            previous_call = self.current_call
            self.current_call = call
            try:
                result, elapsed_ms = self._cuda_elapsed_ms(lambda: original_forward(*args, **kwargs))
            finally:
                self.current_call = previous_call
            call["thinker_forward_latency_ms"] = elapsed_ms
            if phase == "prefill" and seq_len is not None:
                call["attention_flops_estimate"] = self.estimate_attention_flops(seq_len, batch_size or 1, seq_len)
                call["kv_cache_write_bytes_estimate"] = self.estimate_kv_cache_write_bytes(seq_len, batch_size or 1)
                self._process_moe_pending(call)
            else:
                call.pop("moe_pending", None)
            if self.current_sample is not None:
                self.current_sample.setdefault("forward_calls", []).append(call)
            return result

        module.forward = wrapped_forward

    @staticmethod
    def _tensor_first(value):
        if value is None:
            return None
        try:
            return int(value.flatten()[0].detach().cpu().item())
        except Exception:
            return None

    @staticmethod
    def _tensor_last(value):
        if value is None:
            return None
        try:
            return int(value.flatten()[-1].detach().cpu().item())
        except Exception:
            return None

    def _wrap_moe_forward(self, name, module, layer_idx):
        original_forward = module.forward
        self.original_forwards.append((module, original_forward))

        def wrapped_forward(*args, **kwargs):
            event_pair = None
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = original_forward(*args, **kwargs)
                end.record()
                elapsed_ms = None
                event_pair = (start, end)
            else:
                start_t = time.perf_counter()
                result = original_forward(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            call = self.current_call
            if call is not None and layer_idx is not None:
                if elapsed_ms is not None:
                    call["moe_forward_time_ms_by_layer"][str(layer_idx)] = elapsed_ms
                if call["phase"] == "prefill" and isinstance(result, tuple) and len(result) > 1:
                    hidden_states = args[0] if args else kwargs.get("hidden_states")
                    hidden_shape = list(hidden_states.shape) if hidden_states is not None else None
                    call["moe_pending"].append(
                        {
                            "layer_idx": layer_idx,
                            "module_name": name,
                            "hidden_shape": hidden_shape,
                            "router_logits": result[1].detach(),
                            "moe_forward_time_ms": elapsed_ms,
                            "event_pair": event_pair,
                        }
                    )
            return result

        module.forward = wrapped_forward

    def begin_sample(self, sample_index: int, sample: dict[str, Any], input_ids, inputs_meta: dict[str, Any]):
        ids = input_ids[0].detach().cpu().numpy()
        modalities = np.full(ids.shape[0], self.modality_codes["text"], dtype=np.int8)
        modalities[ids == self.token_ids["video"]] = self.modality_codes["video"]
        modalities[ids == self.token_ids["audio"]] = self.modality_codes["audio"]
        modalities[ids == self.token_ids["image"]] = self.modality_codes["image"]
        counts = Counter(self.modality_names[int(x)] for x in modalities)
        self.current_sample = {
            "sample_index": sample_index,
            "video_id": sample.get("video_id"),
            "QA_id": sample.get("QA_id"),
            "task_type": sample.get("task_type", "Unknown"),
            "video_path": sample.get("video_path"),
            "input_seq_len": int(ids.shape[0]),
            "token_modality_counts": dict(counts),
            "input_meta": inputs_meta,
            "forward_calls": [],
            "prefill_layers": [],
            "_modalities": modalities,
            "_route_arrays": {},
        }

    def end_sample(self):
        sample = self.current_sample
        self.current_sample = None
        if sample is None:
            return {}
        prefill_calls = [c for c in sample.get("forward_calls", []) if c.get("phase") == "prefill"]
        sample["prefill_summary"] = {
            "prefill_call_count": len(prefill_calls),
            "chunked_prefill_observed": len(prefill_calls) > 1
            or any(c.get("seq_len") not in (None, sample.get("input_seq_len")) for c in prefill_calls),
            "prefill_latency_ms": sum(c.get("thinker_forward_latency_ms", 0.0) for c in prefill_calls),
            "attention_flops_estimate": sum(c.get("attention_flops_estimate", 0) for c in prefill_calls),
            "moe_ffn_flops_estimate": sum(c.get("moe_ffn_flops_estimate", 0) for c in prefill_calls),
            "kv_cache_write_bytes_estimate": sum(c.get("kv_cache_write_bytes_estimate", 0) for c in prefill_calls),
            "flash_attention_requested": getattr(self.text_config, "_attn_implementation", None) == "flash_attention_2",
            "attention_impl": getattr(self.text_config, "_attn_implementation", None),
        }
        sample["system_profile"] = self.system_profile()
        route_arrays = sample.pop("_route_arrays", None)
        sample.pop("_modalities", None)
        if self.route_dump == "full" and route_arrays:
            route_dir = self.output_dir / "route_dumps"
            route_dir.mkdir(parents=True, exist_ok=True)
            route_path = route_dir / f"sample_{sample['sample_index']:06d}_routes.npz"
            np.savez_compressed(route_path, **{f"layer_{k}": v for k, v in route_arrays.items()})
            sample["full_route_dump"] = str(route_path)
        return sample

    def _process_moe_pending(self, call):
        pending = call.pop("moe_pending", [])
        layer_stats = []
        total_moe_ffn = 0
        modalities = None
        if self.current_sample is not None:
            modalities = self.current_sample.get("_modalities")
        for item in pending:
            if item.get("event_pair") is not None:
                start, end = item.pop("event_pair")
                item["moe_forward_time_ms"] = float(start.elapsed_time(end))
            call["moe_forward_time_ms_by_layer"][str(item["layer_idx"])] = item.get("moe_forward_time_ms")
            stat = self._summarize_router_logits(item, modalities)
            layer_stats.append(stat)
            total_moe_ffn += stat["moe_ffn_flops_estimate"]
            if self.current_sample is not None:
                self.current_sample["prefill_layers"].append(stat)
        call["moe_ffn_flops_estimate"] = int(total_moe_ffn)

    def _summarize_router_logits(self, item, modalities):
        router_logits = item["router_logits"]
        layer_idx = item["layer_idx"]
        with torch.no_grad():
            probs = torch.softmax(router_logits.float(), dim=-1)
            top_weights, selected = torch.topk(probs, self.top_k, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1)
            entropy_mean = float(entropy.mean().detach().cpu().item())
            entropy_norm = entropy_mean / math.log(self.num_experts)
            selected_cpu = selected.detach().cpu().to(torch.int16).numpy()
            top_weights_cpu = top_weights.detach().cpu().float().numpy()

        counts = np.bincount(selected_cpu.reshape(-1), minlength=self.num_experts).astype(np.int64)
        total_assignments = int(counts.sum())
        nonzero = int((counts > 0).sum())
        empty = int(self.num_experts - nonzero)
        mean_count = float(counts.mean())
        std_count = float(counts.std())
        max_count = int(counts.max()) if counts.size else 0
        min_nonzero = int(counts[counts > 0].min()) if nonzero else 0
        hot_expert = int(counts.argmax()) if counts.size else None
        top_experts = [
            {"expert_id": int(idx), "count": int(counts[idx]), "frequency": float(counts[idx] / max(total_assignments, 1))}
            for idx in counts.argsort()[::-1][: min(16, self.num_experts)]
        ]

        modality_counts = {}
        if modalities is not None and len(modalities) >= selected_cpu.shape[0]:
            used_modalities = modalities[: selected_cpu.shape[0]]
            for code, name in self.modality_names.items():
                mask = used_modalities == code
                if not np.any(mask):
                    continue
                m_counts = np.bincount(selected_cpu[mask].reshape(-1), minlength=self.num_experts).astype(np.int64)
                m_total = int(m_counts.sum())
                modality_counts[name] = {
                    "tokens": int(mask.sum()),
                    "activation_counts": m_counts.tolist(),
                    "top_experts": [
                        {
                            "expert_id": int(idx),
                            "count": int(m_counts[idx]),
                            "frequency": float(m_counts[idx] / max(m_total, 1)),
                        }
                        for idx in m_counts.argsort()[::-1][: min(8, self.num_experts)]
                    ],
                }

        stat = {
            "layer_idx": int(layer_idx),
            "module_name": item["module_name"],
            "tokens": int(selected_cpu.shape[0]),
            "top_k": self.top_k,
            "num_experts": self.num_experts,
            "activation_counts": counts.tolist(),
            "activation_frequency": (counts / max(total_assignments, 1)).astype(float).tolist(),
            "routing_entropy_mean": entropy_mean,
            "routing_entropy_normalized": float(entropy_norm),
            "load_balance": {
                "nonzero_experts": nonzero,
                "empty_experts": empty,
                "routed_expert_fraction": float(nonzero / self.num_experts),
                "null_expert_fraction": float(empty / self.num_experts),
                "shared_expert_fraction": 1.0 if self.shared_expert_intermediate_size > 0 else 0.0,
                "mean_expert_batch": mean_count,
                "std_expert_batch": std_count,
                "cv_expert_batch": float(std_count / mean_count) if mean_count else None,
                "max_expert_batch": max_count,
                "min_nonzero_expert_batch": min_nonzero,
                "hot_expert_id": hot_expert,
                "hot_to_mean_ratio": float(max_count / mean_count) if mean_count else None,
            },
            "top_experts_by_frequency": top_experts,
            "modality_expert_distribution": modality_counts,
            "moe_forward_time_ms": safe_float(item.get("moe_forward_time_ms")),
            "moe_ffn_flops_estimate": int(total_assignments * 6 * self.hidden_size * self.moe_intermediate_size),
        }
        if self.route_dump in ("head", "full"):
            head_n = min(self.route_head_tokens, selected_cpu.shape[0])
            stat["token_route_head"] = {
                "num_tokens": head_n,
                "selected_expert_ids": selected_cpu[:head_n].astype(np.uint8).tolist(),
                "routing_weights": top_weights_cpu[:head_n].round(6).tolist(),
            }
        if self.route_dump == "full" and self.current_sample is not None:
            self.current_sample["_route_arrays"][int(layer_idx)] = selected_cpu.astype(np.uint8)
        return stat

    def estimate_attention_flops(self, seq_len: int, batch_size: int, total_kv_len: int):
        h = self.hidden_size
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        proj = 2 * batch_size * seq_len * h * (q_out + kv_out + kv_out + h)
        if seq_len == total_kv_len:
            causal_pairs = seq_len * (seq_len + 1) // 2
            attention = 4 * batch_size * self.num_heads * causal_pairs * self.head_dim
        else:
            attention = 4 * batch_size * self.num_heads * seq_len * total_kv_len * self.head_dim
        return int(self.num_layers * (proj + attention))

    def estimate_kv_cache_write_bytes(self, seq_len: int, batch_size: int):
        per_layer = batch_size * seq_len * self.num_kv_heads * self.head_dim * 2 * self.dtype_bytes
        return int(self.num_layers * per_layer)

    def system_profile(self):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        return {
            "world_size": world_size,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "all_to_all": "not_applicable_single_gpu" if world_size == 1 else "inspect_torch_profile",
            "expert_parallel_dispatch_combine": "local_index_dispatch_combine_in_sparse_moe_block",
            "expert_weight_load_evict": "static_weights_loaded_once_no_eviction_observed",
            "cross_gpu_token_shuffle": "not_applicable_single_gpu" if world_size == 1 else "inspect_torch_profile",
        }


def build_prompt(sample):
    candidates, _ = get_candidates_and_answer(sample)
    candidates_text = "\n".join(candidates)
    return (
        f"{sample['question']}\nOptions:\n{candidates_text}\n"
        "Answer with the option's letter from the given choices directly."
    )


def build_conversation(video_path, prompt, video_fps=None, video_max_frames=None):
    video_item = {"type": "video", "video": video_path}
    if video_fps is not None:
        video_item["fps"] = video_fps
    if video_max_frames is not None:
        video_item["max_frames"] = video_max_frames
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [video_item, {"type": "text", "text": prompt}]},
    ]


def tensor_shape(value):
    if value is None:
        return None
    try:
        return list(value.shape)
    except Exception:
        return None


def input_metadata(inputs, videos, audios, use_audio):
    meta = {
        "use_audio_in_video": bool(use_audio),
        "input_ids_shape": tensor_shape(inputs.get("input_ids")),
        "attention_mask_shape": tensor_shape(inputs.get("attention_mask")),
        "pixel_values_videos_shape": tensor_shape(inputs.get("pixel_values_videos")),
        "input_features_shape": tensor_shape(inputs.get("input_features")),
        "video_grid_thw": inputs.get("video_grid_thw").detach().cpu().tolist() if inputs.get("video_grid_thw") is not None else None,
        "audio_feature_lengths": inputs.get("audio_feature_lengths").detach().cpu().tolist()
        if inputs.get("audio_feature_lengths") is not None
        else None,
        "decoded_video_frames": int(videos[0].shape[0]) if videos else 0,
        "decoded_video_shape": tensor_shape(videos[0]) if videos else None,
        "audio_items": len(audios) if audios else 0,
    }
    return meta


def move_inputs_to_model(inputs, model, device_map):
    if device_map == "auto":
        return inputs.to("cuda").to(model.dtype)
    return inputs.to(model.device).to(model.dtype)


def maybe_torch_profile(sample_index, enabled_count, output_dir):
    if sample_index >= enabled_count:
        return None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=activities, record_shapes=True, with_stack=False)


def save_torch_profile(prof, sample_index: int, output_dir: Path):
    if prof is None:
        return None
    profile_dir = output_dir / "torch_profiler"
    profile_dir.mkdir(parents=True, exist_ok=True)
    table_path = profile_dir / f"sample_{sample_index:06d}_key_averages.txt"
    trace_path = profile_dir / f"sample_{sample_index:06d}_trace.json"
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=80)
    table_path.write_text(table, encoding="utf-8")
    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception:
        trace_path = None
    comm_ops = []
    for evt in prof.key_averages():
        key = evt.key.lower()
        if any(token in key for token in ("all_to_all", "allgather", "all_gather", "nccl", "reduce_scatter")):
            comm_ops.append(
                {
                    "key": evt.key,
                    "cpu_time_total_us": safe_float(getattr(evt, "cpu_time_total", None)),
                    "cuda_time_total_us": safe_float(getattr(evt, "cuda_time_total", None)),
                    "count": int(getattr(evt, "count", 0)),
                }
            )
    return {
        "key_averages": str(table_path),
        "chrome_trace": str(trace_path) if trace_path else None,
        "communication_ops": comm_ops,
    }


def evaluate_one(sample_index, sample, model, processor, collector, args, output_dir: Path):
    from qwen_omni_utils import process_mm_info

    video_path = os.path.join(args.video_dir, sample["video_path"].replace("data/", ""))
    candidates, correct_answer = get_candidates_and_answer(sample)
    prompt = build_prompt(sample)
    use_audio = (not args.no_audio_in_video) and check_video_has_audio(video_path)
    conversation = build_conversation(video_path, prompt, args.video_fps, args.video_max_frames)

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
    meta = input_metadata(inputs, videos, audios, use_audio)
    inputs = move_inputs_to_model(inputs, model, args.device_map)
    collector.begin_sample(sample_index, sample, inputs.input_ids, meta)

    prof_ctx = maybe_torch_profile(sample_index, args.torch_profile_samples, output_dir)
    if prof_ctx is None:
        ctx = None
    else:
        ctx = prof_ctx.__enter__()

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
        generate_latency_ms = (time.perf_counter() - gen_start) * 1000.0
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
        generate_latency_ms = (time.perf_counter() - gen_start) * 1000.0
        resp_text = f"Error: {exc}"
        predicted_answer = None
        error = repr(exc)
    finally:
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)

    sample_profile = collector.end_sample()
    sample_profile["generate_latency_ms"] = generate_latency_ms
    torch_profile = save_torch_profile(ctx, sample_index, output_dir) if ctx is not None else None
    if torch_profile:
        sample_profile["torch_profile"] = torch_profile

    result = {
        "sample_index": sample_index,
        "video_id": sample.get("video_id"),
        "QA_id": sample.get("QA_id"),
        "video_path": sample["video_path"],
        "question": sample["question"],
        "candidates": candidates,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": predicted_answer == correct_answer,
        "task_type": sample.get("task_type", "Unknown"),
        "model_response": resp_text,
        "error": error,
    }
    del inputs
    if cont is not None:
        del cont
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, sample_profile


def read_done_indices(path: Path):
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
    profile_path = output_dir / "profile_samples.jsonl"
    predictions = []
    profiles = []
    if predictions_path.exists():
        predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if profile_path.exists():
        profiles = [json.loads(line) for line in profile_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    total = len(predictions)
    correct = sum(1 for r in predictions if r.get("is_correct"))
    by_task = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in predictions:
        t = r.get("task_type", "Unknown")
        by_task[t]["total"] += 1
        by_task[t]["correct"] += int(bool(r.get("is_correct")))
    task_accuracy = {
        k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0, **v}
        for k, v in sorted(by_task.items())
    }

    prefill_latencies = []
    attention_flops = []
    moe_flops = []
    kv_bytes = []
    flash_values = []
    chunked_values = []
    global_layer_counts = defaultdict(lambda: None)
    modality_counts = defaultdict(lambda: None)
    task_global_counts = defaultdict(lambda: np.zeros(0, dtype=np.int64))
    layer_entropy = defaultdict(list)
    layer_moe_time = defaultdict(list)

    for p in profiles:
        ps = p.get("prefill_summary", {})
        if ps.get("prefill_latency_ms") is not None:
            prefill_latencies.append(float(ps["prefill_latency_ms"]))
        if ps.get("attention_flops_estimate") is not None:
            attention_flops.append(int(ps["attention_flops_estimate"]))
        if ps.get("moe_ffn_flops_estimate") is not None:
            moe_flops.append(int(ps["moe_ffn_flops_estimate"]))
        if ps.get("kv_cache_write_bytes_estimate") is not None:
            kv_bytes.append(int(ps["kv_cache_write_bytes_estimate"]))
        flash_values.append(bool(ps.get("flash_attention_requested")))
        chunked_values.append(bool(ps.get("chunked_prefill_observed")))
        task = p.get("task_type", "Unknown")
        for layer in p.get("prefill_layers", []):
            idx = int(layer["layer_idx"])
            counts = np.array(layer["activation_counts"], dtype=np.int64)
            if global_layer_counts[idx] is None:
                global_layer_counts[idx] = np.zeros_like(counts)
            global_layer_counts[idx] += counts
            if task_global_counts[task].size == 0:
                task_global_counts[task] = np.zeros_like(counts)
            task_global_counts[task] += counts
            layer_entropy[idx].append(float(layer.get("routing_entropy_normalized", 0.0)))
            if layer.get("moe_forward_time_ms") is not None:
                layer_moe_time[idx].append(float(layer["moe_forward_time_ms"]))
            for modality, md in layer.get("modality_expert_distribution", {}).items():
                m_counts = np.array(md["activation_counts"], dtype=np.int64)
                if modality_counts[modality] is None:
                    modality_counts[modality] = np.zeros_like(m_counts)
                modality_counts[modality] += m_counts

    layer_summary = {}
    for idx, counts in sorted(global_layer_counts.items()):
        if counts is None:
            continue
        total_assignments = int(counts.sum())
        top = [
            {"expert_id": int(i), "count": int(counts[i]), "frequency": float(counts[i] / max(total_assignments, 1))}
            for i in counts.argsort()[::-1][:16]
        ]
        layer_summary[str(idx)] = {
            "total_assignments": total_assignments,
            "top_experts": top,
            "empty_experts_global": int((counts == 0).sum()),
            "mean_routing_entropy_normalized": statistics.mean(layer_entropy[idx]) if layer_entropy[idx] else None,
            "mean_moe_forward_time_ms": statistics.mean(layer_moe_time[idx]) if layer_moe_time[idx] else None,
            "p95_moe_forward_time_ms": percentile(layer_moe_time[idx], 95) if layer_moe_time[idx] else None,
        }

    modality_summary = {}
    for modality, counts in sorted(modality_counts.items()):
        if counts is None:
            continue
        total_assignments = int(counts.sum())
        modality_summary[modality] = {
            "total_assignments": total_assignments,
            "top_experts": [
                {"expert_id": int(i), "count": int(counts[i]), "frequency": float(counts[i] / max(total_assignments, 1))}
                for i in counts.argsort()[::-1][:16]
            ],
        }

    task_overlap = {}
    tasks = sorted(task_global_counts)
    top_sets = {}
    for task in tasks:
        counts = task_global_counts[task]
        if counts.size:
            top_sets[task] = set(int(i) for i in counts.argsort()[::-1][:32])
    for i, left in enumerate(tasks):
        for right in tasks[i + 1 :]:
            if left not in top_sets or right not in top_sets:
                continue
            union = top_sets[left] | top_sets[right]
            inter = top_sets[left] & top_sets[right]
            task_overlap[f"{left} :: {right}"] = float(len(inter) / len(union)) if union else None

    summary = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "args": args_dict,
        "files": {
            "predictions": str(predictions_path),
            "profile_samples": str(profile_path),
        },
        "accuracy": {
            "overall_accuracy": correct / total if total else 0.0,
            "total_correct": correct,
            "total_samples": total,
            "task_type_accuracy": task_accuracy,
        },
        "prefill": {
            "latency_ms_mean": statistics.mean(prefill_latencies) if prefill_latencies else None,
            "latency_ms_median": statistics.median(prefill_latencies) if prefill_latencies else None,
            "latency_ms_p95": percentile(prefill_latencies, 95),
            "attention_flops_estimate_mean": statistics.mean(attention_flops) if attention_flops else None,
            "moe_ffn_flops_estimate_mean": statistics.mean(moe_flops) if moe_flops else None,
            "kv_cache_write_bytes_estimate_mean": statistics.mean(kv_bytes) if kv_bytes else None,
            "flash_attention_requested_all_samples": all(flash_values) if flash_values else None,
            "chunked_prefill_observed_samples": sum(chunked_values),
        },
        "moe_routing": {
            "per_layer": layer_summary,
            "modality_expert_distribution": modality_summary,
            "task_expert_overlap_global_top32_jaccard": task_overlap,
        },
        "moe_system_overhead": {
            "single_gpu_note": "worker run uses one NVIDIA H20; all-to-all and cross-GPU token shuffle are not applicable unless torch profiler shows distributed communication ops.",
            "expert_dispatch_combine_time_source": "per-layer SparseMoeBlock CUDA event timing",
            "expert_batch_size_source": "per-layer activation_counts in profile_samples.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.output_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"logs/avut_qwen3_profile/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    predictions_path = output_dir / "predictions.jsonl"
    profile_path = output_dir / "profile_samples.jsonl"
    errors_path = output_dir / "errors.jsonl"
    done = read_done_indices(predictions_path) if args.resume else set()

    print(f"[setup] output_dir={output_dir}", flush=True)
    print(f"[setup] route_dump={args.route_dump} torch_profile_samples={args.torch_profile_samples}", flush=True)
    print("[setup] loading Qwen3-Omni without OmniZip/token-pruning wrapper", flush=True)

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

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
    collector = Qwen3ProfileCollector(model, output_dir, args.route_dump, args.route_head_tokens)

    data = load_json(Path(args.data_path))
    selected = list(enumerate(data))
    if args.start_index:
        selected = [(i, s) for i, s in selected if i >= args.start_index]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"[setup] loaded {len(data)} AVUT samples; scheduled {len(selected)}", flush=True)
    for sample_index, sample in tqdm(selected, desc="AVUT Qwen3-Omni"):
        if sample_index in done:
            continue
        try:
            result, profile = evaluate_one(sample_index, sample, model, processor, collector, args, output_dir)
            append_jsonl(predictions_path, result)
            append_jsonl(profile_path, profile)
            if result.get("error"):
                append_jsonl(errors_path, result)
            print(
                f"[sample {sample_index}] task={result['task_type']} gt={result['correct_answer']} "
                f"pred={result['predicted_answer']} correct={result['is_correct']} "
                f"prefill_ms={profile.get('prefill_summary', {}).get('prefill_latency_ms')}",
                flush=True,
            )
        except Exception as exc:
            err = {
                "sample_index": sample_index,
                "video_path": sample.get("video_path"),
                "task_type": sample.get("task_type", "Unknown"),
                "error": repr(exc),
            }
            append_jsonl(errors_path, err)
            print(f"[sample {sample_index}] fatal_error={exc!r}", flush=True)
        if (sample_index + 1) % 10 == 0:
            aggregate_summary(output_dir, vars(args))

    summary = aggregate_summary(output_dir, vars(args))
    print(json.dumps(summary["accuracy"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] summary={output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
