#!/usr/bin/env python3
"""Run Qwen3-Omni on OmniVideoBench with the Fixed Top-M Mask budget installed.

Step 01 of the implementation plan: mask-only ablation, paired A/B against the
baseline profile run. Reuses the existing profile collector and prediction
output format from ``run_omnivideobench_profile.py`` so that downstream A/B
comparison works on the same JSONL schema.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_omnivideobench_profile import (  # type: ignore  # noqa: E402
    Qwen3ProfileCollector,
    aggregate_summary,
    append_jsonl,
    build_conversation,
    build_prompt,
    check_video_has_audio,
    extract_answer,
    get_candidates_and_answer,
    input_metadata,
    json_default,
    load_json,
    maybe_torch_profile,
    move_inputs_to_model,
    omnizip_config_from_args,
    read_done_indices,
    save_torch_profile,
)
from moe_expert_budget import ExpertBudgetConfig, FixedTopMMaskInstaller  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-Omni on OmniVideoBench with a Fixed Top-M MoE expert "
            "budget mask (step 01 of implementation_planner)."
        )
    )
    parser.add_argument("--model-path", default="models/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--data-path", default="eval_data/OmniVideoBench/data.json")
    parser.add_argument("--video-dir", default="eval_data/OmniVideoBench")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--route-dump", choices=("none", "head", "full"), default="none")
    parser.add_argument("--route-head-tokens", type=int, default=64)
    parser.add_argument("--trajectory-bins", type=int, default=16)
    parser.add_argument("--trajectory-top-experts", type=int, default=8)
    parser.add_argument("--torch-profile-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-audio-in-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use-omnizip", action="store_true")
    parser.add_argument("--omnizip-rho-audio", type=float, default=0.3)
    parser.add_argument("--omnizip-rho-video", type=float, default=0.6)
    parser.add_argument("--omnizip-g", type=int, default=3)
    parser.add_argument("--omnizip-contextual-ratio", type=float, default=0.05)

    parser.add_argument(
        "--budget-config",
        default=str(PROJECT_ROOT / "moe_expert_budget/configs/fixed_topm_default.json"),
        help="Path to a JSON config for ExpertBudgetConfig (mode fixed_topm_mask).",
    )
    parser.add_argument(
        "--budget-disable",
        action="store_true",
        help="Force the budget installer off (equivalent to plain baseline run).",
    )
    parser.add_argument("--budget-top-m", type=int, default=None)
    parser.add_argument("--budget-layer-start", type=int, default=None)
    parser.add_argument("--budget-layer-end", type=int, default=None)
    parser.add_argument("--budget-renormalize", choices=("true", "false"), default=None)
    return parser.parse_args()


def load_budget_config(args: argparse.Namespace) -> ExpertBudgetConfig:
    if args.budget_disable:
        return ExpertBudgetConfig.disabled()
    cfg_path = Path(args.budget_config)
    if cfg_path.exists():
        cfg = ExpertBudgetConfig.from_json(cfg_path)
    else:
        print(f"[expert_budget] config {cfg_path} not found; using dataclass defaults.", flush=True)
        cfg = ExpertBudgetConfig()
    if args.budget_top_m is not None:
        cfg.top_m = int(args.budget_top_m)
    if args.budget_layer_start is not None:
        cfg.layer_start = int(args.budget_layer_start)
    if args.budget_layer_end is not None:
        cfg.layer_end = int(args.budget_layer_end)
    if args.budget_renormalize is not None:
        cfg.renormalize_weights = args.budget_renormalize == "true"
    return cfg


def evaluate_one(
    sample_index: int,
    sample: dict[str, Any],
    model,
    processor,
    collector: Qwen3ProfileCollector,
    budget: FixedTopMMaskInstaller,
    args: argparse.Namespace,
    output_dir: Path,
):
    from qwen_omni_utils import process_mm_info

    sample_video_path = sample.get("video_path") or sample.get("video")
    video_path = os.path.join(args.video_dir, sample_video_path.replace("data/", ""))
    candidates, correct_answer = get_candidates_and_answer(sample)
    prompt = build_prompt(sample)
    use_audio = (not args.no_audio_in_video) and check_video_has_audio(video_path)
    conversation = build_conversation(video_path, prompt, args.video_fps, args.video_max_frames)

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    if videos:
        try:
            model.thinker.nframes = videos[0].shape[0]
        except Exception:
            pass
    if args.use_omnizip and hasattr(model, "thinker"):
        model.thinker.omnizip_config = omnizip_config_from_args(args)
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
    if budget.enabled:
        budget.begin_sample(sample_index, inputs.input_ids)

    prof_ctx = maybe_torch_profile(sample_index, args.torch_profile_samples, output_dir)
    ctx = prof_ctx.__enter__() if prof_ctx is not None else None

    gen_start = time.perf_counter()
    cont = None
    error = None
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
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        resp_text = answers[0].strip() if answers and answers[0] else ""
        predicted_answer = extract_answer(resp_text)
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
    if budget.enabled:
        sample_profile["expert_budget"] = budget.end_sample()
    else:
        sample_profile["expert_budget"] = {"enabled": False}
    torch_profile = save_torch_profile(ctx, sample_index, output_dir) if ctx is not None else None
    if torch_profile:
        sample_profile["torch_profile"] = torch_profile

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
        "expert_budget_summary": sample_profile.get("expert_budget"),
    }
    del inputs
    if cont is not None:
        del cont
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, sample_profile


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.output_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"logs/omnivideobench_qwen3_budget_topm/{ts}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    budget_config = load_budget_config(args)
    (output_dir / "budget_config.json").write_text(
        json.dumps(budget_config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    predictions_path = output_dir / "predictions.jsonl"
    profile_path = output_dir / "profile_samples.jsonl"
    errors_path = output_dir / "errors.jsonl"
    done = read_done_indices(predictions_path) if args.resume else set()

    print(f"[setup] output_dir={output_dir}", flush=True)
    print(f"[setup] budget_config={budget_config.to_dict()}", flush=True)

    if args.use_omnizip:
        from omnizip.qwen3_omni.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
        from transformers import Qwen3OmniMoeProcessor
    else:
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
    if args.use_omnizip and hasattr(model, "thinker"):
        model.thinker.omnizip_config = omnizip_config_from_args(args)
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    # IMPORTANT: budget installer must wrap the SparseMoeBlock BEFORE the profile
    # collector wraps it, so the profile timing+router_logits sees the budget result.
    budget = FixedTopMMaskInstaller(model, budget_config)
    if budget.enabled:
        budget.install()
    collector = Qwen3ProfileCollector(
        model,
        output_dir,
        args.route_dump,
        args.route_head_tokens,
        args.trajectory_bins,
        args.trajectory_top_experts,
    )

    data = load_json(Path(args.data_path))
    selected = list(enumerate(data))
    if args.start_index:
        selected = [(i, s) for i, s in selected if i >= args.start_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(f"[setup] loaded {len(data)} samples; scheduled {len(selected)}", flush=True)

    for sample_index, sample in tqdm(selected, desc="OmniVideoBench Qwen3-Omni budget topM"):
        if sample_index in done:
            continue
        try:
            result, profile = evaluate_one(
                sample_index, sample, model, processor, collector, budget, args, output_dir
            )
            append_jsonl(predictions_path, result)
            append_jsonl(profile_path, profile)
            if result.get("error"):
                append_jsonl(errors_path, result)
            eb = result.get("expert_budget_summary") or {}
            print(
                f"[sample {sample_index}] task={result['task_type']} gt={result['correct_answer']} "
                f"pred={result['predicted_answer']} correct={result['is_correct']} "
                f"prefill_ms={profile.get('prefill_summary', {}).get('prefill_latency_ms')} "
                f"avg_active={eb.get('avg_active_experts_per_token_effective')}",
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
