#!/usr/bin/env python3
"""TODO 2A: bounded, fixed-coverage density diagnostic, not a reference rerun.

Reuse the original CG controller/generation path without editing that runner.
Selection and runtime inputs are label-independent. Timing has no NEW profiling
hooks; execution accounting uses separate, output-checked generation passes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import traceback

import numpy as np
import torch

import run_mb_new2_cgskip_qwen3omni as cg

ROOT = cg.REPO
P1 = ROOT / "research_paper/iclr2027/evidence/runs/pm1_workload_decomposition_worker4193170_20260828/raw_results.jsonl"
DENSITIES = (0.25, 0.5, 1.0)
TAU = {"tau_reduce": 0.08475, "tau_boost": 0.04829}
POLICY = dict(base_budget=8, reduce_budget=4, boost_budget=8,
              reduce_percentile=60, boost_percentile=0)
INPUT_CONTRACT_VERSION = 2


def processor_kwargs(fps):
    # Transformers 4.57.3 does not fall back to flat modality arguments when
    # a nested modality dict is present. Keep each option in its own dict.
    return dict(return_tensors="pt", text_kwargs={"padding": True},
                audio_kwargs={"truncation": False},
                videos_kwargs={"fps": fps, "do_resize": False, "use_audio_in_video": True})


def check_prepared_input(processor, inputs, audio_samples, frame_count, fps):
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )
    ids = inputs["input_ids"][0]
    assert inputs["input_ids"].shape[0] == 1 and bool(inputs["attention_mask"].all())
    assert processor.tokenizer.convert_tokens_to_ids(processor.audio_token) == cg.AUDIO_TOKEN_ID
    assert processor.tokenizer.convert_tokens_to_ids(processor.video_token) == cg.VIDEO_TOKEN_ID
    lengths = inputs["feature_attention_mask"].sum(-1)
    assert lengths.numel() == 1 and abs(int(lengths[0]) - audio_samples / 160) < 3, "audio truncated"
    expected_audio = int(_get_feat_extract_output_lengths(lengths).sum())
    counts = {"video": int((ids == cg.VIDEO_TOKEN_ID).sum()),
              "audio": int((ids == cg.AUDIO_TOKEN_ID).sum())}
    assert counts["audio"] == expected_audio > 0, "audio placeholders absent or mismatched"
    counts["text_and_markers"] = len(ids) - sum(counts.values())
    grid = inputs["video_grid_thw"][0].tolist()
    vp = processor.video_processor
    assert grid[0] * vp.temporal_patch_size == frame_count, "video resampled unexpectedly"
    assert counts["video"] == int(np.prod(grid)) // vp.merge_size ** 2 > 0
    assert np.isclose(float(inputs["video_second_per_grid"][0]), vp.temporal_patch_size / fps)
    assert torch.isfinite(inputs["input_features"]).all()
    assert torch.isfinite(inputs["pixel_values_videos"]).all()
    return {"version": INPUT_CONTRACT_VERSION, "passed": True,
            "expected_audio_tokens": expected_audio, "actual_audio_tokens": counts["audio"],
            "feature_length": int(lengths[0]), "token_counts": counts}


class EmbeddingAudit:
    """Prove audio/video encoder rows reach the language backbone, outside timing."""
    def __init__(self, model, input_ids):
        self.thinker = model.thinker
        self.ids = input_ids
        self.features = {}
        self.calls = 0
        self.counts = {}

    @staticmethod
    def check_rows(embeds, mask, features):
        rows = embeds[mask]
        expected = features.to(device=embeds.device, dtype=embeds.dtype)
        assert rows.shape == expected.shape and rows.shape[0] > 0, "encoder/placeholder shape mismatch"
        assert torch.equal(rows, expected), "encoder features did not reach language input"
        return rows.shape[0]

    def __enter__(self):
        self.originals = {name: getattr(self.thinker, name) for name in
                          ("get_audio_features", "get_video_features")}
        for modality, name in (("audio", "get_audio_features"), ("video", "get_video_features")):
            def capture(*args, modality=modality, name=name, **kwargs):
                output = self.originals[name](*args, **kwargs)
                self.features[modality] = output if modality == "audio" else output[0]
                return output
            setattr(self.thinker, name, capture)

        def language_pre(_module, _args, kwargs):
            embeds = kwargs["inputs_embeds"]
            if embeds.shape[1] == 1:
                return
            self.calls += 1
            assert embeds.shape[:2] == self.ids.shape
            for modality, token in (("audio", cg.AUDIO_TOKEN_ID), ("video", cg.VIDEO_TOKEN_ID)):
                self.counts[modality] = self.check_rows(embeds, self.ids == token, self.features[modality])
            assert torch.isfinite(kwargs["position_ids"]).all()
        self.handle = self.thinker.model.register_forward_pre_hook(language_pre, with_kwargs=True)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        for name, original in self.originals.items():
            setattr(self.thinker, name, original)
        self.features.clear()

    def result(self):
        assert self.calls == 1 and self.counts["audio"] > 0 and self.counts["video"] > 0
        return {"passed": True, "prefill_calls": self.calls, "inserted_rows": self.counts}


def select_inputs():
    # Only these allowlisted fields leave the workload archive.
    unique = {}
    for row in cg.load_jsonl(P1):
        unique.setdefault(row["video"], {k: row[k] for k in
                          ("video", "sample_index", "input_sequence_length")})
    ordered = sorted(unique.values(), key=lambda r: (r["input_sequence_length"], r["video"]))
    rng = np.random.default_rng(cg.SEED)
    selected = []
    for q, indices in enumerate(np.array_split(np.arange(len(ordered)), 4)):
        for i in sorted(rng.choice(indices, size=2, replace=False).tolist()):
            selected.append(dict(ordered[i], length_quartile=q + 1))
    return selected


def frame_indices(n, density):
    count = max(2, 2 * round(n * density / 2))
    idx = torch.linspace(0, n - 1, count).round().long()
    assert len(idx.unique()) == count and idx[0] == 0 and idx[-1] == n - 1
    return idx


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def response_row(res):
    return {k: v for k, v in res.items() if k != "first_logits"}


class WorkCounter:
    """Count real expert input rows on separate diagnostic passes only."""
    def __init__(self, model, controller, input_ids):
        self.model, self.controller = model, controller
        ids = input_ids[0].detach().cpu()
        self.masks = {"video": ids == cg.VIDEO_TOKEN_ID,
                      "audio": ids == cg.AUDIO_TOKEN_ID}
        self.masks["text_and_markers"] = ~(self.masks["video"] | self.masks["audio"])
        self.handles = []
        self.actual = Counter()
        self.assigned = Counter()
        self.histogram = Counter()
        self.visits = Counter()
        self.phase, self.layer = None, None

    def __enter__(self):
        for name, module in self.model.thinker.named_modules():
            if "sparsemoeblock" not in module.__class__.__name__.lower():
                continue
            layer = cg.CGSkipIntervention._layer_index(name)
            if layer is None:
                continue
            def block_pre(_module, args, layer=layer):
                n = args[0].shape[0] * args[0].shape[1]
                self.layer = layer
                self.phase = "prefill" if args[0].shape[1] > 1 else "decode"
                self.visits[(self.phase, layer)] += 1
                if self.controller._arm == "native" or not 12 <= layer < 32:
                    self.add_assigned(torch.full((n,), 8, dtype=torch.long))
            self.handles.append(module.register_forward_pre_hook(block_pre))
            for expert in module.experts:
                def expert_pre(_expert, args, layer=layer):
                    self.actual[(self.phase, layer)] += args[0].numel() // args[0].shape[-1]
                self.handles.append(expert.register_forward_pre_hook(expert_pre))
        self.original_budget = self.controller._per_token_budget
        def budget(weights, count):
            out = self.original_budget(weights, count)
            self.add_assigned(out.detach().cpu())
            return out
        self.controller._per_token_budget = budget
        return self

    def add_assigned(self, budgets):
        self.histogram.update({(self.phase, int(k)): int(v) for k, v in
                               zip(*torch.unique(budgets, return_counts=True))})
        if self.phase == "prefill":
            assert len(budgets) == len(next(iter(self.masks.values())))
            for modality, mask in self.masks.items():
                self.assigned[(self.phase, self.layer, modality)] += int(budgets[mask].sum())
        else:
            self.assigned[(self.phase, self.layer, "generated_text")] += int(budgets.sum())

    def __exit__(self, *args):
        self.controller._per_token_budget = self.original_budget
        for handle in self.handles:
            handle.remove()

    def result(self):
        assigned_totals = Counter()
        for (phase, layer, _), value in self.assigned.items():
            assigned_totals[(phase, layer)] += value
        assert self.actual == assigned_totals, "actual dispatch / assigned ledger mismatch"
        assert all(self.visits[("prefill", layer)] == 1 for layer in range(48))
        assert self.assigned[("prefill", 0, "audio")] > 0, "audio never reached MoE"
        per_modality = Counter()
        for (phase, _, modality), value in self.assigned.items():
            per_modality[(phase, modality)] += value
        return {
            "actual_routed_pairs": {f"{phase}/layer_{layer}": value
                                    for (phase, layer), value in sorted(self.actual.items())},
            "routed_pairs_by_phase_modality": {f"{p}/{m}": v for (p, m), v in sorted(per_modality.items())},
            "budget_histogram": {f"{p}/budget_{b}": v for (p, b), v in sorted(self.histogram.items())},
            "actual_total_pairs": sum(self.actual.values()),
            "actual_prefill_pairs": sum(v for (p, _), v in self.actual.items() if p == "prefill"),
            "actual_decode_pairs": sum(v for (p, _), v in self.actual.items() if p == "decode"),
            "dispatch_ledger_passed": True,
        }


def evaluate_input(model, processor, controller, inputs, native_first):
    order = ["native", "cgskip"] if native_first else ["cgskip", "native"]
    def generate(arm):
        with controller.arm(arm):
            return cg.generate_once(model, processor, inputs, 4)
    # Same prepared tensors reused across every paired pass. Exclude warm-ups.
    for arm in order:
        generate(arm)
    timed = defaultdict(list)
    results = {}
    for repeat_order in (order, list(reversed(order))):
        for arm in repeat_order:
            result = generate(arm)
            timed[arm].append(response_row(result))
            if arm in results:
                assert result["output_text"] == results[arm]["output_text"], "timing repeat output changed"
            results[arm] = result
    for arm in order:
        with WorkCounter(model, controller, inputs["input_ids"]) as counter, \
                EmbeddingAudit(model, inputs["input_ids"]) as embedding:
            profiled = generate(arm)
        assert profiled["output_text"] == results[arm]["output_text"], "profiling changed output"
        results[arm] = dict(response_row(results[arm]), timings=timed[arm], work=counter.result(),
                            embedding_audit=embedding.result(), output_repeat_and_profile_passed=True)
        assert len({r["first_top1_token_id"] for r in timed[arm]} | {profiled["first_top1_token_id"]}) == 1
    assert results["native"]["work"]["actual_prefill_pairs"] == inputs["input_ids"].numel() * 48 * 8
    return results, [order, list(reversed(order))]


def self_test():
    a, b = select_inputs(), select_inputs()
    assert a == b and len(a) == 8 and len({r["video"] for r in a}) == 8
    assert Counter(r["length_quartile"] for r in a) == {1: 2, 2: 2, 3: 2, 4: 2}
    assert set(a[0]) == {"video", "sample_index", "input_sequence_length", "length_quartile"}
    for n in (4, 6, 38, 768):
        for d in DENSITIES:
            ix = frame_indices(n, d)
            assert len(ix) % 2 == 0 and len(ix) <= n
    sig = torch.tensor([0.04, 0.08, 0.09])
    budgets = cg.cg_reduce_budget(sig, **TAU, base_budget=8, reduce_budget=4, boost_budget=8)
    assert budgets.tolist() == [8, 8, 4]
    from transformers import Qwen3OmniMoeProcessor
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(cg.MODEL_PATH), use_fast=False, local_files_only=True)
    conv = cg.build_conversation(Path("synthetic.mp4"), {"question": "What is visible?", "options": ["A. Test", "B. Other"]})
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    waveform = np.sin(np.arange(32000, dtype=np.float32) * 0.03)
    frames = torch.zeros(8, 3, 64, 64)
    counts = []
    for density in DENSITIES:
        selected_frames = frames[frame_indices(8, density)]
        fps = len(selected_frames) / 2
        prepared = processor(text=prompt, audio=[waveform], videos=[selected_frames], **processor_kwargs(fps))
        check = check_prepared_input(processor, prepared, len(waveform), len(selected_frames), fps)
        counts.append(check["token_counts"])
    assert len({r["audio"] for r in counts}) == 1
    assert counts[0]["video"] < counts[1]["video"] < counts[2]["video"]
    # Regression: exercise the exact old mixed-kwargs call and require rejection.
    broken = processor(text=prompt, audio=[waveform], videos=[frames], return_tensors="pt",
                       padding=True, use_audio_in_video=True, audio_kwargs={"truncation": False},
                       videos_kwargs={"fps": 4.0, "do_resize": False})
    assert int((broken["input_ids"] == cg.AUDIO_TOKEN_ID).sum()) == 0
    try:
        check_prepared_input(processor, broken, len(waveform), 8, 4.0)
    except AssertionError as error:
        assert "audio placeholders" in str(error)
    else:
        raise AssertionError("regression gate accepted the old silent failure")
    embeddings = torch.ones(1, 3, 4)
    mask = torch.tensor([[True, False, True]])
    assert EmbeddingAudit.check_rows(embeddings, mask, torch.ones(2, 4)) == 2
    for invalid in (torch.zeros(2, 4), torch.ones(1, 4)):
        try:
            EmbeddingAudit.check_rows(embeddings, mask, invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError("embedding gate accepted corrupt features")
    result = {"self_test": "passed", "old_bug_rejected": True,
              "corrupt_embedding_rejected": True, "synthetic_density_counts": counts, "selected": a}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir is None:
        parser.error("--output-dir required")
    if os.environ.get("TODO2A_WORKER_ID") != "4254369":
        raise RuntimeError("TODO 2A GPU execution is restricted to requested worker 4254369")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise RuntimeError("output already initialized; refusing overwrite or implicit resume")
    random.seed(cg.SEED)
    np.random.seed(cg.SEED)
    torch.manual_seed(cg.SEED)
    torch.cuda.manual_seed_all(cg.SEED)
    torch.set_num_threads(8)
    cg.atomic_json(out / "self_test.json", self_test())
    selected = select_inputs()
    protocol = {
        "experiment": "TODO-2A-Qwen-fixed-coverage-density-pilot",
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "supersedes_invalid_input_run": "todo2a_qwen_density_worker4254369_20260914",
        "result_status": "planned", "worker": "4254369", "selected": selected,
        "densities": DENSITIES, "seed": cg.SEED, "thresholds": TAU,
        "threshold_source": "rounded recorded MB-NEW-2b thresholds; not historical exact reproduction",
        "policy": POLICY, "layer_scope": [12, 32], "max_new_tokens": 4,
        "arms": ["native", "cgskip"], "warmups_per_input_arm": 1,
        "timed_repeats_per_input_arm": 2, "separate_profile_passes_per_input_arm": 1,
        "input_contract": "full audio, no truncation; endpoint-preserving video subsampling; fixed spatial dimensions; actual fps per density",
        "preflight": {"sample_indices": [selected[0]["sample_index"], selected[-1]["sample_index"]],
                      "densities": DENSITIES, "expected_inputs": 6,
                      "checks": "native/forced-top8 identity and actual CG generation; audio/video embedding insertion; all six must pass before timing"},
        "selection_contract": "input-length quartiles, 2 unique videos each; no outcome/category selection",
        "scope": "diagnostic only; no main-table replacement, quality equivalence or submission-level speed claim",
        "runtime_python": sys.executable,
        "sources_sha256": {str(p.relative_to(ROOT)): cg.sha256_file(p) for p in
                           (Path(__file__).resolve(), Path(cg.__file__), P1,
                            cg.BENCHMARKS["omnivideobench"]["dataset"], cg.MODEL_PATH / "config.json")},
    }
    cg.atomic_json(out / "protocol.json", protocol)
    from qwen_omni_utils.v2_5.audio_process import process_audio_info
    from qwen_omni_utils.v2_5.vision_process import fetch_video
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    import transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe as processor_source
    import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as model_source
    import transformers.processing_utils as processor_base
    cg.atomic_json(out / "environment.json", {
        "hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0), "command": sys.argv,
        "library_sources_sha256": {m.__name__: cg.sha256_file(Path(m.__file__))
                                   for m in (processor_source, model_source, processor_base)},
    })
    cg.log("TODO 2A: loading model; 8 videos x 3 densities, no full reference rerun")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(cg.MODEL_PATH), dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2").eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(cg.MODEL_PATH), use_fast=False)
    assert model.thinker.config.audio_token_id == cg.AUDIO_TOKEN_ID
    assert model.thinker.config.video_token_id == cg.VIDEO_TOKEN_ID
    controller = cg.CGSkipIntervention(model, 8, (12, 32))
    assert controller.installed == 48
    controller.policy = POLICY.copy()
    controller.set_thresholds(TAU)
    controller.prepare_sample(None)
    # Strip evaluation metadata at the loader boundary. Labels are joined only
    # by the downstream analysis, after predictions are finalized.
    raw_records = json.loads(cg.BENCHMARKS["omnivideobench"]["dataset"].read_text())
    records = [{k: r[k] for k in ("video", "question", "options")} for r in raw_records]
    del raw_records
    completed = 0
    preflight_completed = 0
    cg.atomic_json(out / "progress.json", {"completed": 0, "expected": 24,
                   "stage": "preflight", "preflight_completed": 0, "result_status": "planned"})
    try:
        # One-token short/long probes at EVERY density are excluded from timing.
        schedule = [(True, 0), (True, len(selected)-1)] + [(False, i) for i in range(len(selected))]
        for preflight, vi in schedule:
            item = selected[vi]
            if not preflight:
                assert preflight_completed == 6, "sampling cannot start before preflight passes"
            si = item["sample_index"]
            record = records[si]
            assert record["video"] == item["video"]
            path = cg.BENCHMARKS["omnivideobench"]["video_dir"] / record["video"]
            conv = cg.build_conversation(path, record)
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios = process_audio_info(conv, use_audio_in_video=True)
            assert len(audios) == 1
            video, dense_fps = fetch_video(conv[1]["content"][0], return_video_sample_fps=True)
            # One spatial resize, before the density intervention. The processor
            # must not choose a density-dependent spatial budget.
            factor = processor.video_processor.patch_size * processor.video_processor.merge_size
            hw = [max(factor, factor * round(s / factor)) for s in video.shape[-2:]]
            video = torch.nn.functional.interpolate(video, size=hw, mode="bicubic", align_corners=False)
            nframes = video.shape[0]
            audio_digest = hashlib.sha256(audios[0].tobytes()).hexdigest()
            density_order = np.random.default_rng(cg.SEED + vi).permutation(DENSITIES).tolist()
            audio_features_hash = spatial_grid = None
            density_checks = {}
            identity_passed = False
            for di, density in enumerate(density_order):
                indices = frame_indices(nframes, density)
                frames = video[indices]
                fps = float(dense_fps * len(indices) / nframes)
                inputs = processor(text=text, audio=audios, videos=[frames], **processor_kwargs(fps))
                input_gate = check_prepared_input(processor, inputs, len(audios[0]), len(indices), fps)
                feature_hash = tensor_hash(inputs["input_features"])
                grid = inputs["video_grid_thw"][0].tolist()
                if audio_features_hash is None:
                    audio_features_hash, spatial_grid = feature_hash, grid[1:]
                assert feature_hash == audio_features_hash and grid[1:] == spatial_grid
                ids = inputs["input_ids"][0]
                token_counts = input_gate["token_counts"]
                density_checks[density] = token_counts
                input_hash = tensor_hash(ids)
                inputs = inputs.to(model.device).to(model.dtype)
                versions = {k: v._version for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                cg.log(f"TODO 2A {'preflight' if preflight else 'generation'}: video {vi+1}/8 sample={si} density={density} tokens={len(ids)} audio={token_counts['audio']}")
                if preflight or not identity_passed:
                    with controller.arm("native"), EmbeddingAudit(model, inputs["input_ids"]) as embedding:
                        native = cg.generate_once(model, processor, inputs, 1)
                    with controller.arm("cgskip", force_budget=8):
                        identity = cg.generate_once(model, processor, inputs, 1)
                    js = cg.distribution_js(native["first_logits"], identity["first_logits"])
                    assert js < 1e-6, f"identity failed: {js}"
                    identity_passed = True
                    if not preflight:
                        cg.append_jsonl(out / "identity.jsonl", {"sample_index": si, "density": density,
                                        "js": js, "embedding_audit": embedding.result()})
                if preflight:
                    with controller.arm("cgskip"), EmbeddingAudit(model, inputs["input_ids"]) as cg_embedding:
                        cg.generate_once(model, processor, inputs, 1)
                    cg.append_jsonl(out / "preflight.jsonl", {"sample_index": si, "density": density,
                                    "input_gate": input_gate, "identity_js": js,
                                    "native_embedding": embedding.result(), "cg_embedding": cg_embedding.result()})
                    preflight_completed += 1
                    cg.atomic_json(out / "progress.json", {"completed": 0, "expected": 24,
                                   "stage": "preflight", "preflight_completed": preflight_completed,
                                   "result_status": "planned"})
                    cg.log(f"TODO 2A preflight passed {preflight_completed}/6")
                    del inputs, frames
                    continue
                arms, order = evaluate_input(model, processor, controller, inputs, (vi + di) % 2 == 0)
                assert versions == {k: v._version for k, v in inputs.items() if isinstance(v, torch.Tensor)}, "paired input mutated"
                row = dict(item, density=density, dense_frames=nframes,
                           selected_frame_indices=indices.tolist(), fps=fps,
                           dense_fps=dense_fps, video_grid_thw=grid,
                           audio_seconds=len(audios[0]) / 16000,
                           audio_waveform_sha256=audio_digest, audio_features_sha256=feature_hash,
                           input_ids_sha256=input_hash, input_sequence_length=len(ids),
                           token_counts_by_modality=token_counts, input_gate=input_gate, arm_orders=order,
                           arms=arms, result_status="placeholder")
                cg.append_jsonl(out / "raw_results.jsonl", row)
                completed += 1
                cg.atomic_json(out / "progress.json", {"completed": completed, "expected": 24,
                               "stage": "generation", "preflight_completed": preflight_completed,
                               "result_status": "planned"})
                cg.log(f"TODO 2A completed {completed}/24 paired inputs")
                del inputs, frames
            assert len({c["audio"] for c in density_checks.values()}) == 1
            assert len({c["text_and_markers"] for c in density_checks.values()}) == 1
            assert density_checks[0.25]["video"] < density_checks[0.5]["video"] < density_checks[1.0]["video"]
        cg.atomic_json(out / "progress.json", {"completed": completed, "expected": 24,
                       "stage": "complete", "preflight_completed": preflight_completed, "result_status": "placeholder",
                       "note": "all in-run gates passed; independent analysis pending"})
    except Exception:
        cg.atomic_json(out / "error.json", {"completed": completed, "traceback": traceback.format_exc()})
        raise
    finally:
        controller.close()


if __name__ == "__main__":
    main()
