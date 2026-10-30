"""Opt-in CUDA-event profiler for the ARC-SCEB sparse router path.

Load this *after* scripts/megatron_arc_sceb_router.py (as an extra external
plugin).  It wraps the ARC internal functions with CUDA-event timers and prints
a per-segment breakdown every ARC_SCEB_PROFILE_EVERY applied calls, so we can
see where the middle-layer ARC time actually goes before optimizing anything.

Segments timed (per middle-layer routing call):
  topk_score : topk_routing_with_score_function (dense_output=True)
  sort       : _sort_sparse_topk (the suspected redundant full-token sort)
  routemap   : _build_routing_map
  auxloss    : the three aux-loss applications
  mask       : _mask_selected_router_probs (ARC core)
  scatter    : _scatter_selected_probs

Nothing here changes numerics; it only measures.  Enabled only when
ARC_SCEB_PROFILE=1.
"""
from __future__ import annotations

import os
import torch

import importlib
mod = importlib.import_module("megatron_arc_sceb_router")

_ENABLED = os.environ.get("ARC_SCEB_PROFILE", "0") == "1"
_EVERY = int(os.environ.get("ARC_SCEB_PROFILE_EVERY", "50"))
_WARMUP = int(os.environ.get("ARC_SCEB_PROFILE_WARMUP", "40"))


class _Timer:
    """Accumulate CUDA-event elapsed ms per named segment."""

    def __init__(self):
        self.sum = {}
        self.cnt = {}
        self.calls = 0

    def add(self, name, ms):
        self.sum[name] = self.sum.get(name, 0.0) + ms
        self.cnt[name] = self.cnt.get(name, 0) + 1

    def report(self):
        order = ["topk_score", "sort", "routemap", "auxloss", "mask", "scatter", "total"]
        parts = []
        for k in order:
            if k in self.sum and self.cnt[k]:
                parts.append(f"{k}={self.sum[k]/self.cnt[k]:.3f}ms")
        return " ".join(parts)


_T = _Timer()


def _timed(name, fn, *a, **k):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn(*a, **k)
    e.record()
    torch.cuda.synchronize()
    _T.add(name, s.elapsed_time(e))
    return out


if _ENABLED:
    _orig_sort = mod._sort_sparse_topk
    _orig_routemap = mod._build_routing_map
    _orig_mask = mod._mask_selected_router_probs
    _orig_scatter = mod._scatter_selected_probs

    def _p_sort(*a, **k):
        return _timed("sort", _orig_sort, *a, **k)

    def _p_routemap(*a, **k):
        return _timed("routemap", _orig_routemap, *a, **k)

    def _p_mask(*a, **k):
        return _timed("mask", _orig_mask, *a, **k)

    def _p_scatter(*a, **k):
        return _timed("scatter", _orig_scatter, *a, **k)

    mod._sort_sparse_topk = _p_sort
    mod._build_routing_map = _p_routemap
    mod._mask_selected_router_probs = _p_mask
    mod._scatter_selected_probs = _p_scatter

    # Wrap the whole sparse routing to time the full call and drive reporting.
    _orig_route = mod._arc_sceb_sparse_topk_routing

    def _p_route(router, logits, padding_mask=None):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = _orig_route(router, logits, padding_mask=padding_mask)
        e.record()
        torch.cuda.synchronize()
        _T.add("total", s.elapsed_time(e))
        _T.calls += 1
        if (mod._is_rank_zero() and _T.calls > _WARMUP
                and (_T.calls - _WARMUP) % _EVERY == 0):
            print(f"[ARC-PROF] calls={_T.calls} {_T.report()}", flush=True)
        return out

    mod._arc_sceb_sparse_topk_routing = _p_route
    print(f"[ARC-PROF] installed sparse-path profiler every={_EVERY} warmup={_WARMUP}", flush=True)


# ---------------------------------------------------------------------------
# Micro-benchmark: compare the current full-token sort against a video-only
# selective sort on the *same* real logits, to decide if selective is worth it.
# Enabled together with the profiler.  Purely measurement; results are not used
# to change routing.
if _ENABLED and os.environ.get("ARC_SCEB_SORT_MICROBENCH", "1") == "1":
    _SORT_T = {"full": 0.0, "video_only": 0.0, "n": 0}
    _orig_sort_fn = mod._sort_sparse_topk  # this is _p_sort already; use raw

    def _raw_full_sort(top_values, top_indices):
        sv, ro = torch.topk(top_values, k=mod._TOP_K, dim=-1, sorted=True)
        return sv, torch.gather(top_indices, 1, ro)

    def _video_only_sort(top_values, top_indices, video_rows):
        # sort only the video rows; leave the rest untouched
        vv = top_values[video_rows]
        sv, ro = torch.topk(vv, k=mod._TOP_K, dim=-1, sorted=True)
        out_v = top_values.clone()
        out_i = top_indices.clone()
        out_v[video_rows] = sv
        out_i[video_rows] = torch.gather(top_indices[video_rows], 1, ro)
        return out_v, out_i

    # Hook into the topk call to time it and micro-bench the two sort strategies.
    _orig_topk_fn = mod.topk_routing_with_score_function

    def _timed_topk(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); tv, ti = _orig_topk_fn(*a, **k); e.record(); torch.cuda.synchronize()
        _T.add("topk_score", s.elapsed_time(e))
        # micro-bench the two sort strategies on this real tensor
        ctx = mod._CONTEXT
        if ctx is not None and ctx.video_rows is not None and tv.shape[0] == ctx.token_count:
            vr = ctx.video_rows.to(tv.device)
            for _ in range(3):  # a few reps to average out noise
                s1 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                s1.record(); _raw_full_sort(tv, ti); e1.record(); torch.cuda.synchronize()
                _SORT_T["full"] += s1.elapsed_time(e1)
                s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
                s2.record(); _video_only_sort(tv, ti, vr); e2.record(); torch.cuda.synchronize()
                _SORT_T["video_only"] += s2.elapsed_time(e2)
                _SORT_T["n"] += 1
            if mod._is_rank_zero() and _SORT_T["n"] % 300 == 0:
                nn = _SORT_T["n"]
                print(f"[ARC-PROF-SORT] reps={nn} "
                      f"full={_SORT_T['full']/nn:.4f}ms "
                      f"video_only={_SORT_T['video_only']/nn:.4f}ms "
                      f"(video_rows={vr.numel()}/{tv.shape[0]})", flush=True)
        return tv, ti

    mod.topk_routing_with_score_function = _timed_topk
    print("[ARC-PROF] installed sort micro-benchmark", flush=True)
