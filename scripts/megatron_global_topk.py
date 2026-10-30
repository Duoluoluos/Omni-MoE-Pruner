"""Global top-k reduction for Megatron-Core MoE routing.

This is the *simplest possible* speed experiment: instead of ARC-SCEB's
per-token, video-only, middle-layer budget, this plugin bluntly lowers the
router top-k for **every** token in **every** MoE layer (e.g. 8 -> 4).

Why this is the clean way to test "fewer experts => faster":
  * MCore ``TopKRouter`` copies ``self.topk = config.moe_router_topk`` at
    construction (router.py), while the token dispatcher reads
    ``config.moe_router_topk`` directly to size ``num_out_tokens =
    num_tokens * moe_router_topk`` (token_dispatcher.py).  Both are *static*.
  * So we must lower BOTH the config field and every already-built router's
    ``.topk`` attribute to keep them consistent.  When they agree, the
    grouped-GEMM simply processes ``num_tokens * new_topk`` rows -- a real,
    static reduction of expert compute with **no** GPU->CPU sync (unlike the
    variable-length compact-dispatch path we removed).

Unlike ARC-SCEB this is NOT modality-aware: text/audio/image quality will also
drop.  It is a throughput probe / ablation baseline, not a deployment policy.

Enabled only when ``MOE_GLOBAL_TOPK`` is set to a positive integer.  Load it as
an external plugin exactly like the ARC-SCEB router patch.
"""

from __future__ import annotations

import os
from typing import Any

from megatron.core.transformer.moe.router import TopKRouter


def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"


_RAW = os.environ.get("MOE_GLOBAL_TOPK", "0").strip()
try:
    _TARGET_TOPK = int(_RAW)
except ValueError as exc:  # pragma: no cover - defensive config guard
    raise RuntimeError(
        f"MOE_GLOBAL_TOPK must be a non-negative integer, got {_RAW!r}."
    ) from exc
if _TARGET_TOPK < 0:
    raise RuntimeError("MOE_GLOBAL_TOPK must be a non-negative integer.")

# ``0`` (the default) means "do nothing": leave the model's native top-k
# untouched so this plugin is transparent unless explicitly opted in.
_ENABLED = _TARGET_TOPK > 0

# Stash the *true* original constructor on the class the first time we are
# imported.  A second import / ``importlib.reload`` must recover this pristine
# reference instead of capturing our own already-installed wrapper, otherwise
# the wrapper would call itself and recurse infinitely.
if not hasattr(TopKRouter, "_global_topk_orig_init"):
    TopKRouter._global_topk_orig_init = TopKRouter.__init__
_ORIGINAL_TOPK_ROUTER_INIT = TopKRouter._global_topk_orig_init


def _global_topk_router_init(self: TopKRouter, *args: Any, **kwargs: Any) -> None:
    """Force both the router instance and its config onto the reduced top-k.

    ``TopKRouter.__init__`` sets ``self.topk = self.config.moe_router_topk``, so
    after the original constructor runs we overwrite both to guarantee the
    router *and* the token dispatcher (which reads ``config.moe_router_topk``)
    agree on the reduced value.  Consistency is what keeps the static
    ``num_out_tokens`` optimization valid and avoids the dropless mismatch that
    made variable-length compact dispatch slow.
    """
    _ORIGINAL_TOPK_ROUTER_INIT(self, *args, **kwargs)
    native_topk = int(self.config.moe_router_topk)
    if _TARGET_TOPK > native_topk:
        raise RuntimeError(
            f"[MOE-GLOBAL-TOPK] MOE_GLOBAL_TOPK={_TARGET_TOPK} exceeds the model's "
            f"native moe_router_topk={native_topk}; refusing to add experts."
        )
    self.config.moe_router_topk = _TARGET_TOPK
    self.topk = _TARGET_TOPK
    if _is_rank_zero() and not getattr(TopKRouter, "_global_topk_logged", False):
        print(
            f"[MOE-GLOBAL-TOPK] router top-k reduced {native_topk} -> {_TARGET_TOPK} "
            f"(all tokens, all MoE layers; layer_number="
            f"{getattr(self, 'layer_number', None)})",
            flush=True,
        )
        TopKRouter._global_topk_logged = True


if _ENABLED and not getattr(TopKRouter, "_global_topk_patch", False):
    TopKRouter.__init__ = _global_topk_router_init
    TopKRouter._global_topk_patch = True
    if _is_rank_zero():
        print(
            f"[MOE-GLOBAL-TOPK] installed global top-k patch target_topk={_TARGET_TOPK}",
            flush=True,
        )
elif not _ENABLED and _is_rank_zero():
    print(
        "[MOE-GLOBAL-TOPK] MOE_GLOBAL_TOPK unset/0; leaving native router top-k unchanged.",
        flush=True,
    )
