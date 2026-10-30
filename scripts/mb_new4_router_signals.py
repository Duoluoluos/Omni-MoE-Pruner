"""Leakage-free router-geometry signal library for MB-NEW-4 signal search.

Each signal maps a per-token router distribution to a per-token budget in a
candidate set, using CALIBRATED QUANTILES of the signal (so every signal yields a
structurally dispersed budget histogram and cannot degenerate to a constant).

Inputs are always a probability tensor. Two conventions are supported:
  - full_probs: (n_tokens, num_experts) softmax over ALL experts (Qwen top-8 path
    passes the full 128-way softmax; we then take top_k for the signals that need
    ranked weights).
  - For Uni-MoE, the dynamic-expert softmax (n_tokens, K) is passed directly.

The mapping is monotone in a scalar signal value s(token): tokens with the
LOWEST s get the smallest budget, tokens with the HIGHEST s get the largest,
split by calibrated quantiles. Each signal defines s so that "larger s => keep
more experts" is the hypothesis under test; the direction is NOT assumed correct
and is exactly what the 50-sample screen measures. For signals whose natural
direction is "larger => fewer experts", we expose an explicit reversed variant.

No signal reads any label, task_type, audio_type, answer, or outcome feature.
"""

from __future__ import annotations

import numpy as np
import torch


def _ranked_weights(probs: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return the sorted-descending top_k routing weights, renormalized to sum 1."""
    vals, _ = torch.topk(probs, top_k, dim=-1)
    vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return vals


def signal_values(probs: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
    """Compute all candidate scalar signals s(token). Returns name -> (n_tokens,).

    Convention: larger s => hypothesis says keep MORE experts. Reversed variants
    are provided where the opposite is equally plausible.
    """
    w = _ranked_weights(probs, top_k)  # (n, top_k), sorted desc, sum 1
    n, k = w.shape
    eps = 1e-12
    p = w.clamp_min(eps)

    # Normalized Shannon entropy in [0,1].
    ent = -(p * p.log()).sum(dim=-1) / float(np.log(k))
    # Top-1 dominance.
    top1 = w[:, 0]
    # Top-2 margin.
    margin12 = w[:, 0] - (w[:, 1] if k > 1 else torch.zeros_like(w[:, 0]))
    # Gini concentration of the ranked weights.
    idx = torch.arange(1, k + 1, device=w.device, dtype=w.dtype).unsqueeze(0)
    gini = (2.0 * (idx * w).sum(dim=-1) / (k * w.sum(dim=-1).clamp_min(eps))) - (
        k + 1.0
    ) / k
    # Participation ratio (effective number of experts): (sum p)^2 / sum p^2.
    pr = (w.sum(dim=-1) ** 2) / (w.pow(2).sum(dim=-1).clamp_min(eps))
    pr_norm = pr / k  # in (0,1]
    # Cumulative mass at half the candidate ceiling (how concentrated the mass is).
    csum = torch.cumsum(w, dim=-1)
    half = max(1, k // 2)
    cummass_half = csum[:, half - 1]

    return {
        # "keep more when uncertain/spread" family (larger s => more experts)
        "entropy_high": ent,
        "participation_high": pr_norm,
        # "keep more when concentrated" family (larger s => more experts)
        "top1_dominance": top1,
        "margin12": margin12,
        "gini": gini,
        "cummass_half": cummass_half,
        # explicit reversals of the two most plausible-either-way signals
        "entropy_low": -ent,
        "top1_low": -top1,
    }


ALL_SIGNALS = [
    "entropy_high",
    "entropy_low",
    "participation_high",
    "top1_dominance",
    "top1_low",
    "margin12",
    "gini",
    "cummass_half",
]


def budget_from_quantiles(
    s: torch.Tensor,
    q_low: float,
    q_high: float,
    candidate_budgets: list[int],
) -> torch.Tensor:
    """Map scalar signal s to a budget via calibrated tertile thresholds.

    s below q_low  -> smallest budget
    s in [q_low,q_high) -> middle budget
    s at/above q_high -> largest budget
    """
    cands = sorted(int(b) for b in candidate_budgets)
    lo_b, mid_b, hi_b = cands[0], cands[len(cands) // 2], cands[-1]
    out = torch.full_like(s, float(mid_b))
    out = torch.where(s < q_low, torch.full_like(s, float(lo_b)), out)
    out = torch.where(s >= q_high, torch.full_like(s, float(hi_b)), out)
    return out.to(torch.int64).clamp_(min=lo_b, max=hi_b)


def boundary_margins(
    probs: torch.Tensor, top_k: int, candidate_budgets: list[int]
) -> dict[int, torch.Tensor]:
    """Original SCEB boundary margins g_m = p_m - p_{m+1} for each candidate m.

    A large g_m means the router decisively stops at m experts (tail redundant).
    Uses the renormalized sorted top_k weights.
    """
    w = _ranked_weights(probs, top_k)
    out: dict[int, torch.Tensor] = {}
    for m in sorted(int(b) for b in candidate_budgets):
        if m < w.shape[1]:
            out[m] = w[:, m - 1] - w[:, m]
        else:
            out[m] = w[:, m - 1]
    return out


def boundary_budget(
    probs: torch.Tensor,
    top_k: int,
    candidate_budgets: list[int],
    thresholds: dict[int, float],
) -> torch.Tensor:
    """Original SCEB boundary-margin selection with per-candidate thresholds.

    For each token pick the SMALLEST candidate m whose boundary margin g_m
    exceeds tau_m (router decisively stops at m), else grow to the ceiling.
    Per-candidate thresholds (each calibrated as a quantile of that candidate's
    own margin distribution) let the budget disperse across the candidate set,
    avoiding the v1 single-absolute-threshold degeneracy on small softmaxes.
    """
    cands = sorted(int(b) for b in candidate_budgets)
    g = boundary_margins(probs, top_k, cands)
    n = probs.shape[0]
    ceiling = cands[-1]
    chosen = torch.full((n,), ceiling, dtype=torch.int64, device=probs.device)
    for m in sorted(cands, reverse=True):
        decisive = g[m] >= thresholds[m]
        chosen = torch.where(decisive, torch.full_like(chosen, m), chosen)
    return chosen.clamp_(min=cands[0], max=ceiling)
