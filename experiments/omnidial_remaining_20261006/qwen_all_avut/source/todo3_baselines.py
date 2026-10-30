"""Paper-equation ports for TODO 3; no benchmark metadata enters this module.

Alloc-MoE: ACL 2026 long.437, Algorithms 1 / Eq. 3--10 and Sec. 3.3.
ASET: Findings ACL 2026 .2140, Algorithm 1 / Eq. 4--9.
These are independent ports, not authors' released implementation code.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
import torch


def aset_budget(probs, native_budget, theta_min, theta_max, gamma=2.0):
    """Smallest cumulative prefix, entropy over the theta_max candidate prefix."""
    if not 0 <= theta_min <= theta_max <= 1 or gamma <= 1:
        raise ValueError("Invalid ASET parameters")
    p = probs.float().sort(dim=-1, descending=True).values
    cumulative = p.cumsum(-1)
    candidate_n = (cumulative < theta_max).sum(-1).add(1).clamp(max=p.shape[-1])
    ranks = torch.arange(p.shape[-1], device=p.device)
    candidate = p * (ranks < candidate_n[:, None])
    candidate = candidate / candidate.sum(-1, keepdim=True).clamp_min(1e-30)
    entropy = -(candidate * candidate.clamp_min(1e-30).log()).sum(-1)
    entropy = torch.where(candidate_n > 1, entropy / candidate_n.float().log().clamp_min(1e-30), 0)
    theta = theta_min + (theta_max - theta_min) * entropy.clamp(0, 1).pow(gamma)
    native = torch.as_tensor(native_budget, device=p.device).long().expand(p.shape[0])
    mass = cumulative.gather(1, (native - 1)[:, None]).squeeze(1)
    count = (cumulative < (mass * theta)[:, None]).sum(-1).add(1)
    return torch.minimum(count, native)


def alloc_token_mask(sorted_scores, layer_budget, base=1):
    """Exact Alloc-T global selection, with a deterministic prefix tie rule.

    Candidates are K_orig (Sec. 3.3's operational description); Eq. 7's
    |C|=T*K_l would otherwise leave nothing to redistribute. Native router
    probabilities are used before selected-weight normalization.
    """
    tokens, maximum = sorted_scores.shape
    total = int(round(tokens * float(layer_budget)))
    if not tokens * base <= total <= tokens * maximum:
        raise ValueError("Infeasible Alloc-T budget")
    mask = torch.zeros_like(sorted_scores, dtype=torch.bool)
    mask[:, :base] = True
    remaining = total - tokens * base
    if remaining == 0:
        return mask
    if remaining == tokens * (maximum - base):
        mask[:] = True
        return mask
    tail = sorted_scores[:, base:].reshape(-1)
    indices = torch.topk(tail, remaining, sorted=False).indices
    # Tie correction only affects equal scores at the selection boundary.
    boundary = tail[indices].min()
    greater = tail > boundary
    equal = tail == boundary
    tie_rank = equal.to(torch.int64).cumsum(0)
    selected = greater | (equal & (tie_rank <= remaining - greater.sum()))
    mask[:, base:] = selected.reshape(tokens, maximum - base)
    return mask


def alloc_layer_dp(sensitivity, budget, exact=False):
    """Exact multiple-choice knapsack; rows index k=1..K_orig.

    Algorithm 1 returns raw PPL. Row-offset subtraction preserves its argmin
    without inventing an unspecified multiplicative normalization.
    Default is the paper's <=B constraint; exact=True is a separately named
    work-matched control, not the primary Alloc-MoE port.
    """
    s = np.asarray(sensitivity, dtype=np.float64)
    if s.ndim != 2 or not np.isfinite(s).all():
        raise ValueError("Sensitivity must be a finite matrix")
    layers, maximum = s.shape
    budget = int(budget)
    if not layers <= budget <= layers * maximum:
        raise ValueError("Infeasible Alloc-L budget")
    s = s - s.min(axis=1, keepdims=True)
    dp = np.full((layers + 1, budget + 1), np.inf)
    choices = np.zeros_like(dp, dtype=int)
    dp[0, 0] = 0
    for i in range(1, layers + 1):
        for b in range(i, min(budget, i * maximum) + 1):
            for k in range(1, min(maximum, b) + 1):
                value = dp[i - 1, b - k] + s[i - 1, k - 1]
                if value < dp[i, b]:
                    dp[i, b], choices[i, b] = value, k
    b = budget if exact else int(np.argmin(dp[layers]))
    allocation = []
    for i in range(layers, 0, -1):
        k = int(choices[i, b])
        if not k:
            raise ValueError("No feasible allocation")
        allocation.append(k)
        b -= k
    return allocation[::-1]


def self_test():
    rng = np.random.default_rng(20260921)
    # Independent scalar ASET reference, including a singleton candidate set.
    cases = np.vstack([rng.dirichlet(np.ones(8), 40), [0.99] + [0.01 / 7] * 7])
    for low, high in [(0.2, 0.4), (0.5, 0.7), (0.8, 1.0), (1., 1.)]:
        expected = []
        for row in cases:
            p = sorted(row, reverse=True)
            nw = next((i + 1 for i in range(8) if sum(p[:i+1]) >= high), 8)
            q = [v / sum(p[:nw]) for v in p[:nw]]
            h = -sum(v * math.log(v) for v in q) / math.log(nw) if nw > 1 else 0
            threshold = sum(p[:4]) * (low + (high-low)*h**2)
            expected.append(next((i+1 for i in range(4) if sum(p[:i+1]) >= threshold - 1e-7), 4))
        got = aset_budget(torch.tensor(cases), 4, low, high).tolist()
        assert got == expected, (low, high, got, expected)
    # Exhaustive optimization checks prove global-selection and DP objectives.
    scores = torch.tensor([[.5,.3,.2], [.8,.1,.1]])
    for total in range(2, 7):
        mask = alloc_token_mask(scores, total / 2)
        assert mask.sum() == total and mask[:, 0].all()
        assert (mask[:, 1:].int() <= mask[:, :-1].int()).all()
        optimum = max(float((scores*torch.tensor(bits).reshape(2,3)).sum())
                      for bits in itertools.product([0,1], repeat=6)
                      if sum(bits) == total and bits[0] and bits[3])
        assert abs(float((scores*mask).sum())-optimum) < 1e-6
    for _ in range(20):
        s = rng.normal(size=(3, 3))
        for b in range(3, 10):
            for exact in [False, True]:
                feasible = [c for c in itertools.product(range(1,4), repeat=3)
                            if (sum(c) == b if exact else sum(c) <= b)]
                optimum = min(sum(s[i,k-1] for i,k in enumerate(c)) for c in feasible)
                got = alloc_layer_dp(s,b,exact=exact)
                assert abs(sum(s[i,k-1] for i,k in enumerate(got))-optimum) < 1e-9
    print("TODO3 baseline algorithm tests passed")


if __name__ == "__main__":
    self_test()
