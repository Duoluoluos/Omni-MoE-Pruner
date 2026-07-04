## Evidence 1: Uniform Top-k Reduction Is Brittle

Goal:

> Show that reducing the expert budget uniformly is not enough. The same or
> stronger compute reduction can hurt accuracy more than SCEB, because pruning
> tolerance is not uniform across tokens and layers.


| Method | Budget Rule | Avg. Active Experts | Top-4 Share | Top-6 Share | Top-8 Share | Accuracy | Prefill Mean | E2E Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | original top-8 everywhere | 8.00 | 0.0% | 0.0% | 100.0% | 39.4% | 12.54s | 15.63s |
| Uniform top-6 | top-6 for all eligible token-layer pairs | 6.00 | 0.0% | 100.0% | 0.0% | 37.8% | 11.30s | 14.45s |
| Uniform top-4 | top-4 for all eligible token-layer pairs | 4.00 | 100.0% | 0.0% | 0.0% | 34.9% | 10.65s | 13.90s |
| Global SCEB | confidence-constrained top-4/6/8 | 7.06 | 10.6% | 25.6% | 63.8% | 39.7% | 11.59s | 14.52s |


Uniform top-k produces a monotonic latency reduction, but the accuracy loss is
disproportionate. Global SCEB keeps a higher average budget than uniform top-6,
but its key advantage is selective allocation: it applies smaller budgets only
where router confidence and structural eligibility indicate lower risk.

Main figure candidate:

| Method | Accuracy Delta vs Vanilla | Prefill Delta vs Vanilla |
|---|---:|---:|
| Uniform top-6 | -1.6 pp | -9.9% |
| Uniform top-4 | -4.5 pp | -15.1% |
| Global SCEB | +0.3 pp | -7.6% |

## Evidence 2: Modality-Layer Budgetability Is Structured

Goal:

> Show that budgetability concentrates in video tokens and middle layers, while
> early/late layers and non-video tokens are more sensitive.

Probe design:

- Apply a fixed top-4 budget to one modality-stage region at a time.
- Keep all other token-layer pairs at the original top-8.
- Evaluate answer flip rate and accuracy delta relative to vanilla on a matched
  subset.

 table:

| Target Region | Avg. Active Experts in Target | Answer Flip Rate | Accuracy Delta | Prefill Delta | Interpretation |
|---|---:|---:|---:|---:|---|
| Video early layers | 4.0 | 8.5% | -1.8 pp | -2.1% | Early visual processing is sensitive |
| Video middle layers | 4.0 | 4.8% | -0.4 pp | -7.0% | Best budgetable region |
| Video late layers | 4.0 | 7.6% | -1.5 pp | -2.9% | Late semantic integration is sensitive |
| Text/special middle layers | 4.0 | 12.4% | -3.2 pp | -0.4% | High risk, low speed gain |
| Audio middle layers | 4.0 | 10.8% | -2.4 pp | -0.3% | Task-dependent and low compute share |
| All middle layers | 4.0 | 13.1% | -3.6 pp | -7.6% | Mixing sensitive modalities hurts |

 heatmap shape:

| Modality / Stage | Early | Middle | Late |
|---|---:|---:|---:|
| Video | medium risk | low risk | medium risk |
| Text/special | high risk | high risk | high risk |
| Audio | high risk | medium-high risk | high risk |
| Image | high risk | medium-high risk | high risk |

 interpretation:

The speed opportunity mostly comes from video tokens because they dominate the
long-context prefill compute. However, safe pruning is not simply a modality
decision: layer stage matters. This motivates a structural eligibility mask
instead of global expert-budget reduction.

## Evidence 3: Router Confidence Predicts Pruning Risk

Goal:

> Show that router probability geometry is a useful label-free proxy for pruning
> risk. Higher top-expert decisiveness and stronger budget-boundary separation
> should correspond to lower perturbation and lower answer flip rate.

Probe design:

- For each candidate token-layer pair, compute:
  - decisiveness: `p1 - p2`
  - boundary stability: `p_m - p_{m+1}`
- Bucket token-layer pairs by confidence.
- Apply reduced budget in each bucket and measure output perturbation on matched
  samples.

 table, decisiveness buckets:

| `p1 - p2` Bucket | Token-Layer Share | Mean Chosen Budget | Hidden Cosine Distance | Logit KL | Answer Flip Rate | Accuracy Delta |
|---|---:|---:|---:|---:|---:|---:|
| Low | 33.3% | 8.0 | 0.018 | 0.092 | 9.8% | -2.7 pp |
| Medium | 33.3% | 6.0 | 0.010 | 0.047 | 5.6% | -0.9 pp |
| High | 33.3% | 4.0 | 0.004 | 0.018 | 2.9% | +0.1 pp |

 table, boundary-stability buckets:

| `p_m - p_{m+1}` Bucket | Token-Layer Share | Hidden Cosine Distance | Logit KL | Answer Flip Rate | Accuracy Delta |
|---|---:|---:|---:|---:|---:|
| Low boundary margin | 33.3% | 0.021 | 0.104 | 10.7% | -3.1 pp |
| Medium boundary margin | 33.3% | 0.011 | 0.052 | 5.9% | -1.1 pp |
| High boundary margin | 33.3% | 0.005 | 0.021 | 3.1% | -0.1 pp |

 combined-confidence table:

| Decisiveness | Boundary Stability | Suggested Budget | Answer Flip Rate | Accuracy Delta | Interpretation |
|---|---|---:|---:|---:|---|
| High | High | top-4 | 2.4% | +0.2 pp | Safest aggressive pruning |
| High | Low | top-6/top-8 | 6.8% | -1.2 pp | Top expert is clear, but truncation boundary is unstable |
| Low | High | top-6/top-8 | 7.5% | -1.6 pp | Boundary is stable, but routing is not decisive |
| Low | Low | top-8 | 12.0% | -3.5 pp | Unsafe region |

 interpretation:

Both confidence terms are needed. `p1 - p2` captures whether the router has a
clear dominant expert, while `p_m - p_{m+1}` captures whether cutting at a given
budget crosses an unstable expert boundary.

## Evidence 4: Signal Complementarity Ablation

Goal:

> Show that routing decisiveness and boundary stability are complementary. Each
> signal alone helps, but using both gives the best latency-accuracy tradeoff.

Expected table:

| Variant | Budget Rule | Avg. Active Experts | Top-4 Share | Top-6 Share | Top-8 Share | Accuracy | Prefill Mean | E2E Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | original top-8 everywhere | 8.00 | 0.0% | 0.0% | 100.0% | 39.4% | 12.54s | 15.63s |
| Decisiveness only | use `p1 - p2`, ignore boundary | 6.72 | 18.5% | 27.0% | 54.5% | 38.7% | 11.28s | 14.32s |
| Boundary only | use `p_m - p_{m+1}`, ignore `p1 - p2` | 6.86 | 14.2% | 31.0% | 54.8% | 39.0% | 11.40s | 14.41s |
| Both signals | full SCEB risk constraint | 7.06 | 10.6% | 25.6% | 63.8% | 39.7% | 11.59s | 14.52s |

 paired analysis:

| Variant | Answer Flip Rate vs Vanilla | Base Correct -> Wrong | Base Wrong -> Correct | Accuracy Delta |
|---|---:|---:|---:|---:|
| Decisiveness only | 7.2% | 24 | 17 | -0.7 pp |
| Boundary only | 6.4% | 20 | 16 | -0.4 pp |
| Both signals | 5.3% | 12 | 15 | +0.3 pp |

 interpretation:

The single-signal variants prune more aggressively but incur more answer flips.
The full SCEB rule is slightly more conservative, but it avoids the unstable
regions missed by either single signal and therefore produces the best
accuracy-preserving acceleration.

## How These Four Evidence Blocks Support the Story

| Story Claim | Supporting Evidence |
|---|---|
| Uniform expert-budget reduction is brittle | Evidence 1 |
| Redundancy is structured by modality and layer stage | Evidence 2 |
| Router geometry predicts pruning risk | Evidence 3 |
| SCEB is more than a heuristic threshold rule | Evidence 3 + Evidence 4 |
| Token-layer budgeting is the right abstraction | Evidence 1 + Evidence 2 + Evidence 4 |

## Replacement Checklist for Real Experiments

- Replace simulated uniform top-6/top-4 rows with measured full-run results.
- Replace modality-layer probe rows with measured answer flip and accuracy delta.
- Replace confidence bucket rows with measured hidden/logit/answer perturbation.
- Replace signal-complementarity rows with full or matched-subset ablations.
- Keep the interpretations only if the measured trends match the simulated
  direction.
