# SCEB Research Context Compact

This note is the compact context anchor for the SCEB research thread.

## Core Story

Multimodal MoE video inference is expensive because long video prompts activate
multiple experts across many token-layer pairs. The central claim is not merely
that fewer experts can be used, but that expert redundancy is structured.

SCEB frames inference acceleration as token-layer expert budget allocation:

- structurally eligible regions receive a reduced expert budget;
- sensitive regions keep the original budget;
- router probability geometry provides a label-free risk signal;
- the method is training-free and does not alter model weights.

Current strongest deployed setting:

- Global SCEB is the main latency-accuracy result.
- Layer-Calibrated SCEB is diagnostic / accuracy-oriented, not the headline
  efficiency setting.
- Compact dispatch is mathematically aligned with mask-only budgeting, but the
  current implementation is slower because dispatch/scatter/kernel overhead
  offsets saved expert MLP work.

## Known Full-Run Anchors

| Method | Accuracy | Prefill Mean | E2E Mean | Note |
|---|---:|---:|---:|---|
| Vanilla | 39.38% | 12540 ms | 15632 ms | original top-8 |
| Global SCEB | 39.68% | 11586 ms | 14516 ms | current main setting |
| Compact SCEB | 39.68% | 12098 ms | 15215 ms | accurate but slower than Global |
| Layer-Calibrated SCEB | 39.88% | 12671 ms | 15711 ms | diagnostic frontier |
| Budget-Matched Layer-Calibrated | 39.08% | 13351 ms | 16447 ms | too slow / accuracy drop |
| Task-Layer Protection | 39.28% | 12570 ms | 15584 ms | coarse restoration baseline |

## Four Evidence Blocks To Complete

### 1. Uniform top-6/top-4 true baselines

Most important missing baseline. This supports:

> SCEB is not just lowering top-k; it allocates reduced expert budgets where
> reduction is safe.

Needed:

- true uniform top-6;
- true uniform top-4;
- preferably all token-layer pairs, or clearly defined all eligible pairs;
- report accuracy, prefill/e2e latency, avg active experts.

### 2. Confidence signal ablation

Low-cost method-depth evidence:

- `p1 - p2` decisiveness only;
- `p_m - p_{m+1}` boundary stability only;
- both signals.

This supports that the router-risk estimator is not a single arbitrary
threshold.

### 3. Confidence-risk bucket probe

Most insight-like evidence:

- bucket token-layer pairs by router confidence;
- measure hidden/logit perturbation and answer flip on a small paired subset;
- full run is not required.

This supports:

> Router geometry predicts pruning risk.

### 4. Modality-layer budgetability map

Motivation figure:

- fixed budget probes for video early/middle/late;
- compare with protected/non-video tokens;
- avoid making it too heavy.

This supports:

> Budgetability is structured by modality and layer stage.

## Current Priority

Start with Evidence 1. Existing fixed top-m runs were video-middle only, not
true uniform baselines. Add minimal selector support for `target_modality: "all"`
and run top-6/top-4 baselines on worker 3936304.
