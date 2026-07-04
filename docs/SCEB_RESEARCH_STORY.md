# SCEB Research Story

## Working Title

Structured Confidence Expert Budgeting for Efficient Multimodal MoE Video Inference

## Core Claim

Multimodal MoE expert redundancy is not uniform. In video-heavy inference, budgetable redundancy is structured by modality, layer stage, and router confidence. SCEB exploits this structure by assigning token-level expert budgets under calibrated risk constraints, without retraining or labels.

## Motivation

Modern multimodal MoE models are expensive at inference time because long video inputs trigger top-k expert computation for many tokens across many transformer layers. A direct reduction of top-k is risky: expert routing carries semantic and modality-specific information, and the cost/accuracy tradeoff varies across layers and tasks.

The central observation from our experiments is that pruning safety is structured:

- Video tokens expose more budgetable redundancy than protected text/audio/image/special tokens.
- Middle layers are more budgetable than early and late layers.
- Router confidence predicts whether a token can tolerate a smaller expert budget.
- Coarse task/layer restoration is not sufficient; it can recover isolated samples while hurting latency and unrelated tasks.

## Method Framing

We formulate MoE pruning as a constrained expert-budget allocation problem. For each token-layer pair, the model originally executes `K` experts. SCEB chooses a discrete budget:

```text
m_{l,i} in {4, 6, 8}
```

The objective is to reduce total expert computation:

```text
minimize sum_{l,i} m_{l,i}
```

subject to a pruning-risk constraint:

```text
Risk(l, i, m_{l,i}) <= tau
```

The true perturbation risk is unknown during inference, so SCEB estimates risk from the router probability geometry.

## Router-Confidence Risk Estimator

For a token-layer router distribution, SCEB uses two label-free signals:

- Routing decisiveness: `p_1 - p_2`
- Boundary stability at budget `m`: `p_m - p_{m+1}`

SCEB chooses the smallest budget whose confidence and boundary-stability conditions pass calibrated thresholds:

```text
m = 4, if p_1 - p_2 >= theta_high and p_4 - p_5 >= gamma
m = 6, if p_1 - p_2 >= theta_mid  and p_6 - p_7 >= gamma
m = 8, otherwise
```

This is better described as calibrated risk-controlled budget allocation, not fixed expert pruning.

## Structural Constraints

SCEB applies the budget allocator only to budgetable token-layer pairs:

```text
m_{l,i} = K, if token modality is protected
m_{l,i} = K, if layer l is outside the budgetable stage
m_{l,i} >= task_floor, for risk-sensitive task groups
```

The main method should emphasize modality and layer-stage constraints. Task floors are best presented as a conservative guardrail or ablation, not the core innovation.

## Global vs Layer-Calibrated SCEB

Global SCEB calibrates one set of thresholds across all budgetable middle layers. It provides the strongest latency improvement:

- Vanilla: 39.38% accuracy, 15632 ms end-to-end mean latency
- Global SCEB: 39.68% accuracy, 14516 ms end-to-end mean latency

Layer-Calibrated SCEB calibrates thresholds separately for each dynamic layer. It exposes that routing confidence distributions are layer-dependent:

- Layer-Calibrated SCEB: 39.88% accuracy, 15711 ms end-to-end mean latency

This gives a useful speed-accuracy frontier: global calibration is speed-oriented, while layer calibration is accuracy-oriented.

## Negative Result That Strengthens the Story

Coarse task-layer protection is not the right abstraction. Protecting hypothetical/relationship reasoning at layers 24-31 improved a narrow slice but reduced overall accuracy and gave up the latency benefit:

- Task-layer protection: 39.28% accuracy, 15584 ms end-to-end mean latency

This supports the method claim that token-level confidence allocation is more reliable than coarse manual restoration.

## Suggested Contributions

1. We identify structured MoE redundancy in multimodal video inference, concentrated in video-token, middle-layer, high-confidence routing regions.
2. We propose SCEB, a label-free calibrated expert-budget allocator that converts router confidence into per-token expert budgets under structural constraints.
3. We introduce layer-calibrated SCEB to expose layer-dependent routing risk and a controllable speed-accuracy frontier.
4. We provide ablations showing that coarse task/layer protection is inferior to calibrated token-level budgeting.

## Intro Skeleton

1. Large multimodal MoE models are increasingly used for video understanding, but long video inputs make inference costly because top-k expert routing is executed for many tokens and layers.
2. Existing efficiency approaches often treat pruning as uniform or coarse-grained, but MoE routing in multimodal models is heterogeneous: different modalities, layer stages, and router-confidence regimes have different tolerance to expert-budget reduction.
3. We ask whether MoE expert computation can be reduced without retraining by using the router itself as a confidence signal.
4. We show that safe budget reduction is concentrated in video tokens at middle layers when router assignments are decisive and truncation boundaries are stable.
5. We propose SCEB, which assigns the smallest expert budget satisfying calibrated confidence and boundary-stability constraints while protecting non-budgetable modalities and layer stages.
6. Experiments on OmniVideoBench show that global SCEB improves the latency-accuracy tradeoff, while layer-calibrated SCEB improves accuracy and reveals a speed-accuracy frontier.

## Next Experiment

Run `layer_calibrated_budget_matched_sceb_v1`:

- Keep `threshold_granularity = per_layer`.
- Make thresholds slightly more aggressive to match the global SCEB budget:
  - `high_margin_percentile`: 70 -> 65
  - `medium_margin_percentile`: 40 -> 35
  - `boundary_margin_percentile`: 50 -> 45
- Target: top-8 share close to global SCEB (`~63.83%`) while keeping the layer-calibrated accuracy benefit.

Interpretation:

- If accuracy stays near or above global SCEB and latency improves, this is the strongest method variant.
- If accuracy drops but budget matches, it still completes the speed-accuracy frontier story.
- If it remains slow, layer calibration should be presented as accuracy-oriented rather than the main deployed setting.
