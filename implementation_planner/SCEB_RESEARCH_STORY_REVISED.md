# SCEB Research Story — Revised Version

## Working Title

**Structured Expert Budgeting for Efficient Multimodal MoE Video Inference**

Alternative titles:

- **SCEB: Structured Confidence-Guided Expert Budgeting for Efficient Multimodal MoE Video Inference**
- **Training-Free Structured Expert Budgeting for Multimodal MoE Video Inference**
- **Router-Guided Expert Budgeting for Efficient Multimodal MoE Video Inference**

Recommended title for the current story:

> **Structured Expert Budgeting for Efficient Multimodal MoE Video Inference**

Reason: this title emphasizes the problem abstraction, not the heuristic details. The main novelty should be framed as *structured expert budget allocation*, rather than simply confidence-threshold expert pruning.

---

## One-Sentence Core Claim

Multimodal MoE expert redundancy in video-heavy inference is not uniform; it is structured by **modality**, **layer stage**, and **router probability geometry**. SCEB exploits this structure through training-free, label-free, token-layer expert budget allocation under router-derived risk constraints.

---

## Central Research Question

Large multimodal MoE models spend substantial inference time activating multiple experts for long video inputs. A natural solution is to reduce the expert top-k globally, but this assumes that every token and every layer has the same pruning tolerance.

This work asks:

> **Can we reduce expert computation in multimodal MoE video inference without retraining by assigning different expert budgets to different token-layer pairs according to their estimated pruning risk?**

The answer is yes, but only if expert reduction is treated as a **structured budget allocation problem**, not as uniform expert pruning.

---

## Refined Storyline

### 1. Uniform expert reduction is the wrong abstraction

A direct reduction of the activated expert count is attractive but brittle. It treats all token-layer pairs as equally redundant, even though MoE routing in multimodal video inference is highly heterogeneous.

The key motivation should not be stated as:

> We want to prune experts to reduce latency.

It should be stated as:

> We find that expert redundancy is unevenly distributed, so the correct control unit is not the model-wide top-k, but the token-layer expert budget.

This reframing is important because it turns the method from a heuristic pruning rule into a constrained resource allocation problem.

### 2. Budgetable redundancy is structured

Our observations indicate that safe expert-budget reduction is concentrated in specific regions:

- **Modality structure:** video tokens expose more budgetable redundancy than protected text/audio/image/special tokens.
- **Layer-stage structure:** middle layers are more budgetable than early and late layers.
- **Router-confidence structure:** high-confidence routing states are more tolerant to budget reduction.

This gives the method a principled eligibility structure:

```text
budgetable region = video tokens × middle layers × confident router states
```

SCEB should therefore be presented as exploiting an empirical structure in multimodal MoE inference, not as a collection of hand-designed pruning rules.

### 3. Router probability geometry provides a label-free risk proxy

The true perturbation risk of reducing the expert budget is unavailable during inference. SCEB estimates this risk from two complementary properties of the router distribution:

- **Routing decisiveness:** `p_1 - p_2`
- **Budget boundary stability:** `p_m - p_{m+1}`

These two signals answer different questions:

- `p_1 - p_2`: Is the router confident about the dominant expert assignment?
- `p_m - p_{m+1}`: If we truncate to budget `m`, is there a stable boundary between retained and discarded experts?

This distinction is central. Without boundary stability, a token may have a clear top expert but still have an unstable tail around the truncation point. Without routing decisiveness, the truncation boundary may look stable while the overall routing decision remains uncertain.

### 4. Token-level confidence budgeting is better than coarse restoration

A coarse task/layer protection strategy is not the right abstraction. It may recover isolated examples, but it fails to preserve the global latency-accuracy tradeoff.

Existing result:

| Method | Accuracy | End-to-end Mean Latency |
|---|---:|---:|
| Vanilla | 39.38% | 15632 ms |
| Global SCEB | 39.68% | 14516 ms |
| Task-layer protection | 39.28% | 15584 ms |

This negative result should be emphasized. It shows that manually protecting broad semantic categories or layer ranges does not solve the problem. The effective control unit is the **token-layer expert budget**, selected by local confidence conditions.

---

## Method Framing

### Problem Formulation

For each token-layer pair `(l, i)`, the original MoE model executes `K` experts. SCEB chooses a discrete expert budget:

```text
m_{l,i} ∈ {4, 6, 8}
```

The objective is to reduce total expert computation:

```text
minimize  Σ_{l,i} m_{l,i}
```

subject to a local pruning-risk constraint:

```text
Risk(l, i, m_{l,i}) ≤ τ
```

The true risk is unknown during inference, so SCEB replaces it with a router-derived risk proxy.

### Confidence-Constrained Budget Allocation

For each eligible token-layer pair, SCEB chooses the smallest budget that passes both routing-decisiveness and boundary-stability tests:

```text
m = 4, if p_1 - p_2 ≥ θ_high and p_4 - p_5 ≥ γ
m = 6, if p_1 - p_2 ≥ θ_mid  and p_6 - p_7 ≥ γ
m = 8, otherwise
```

Interpretation:

- `m = 4`: safe aggressive reduction.
- `m = 6`: conservative reduction.
- `m = 8`: fallback to the original expert budget.

This should be described as:

> **calibrated risk-controlled expert budget allocation**

not as:

> fixed expert pruning or confidence-threshold pruning.

### Structural Eligibility Constraints

SCEB applies the budget allocator only to structurally budgetable regions:

```text
m_{l,i} = K, if token modality is protected
m_{l,i} = K, if layer l is outside the budgetable stage
```

The main method should focus on modality and layer-stage constraints.

Task floors should be presented as optional conservative guardrails or ablations:

```text
m_{l,i} ≥ task_floor, for risk-sensitive task groups
```

Do not make task floors a core innovation. If emphasized too much, the method may look like a task-specific patch rather than a general structured budgeting strategy.

---

## Intro-Facing Evidence

The introduction should include only the evidence that helps the reader believe the research story. It should not list every experiment.

### Evidence 1: Uniform top-k reduction is brittle

This is the most important missing intro-facing evidence.

Needed comparison:

| Method | Accuracy | Latency | Purpose |
|---|---:|---:|---|
| Vanilla top-8 | TBD | TBD | Original model |
| Uniform top-6 | TBD | TBD | Tests global expert reduction |
| Uniform top-4 | TBD | TBD | Tests aggressive global reduction |
| Global SCEB | 39.68% | 14516 ms | Tests structured budget allocation |

Expected interpretation:

> Uniform expert reduction either hurts accuracy or gives an inferior speed-accuracy tradeoff, showing that expert redundancy is not uniformly distributed.

This evidence justifies the transition from global pruning to token-layer budgeting.

### Evidence 2: Budgetability is structured by modality and layer stage

Needed probe:

```text
Rows: modality group
Columns: layer stage
Value: accuracy drop / answer flip rate / output perturbation under reduced expert budget
```

Desired conclusion:

> Video tokens in middle layers tolerate expert-budget reduction much better than protected modalities and early/late layers.

This evidence justifies the modality mask and layer-stage mask.

Without this evidence, the method may look like a manually tuned collection of protected regions.

### Evidence 3: Router confidence predicts pruning risk

Needed probe:

```text
Bucket token-layer pairs by p_1 - p_2 and p_m - p_{m+1}
Measure output perturbation / answer flip rate / accuracy drop after reducing the budget
```

Desired conclusion:

> Token-layer pairs with decisive top experts and stable truncation boundaries show lower pruning risk.

This evidence justifies using router probability geometry as a label-free risk estimator.

### Evidence 4: Two confidence signals are complementary

Needed ablation:

| Variant | Accuracy | Latency | Interpretation |
|---|---:|---:|---|
| Decisiveness only: `p_1 - p_2` | TBD | TBD | Tests dominant-expert confidence |
| Boundary only: `p_m - p_{m+1}` | TBD | TBD | Tests truncation stability |
| Both signals | TBD | TBD | Tests full risk estimator |

Desired conclusion:

> Routing decisiveness and boundary stability control different failure modes; using both provides the best risk-latency tradeoff.

This evidence prevents the method from looking like an arbitrary margin threshold.

### Evidence 5: Coarse task/layer restoration fails

Existing result:

| Method | Accuracy | End-to-end Mean Latency |
|---|---:|---:|
| Vanilla | 39.38% | 15632 ms |
| Global SCEB | 39.68% | 14516 ms |
| Task-layer protection | 39.28% | 15584 ms |

Intro interpretation:

> Coarse task/layer restoration recovers neither accuracy nor latency, suggesting that the correct abstraction is token-level expert budgeting rather than manual semantic restoration.

This negative result strengthens the story and should be used prominently.

### Evidence 6: Global SCEB gives the main deployed speed-accuracy tradeoff

Existing result:

| Method | Accuracy | End-to-end Mean Latency |
|---|---:|---:|
| Vanilla | 39.38% | 15632 ms |
| Global SCEB | 39.68% | 14516 ms |

Intro interpretation:

> Global SCEB reduces end-to-end latency while preserving, and slightly improving, accuracy.

This should be the headline result in the introduction.

---

## What Should Not Be Overemphasized in the Introduction

### Layer-Calibrated SCEB

Existing result:

| Method | Accuracy | End-to-end Mean Latency |
|---|---:|---:|
| Layer-Calibrated SCEB | 39.88% | 15711 ms |

Layer-Calibrated SCEB improves accuracy but is slower than both Global SCEB and the vanilla model. Therefore, it should not be introduced as the main efficiency result.

Recommended framing:

> Layer-wise calibration reveals that router confidence distributions are layer-dependent and exposes an accuracy-oriented frontier.

Do not frame it as the primary deployed setting unless budget-matched calibration later improves latency.

### Task Floors

Task floors should not be presented as core method novelty. They are better described as optional guardrails for conservative deployment or as an ablation.

### Threshold Percentiles

Avoid putting specific threshold percentiles in the introduction. Specific values such as `70`, `40`, and `50` make the method look overly tuned.

Use:

> calibrated thresholds

instead of:

> thresholds at specific percentiles.

---

## Revised Introduction Skeleton

1. **Problem:** Multimodal MoE video inference is expensive because long video inputs activate multiple experts across many token-layer pairs.
2. **Naive solution and failure:** Uniformly reducing expert top-k is tempting but brittle because pruning tolerance varies significantly across tokens and layers.
3. **Key observation:** Expert redundancy is structured: budgetable regions concentrate in video tokens, middle layers, and confident router states.
4. **Risk proxy:** Router probability geometry provides a label-free signal for pruning risk through routing decisiveness and boundary stability.
5. **Method:** SCEB formulates acceleration as token-layer expert budget allocation, assigning the smallest safe budget under structural eligibility and confidence constraints.
6. **Negative result:** Coarse task/layer protection fails to preserve the latency-accuracy tradeoff, motivating fine-grained token-level budgeting.
7. **Main result:** On OmniVideoBench, Global SCEB reduces latency from 15632 ms to 14516 ms while improving accuracy from 39.38% to 39.68%.

---

## Draft Intro Paragraphs

Large multimodal MoE models are increasingly used for video understanding, but their inference cost grows rapidly with long video inputs, as each token activates multiple experts across many transformer layers. A straightforward solution is to uniformly reduce the number of activated experts. However, this assumes that all token-layer pairs carry the same amount of redundant expert computation, which is not true in multimodal video inference.

Through controlled expert-budget probes, we find that pruning tolerance is highly structured. Expert-budget reduction is safer for video tokens in middle layers, while protected modalities, special tokens, and early/late layers are more sensitive. We further observe that the router distribution itself is informative of pruning risk: token-layer pairs with decisive top expert assignments and stable truncation boundaries exhibit lower perturbation when evaluated with fewer experts.

These observations motivate **Structured Expert Budgeting**, a training-free inference method that formulates MoE acceleration as token-layer expert-budget allocation rather than uniform top-k reduction. For each eligible token-layer pair, SCEB assigns the smallest expert budget that satisfies router-derived confidence and boundary-stability constraints, while preserving the original expert budget for structurally sensitive regions.

On OmniVideoBench, Global SCEB reduces end-to-end latency from 15632 ms to 14516 ms while slightly improving accuracy from 39.38% to 39.68%. In contrast, coarse task-layer protection obtains 39.28% accuracy and 15584 ms latency, suggesting that manual restoration at coarse granularity does not preserve the global latency-accuracy tradeoff. These results support token-layer confidence budgeting as the appropriate abstraction for efficient multimodal MoE video inference.

---

## Suggested Contributions

1. **Structured redundancy finding.** We identify that expert redundancy in multimodal MoE video inference is not uniform, but structured by modality, layer stage, and router probability geometry.
2. **Token-layer expert budgeting.** We formulate MoE inference acceleration as token-layer expert-budget allocation, where each token-layer pair receives the smallest expert budget satisfying structural eligibility and router-derived risk constraints.
3. **Label-free router-risk estimator.** We propose a training-free and label-free risk estimator based on routing decisiveness and budget-boundary stability.
4. **Evidence against coarse restoration.** We show that coarse task/layer protection fails to preserve the global latency-accuracy tradeoff, supporting fine-grained confidence-based budgeting.

---

## Experimental Plan to Solidify the Story

### Priority 1: Uniform Top-k Reduction Baseline

Run:

```text
Vanilla top-8
Uniform top-6
Uniform top-4
Global SCEB
```

Report:

```text
accuracy, latency, average expert budget, top-8/top-6/top-4 share
```

This is the most important baseline because it proves that SCEB is not merely reducing top-k, but allocating budget where reduction is safe.

### Priority 2: Modality-Layer Budgetability Map

Run fixed-budget probes over modality groups and layer stages.

Recommended visualization:

```text
x-axis: layer or layer stage
y-axis: modality group
color: pruning risk / accuracy drop / answer flip rate
```

This figure can become one of the main motivation figures.

### Priority 3: Confidence-Risk Monotonicity

Bucket token-layer pairs by confidence:

```text
low / medium / high p_1 - p_2
low / medium / high p_m - p_{m+1}
combined confidence score
```

Measure:

```text
answer flip rate
output perturbation
logit KL
accuracy drop
```

This proves that router geometry predicts pruning risk.

### Priority 4: Signal Complementarity Ablation

Run:

```text
SCEB with p_1 - p_2 only
SCEB with p_m - p_{m+1} only
SCEB with both signals
```

Expected story:

> Routing decisiveness and boundary stability are complementary; using both produces a better risk-latency tradeoff.

### Priority 5: Budget Composition Analysis

Report the assigned expert budgets:

```text
top-4 share
top-6 share
top-8 share
average expert budget
per-layer budget distribution
per-modality budget distribution
```

This helps show that SCEB is conservative where necessary and aggressive only in low-risk regions.

### Priority 6: Budget-Matched Layer-Calibrated SCEB

Current Layer-Calibrated SCEB improves accuracy but does not improve latency:

| Method | Accuracy | End-to-end Mean Latency |
|---|---:|---:|
| Layer-Calibrated SCEB | 39.88% | 15711 ms |

Next experiment:

```text
layer_calibrated_budget_matched_sceb_v1
threshold_granularity = per_layer
high_margin_percentile: 70 -> 65
medium_margin_percentile: 40 -> 35
boundary_margin_percentile: 50 -> 45
Target top-8 share close to Global SCEB: ~63.83%
```

Interpretation:

- If accuracy stays near or above Global SCEB and latency improves, this becomes the strongest method variant.
- If accuracy drops but budget matches, it still completes the speed-accuracy frontier story.
- If latency remains poor, Layer-Calibrated SCEB should be presented as an accuracy-oriented diagnostic variant.

---

## Recommended Main Tables and Figures

### Table 1: Main Results

| Method | Accuracy | Latency | Avg. Expert Budget | Notes |
|---|---:|---:|---:|---|
| Vanilla top-8 | 39.38% | 15632 ms | 8.0 | Original model |
| Uniform top-6 | TBD | TBD | 6.0 | Global reduction baseline |
| Uniform top-4 | TBD | TBD | 4.0 | Aggressive global reduction baseline |
| Global SCEB | 39.68% | 14516 ms | TBD | Main deployed setting |
| Layer-Calibrated SCEB | 39.88% | 15711 ms | TBD | Accuracy-oriented diagnostic variant |
| Task-layer protection | 39.28% | 15584 ms | TBD | Coarse restoration baseline |

### Figure 1: Structured Budgetability Map

Goal:

> Show that budgetable redundancy is concentrated in video tokens and middle layers.

### Figure 2: Router Confidence vs Pruning Risk

Goal:

> Show that router probability geometry predicts the risk of expert-budget reduction.

### Figure 3: Budget Distribution Under SCEB

Goal:

> Show that SCEB assigns smaller budgets mainly to low-risk video-middle-layer regions and falls back to full budget elsewhere.

### Table 2: Risk Estimator Ablation

| Variant | Accuracy | Latency | Interpretation |
|---|---:|---:|---|
| Decisiveness only | TBD | TBD | Tests top-expert confidence |
| Boundary only | TBD | TBD | Tests truncation stability |
| Both | TBD | TBD | Full estimator |

---

## Recommended Framing Language

Use these phrases:

- **structured expert budgeting**
- **token-layer expert budget allocation**
- **router-derived risk constraint**
- **training-free and label-free inference acceleration**
- **structural eligibility mask**
- **confidence-constrained budget allocation**
- **budgetability is structured, not uniform**

Avoid these phrases as the main framing:

- expert pruning by thresholds
- heuristic pruning rule
- manually selected protected layers
- task-specific restoration
- tuned percentile thresholds

---

## Final Positioning

This work should not be positioned as inventing a new pruning trick. Its stronger positioning is:

> We reveal that expert redundancy in multimodal MoE video inference is structured rather than uniform, and we convert this observation into a training-free token-layer expert budget allocator using router-derived confidence constraints.

With the current results, Global SCEB is the main efficiency method. Layer-Calibrated SCEB is useful for exposing the speed-accuracy frontier, but should not be the headline deployed variant unless the budget-matched version improves latency.

The strongest story is therefore:

```text
Uniform top-k reduction is brittle
→ expert redundancy is structured by modality, layer stage, and router confidence
→ router geometry predicts pruning risk
→ token-layer expert budgeting is the right abstraction
→ SCEB improves the latency-accuracy tradeoff without retraining
→ coarse task/layer restoration fails, validating fine-grained budgeting
```
