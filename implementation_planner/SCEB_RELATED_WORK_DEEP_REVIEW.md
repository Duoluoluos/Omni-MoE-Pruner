# SCEB Related Work Deep Review

Date: 2026-07-09
Scope: `reference/sceb_related/*.pdf` plus existing MACS note, interpreted against `implementation/SCEB_AUTORESEARCH_HANDOFF.md`.

## Executive Verdict

The current broad SCEB framing is no longer safe:

```text
Structured / router-guided dynamic expert budgeting for multimodal or omnimodal MoE inference
```

This space is heavily occupied by two close papers:

- **AnyExperts** already claims multimodal token-level dynamic expert allocation across image, video, audio, text, with learned token importance, virtual experts, and full multimodal training.
- **MoDES** already claims training-free multimodal MoE expert skipping, with layer-aware importance calibration and modality-specific thresholds on Qwen3-VL-MoE / Kimi-VL / InternVL.

The defensible SCEB position should be narrowed to:

```text
Post-hoc router-risk calibration for frozen omnimodal MoE video inference.
```

In paper language:

```text
Given a frozen omnimodal MoE model, can we safely skip experts in video-heavy inference by using only existing router geometry and conservative modality/layer guards, without model retraining, auxiliary importance predictors, virtual experts, or offline KL threshold search?
```

This means:

- Do **not** claim "first dynamic expert budgeting for multimodal / omnimodal MoE".
- Do **not** claim "first training-free expert skipping for MLLM".
- Do **not** headline AV-SCEB as "audio-video dynamic expert allocation".
- Do headline "frozen-model deployment policy", "router-risk proxy", "video-only safe region", and "protected audio/text policy".

## Collision Map

| Paper | Control Unit | Training Needed | Modality Scope | Threat To SCEB | How SCEB Must Differentiate |
|---|---|---:|---|---|---|
| AnyExperts | dynamic real/virtual expert slots per token | yes, full multimodal training | image/video/audio/text | very high for broad dynamic allocation | SCEB is post-hoc on a frozen model; no importance MLP, no virtual experts, no retraining, only conservative down-budgeting |
| MoDES | expert skipping by calibrated expert importance | no model training, but offline calibration/search | image/video + text MLLMs | very high for training-free expert skipping | SCEB must focus on omnimodal audio-video models, protected audio/text, router-boundary risk, and no KL frontier search |
| MACS | modality-aware capacity/load scaling | no | image/video + text MLLMs | medium-high systems neighbor | SCEB changes active expert budget; MACS changes capacity/load/rerouting |
| ReaLB | per-rank precision/load balancing | no | multimodal MoE | medium systems neighbor | SCEB is algorithmic expert skipping; ReaLB is runtime precision scheduling |
| BEAM | trainable binary mask over top-k experts | yes, SFT | mostly LLM MoE | medium | BEAM validates mask-based sparsity but needs training; SCEB is label-free inference policy |
| SMIDT | serving framework for dynamic Top-K | no routing policy itself | MoE LLM serving | medium-low | Useful for compact-dispatch implementation once SCEB has dynamic top-m |
| Harder Tasks Need More Experts | dynamic top-p expert routing | yes, trained model | text-only | low-medium conceptual prior | Use as text-only precedent for difficulty/adaptive experts |
| OmniZip | audio-guided AV token compression | no | Qwen2.5-Omni audio-video | medium for AV-SCEB | Borrow audio-as-guard idea, but SCEB budgets experts not tokens |
| OmniSIFT | vision-guided audio token selection | yes, light FT | Qwen2.5-Omni audio-video | medium for audio branch | Warns direct audio compression/budgeting may need learned guidance |
| OmniSelect | query/modality-aware AV token compression | no | Qwen2.5-Omni audio-video | medium for AV-SCEB | Borrow query-regime controller for video expert budget |

## Paper-By-Paper Deep Notes

### AnyExperts

**What it does.** AnyExperts replaces fixed top-k routing with on-demand token-level expert allocation. It estimates token semantic importance using a lightweight MLP, maps the importance score to a dynamic expert-slot count within `[Kmin, Kmax]`, and fills slots with real experts or virtual experts. Virtual experts bypass standard expert computation, so low-importance tokens consume less real expert compute.

**Key details.**

- Importance score: `w_i = sigmoid(MLP(LayerNorm(x_i)))`.
- Dynamic slots: `K_i = Kmin + (Kmax - Kmin) * w_i`.
- Real/virtual routing is biased by importance: high-importance tokens prefer real experts; redundant tokens prefer virtual experts.
- Uses auxiliary token importance regularization and balancing loss.
- Follows Ming-Omni-style training: large-scale pretraining plus SFT.
- Evaluates general image, OCR, video, speech/audio, and NLP tasks.
- Reports that image/video can tolerate up to 40% fewer real activations, while OCR/NLP only tolerate mild reduction.

**Threat.** This fully occupies the broad "multimodal MoE dynamic expert allocation" story. If SCEB is pitched as dynamic expert budget allocation across multimodal tokens, reviewers will immediately compare to AnyExperts and ask why SCEB is less complete.

**Safe SCEB angle.** AnyExperts is a model-design/training method. SCEB should be framed as a post-hoc deployment policy for a frozen model, with no MLP, no virtual experts, no changed training objective, and no up-scaling of expert slots.

### MoDES

**What it does.** MoDES is the closest direct prior. It is a training-free expert skipping framework for MoE MLLMs. It identifies two MLLM-specific issues: layer-wise contribution imbalance and modality gap between vision/text tokens. It then combines offline layer calibration with local routing probability and applies separate thresholds for text and vision.

**Key details.**

- Local router probability: `pi_i^(l)`.
- Layer global factor: `alpha^(l)` from KL divergence between original outputs and outputs with the `l`-th layer experts skipped.
- Expert importance: `s_i^(l) = alpha^(l) * pi_i^(l)`.
- Dual-modality thresholds: `tau_t` for text, `tau_v` for vision.
- Uses a frontier search over threshold grid to satisfy a target skipping ratio.
- Calibrates on 1024 GQA examples.
- Evaluates Kimi-VL-A3B, Qwen3-VL-MoE-30B-A3B, InternVL-3.5.
- Reports roughly 2x prefill speedup and strong retention at high skipping ratios.

**Threat.** Very high. It already says training-free, MLLM, expert skipping, layer-aware, modality-aware, Qwen3-VL-MoE, video benchmarks. This forces SCEB to narrow sharply.

**Safe SCEB angle.** SCEB must avoid claiming novelty around training-free MLLM expert skipping. The possible remaining niche is:

- **omnimodal audio-video model**, not only vision-language MLLM;
- **audio/text protected policy**, not text-vs-vision dual thresholds;
- **router-boundary risk proxy**, not KL-calibrated layer global importance;
- **no frontier search**, if we keep thresholding simple and inference-local;
- **video-only token-layer risk evidence** on OmniVideoBench / Qwen3-Omni-like workloads.

If we cannot establish these differences experimentally, SCEB becomes more of an internal engineering variant than a strong standalone paper.

### MACS

**What it does.** MACS is a training-free modality-aware capacity scaling method for efficient MLLM MoE inference. It uses visual token information estimates and batch modality composition to adjust expert capacity and reroute/drop overflow tokens more safely.

**Threat.** Medium-high as a systems neighbor. It does not choose fewer active experts per token, but it shares the high-level claim that multimodal MoE inference should be modality-aware and that visual tokens are more redundant.

**Use for SCEB.** Cite as deployment-side capacity/load neighbor. SCEB should distinguish "active expert count/budget" from "capacity scaling and overflow policy".

### ReaLB

**What it does.** ReaLB targets expert-parallel load imbalance in multimodal MoE inference. It observes that vision-heavy experts/ranks produce stragglers and dynamically switches overloaded vision-heavy ranks to lower-precision execution. It uses current routing state, not historical prediction, and hides precision conversion overhead under dispatch.

**Threat.** Medium. It is a runtime system paper, not expert budget allocation, but it validates that modality composition and vision-heavy routing are major causes of MoE inference inefficiency.

**Use for SCEB.** Cite for the systems reality that multimodal MoE routing is modality-skewed and dynamic. SCEB should report MoE block latency / dispatch stats if claiming real speedup.

### BEAM

**What it does.** BEAM learns a binary activation mask over the standard top-k expert candidates. It adds a mask router, binarizes masks with a threshold, trains with a straight-through estimator and L1 sparsity loss, and deploys with custom CUDA/vLLM kernels.

**Threat.** Medium. It strongly covers "token-adaptive expert masking", but it needs training/SFT and is mainly LLM MoE rather than omnimodal.

**Use for SCEB.** Useful to justify mask/compact-dispatch as a hardware-friendly representation. SCEB should not imitate "trainable binary mask" language.

### SMIDT

**What it does.** SMIDT is a serving framework for MoE models with dynamic Top-K routing. It observes that dynamic top-k creates layer-wise workload variation that breaks uniform pipeline partitioning. It introduces fine-grained module-level partitioning, DPMoE replication, and search for PP/TP/EP/DP strategies.

**Threat.** Low to method novelty; medium to systems claims. It assumes a dynamic top-k policy exists and optimizes serving around it.

**Use for SCEB.** If SCEB moves beyond mask-only into compact dispatch, SMIDT is a useful implementation reference. It also implies we must report layer-wise top-m distribution because dynamic top-m affects serving partitioning.

### Harder Tasks Need More Experts

**What it does.** Text-only dynamic routing. It activates experts until cumulative router probability exceeds a threshold, arguing hard inputs need more experts. It requires model training with a dynamic loss and shows layer-wise expert needs vary.

**Threat.** Low-medium. It is not multimodal and not post-hoc, but it is the core conceptual predecessor for "fixed top-k is inefficient".

**Use for SCEB.** Cite as text-only dynamic routing precedent. Use it to motivate why uniform top-k is a bad baseline, not as the main competitor.

### OmniZip

**What it does.** Training-free audio-guided audio-video token compression for Qwen2.5-Omni. It selects salient audio tokens, computes per-window audio retention, and uses audio retention to dynamically guide video token pruning. It shows audio-video tasks are more pruning-sensitive than pure video and that audio should often be pruned less than video.

**Threat.** Medium for AV-SCEB, low for expert-budget SCEB. It works on tokens, not experts.

**Use for SCEB.** Strong support for handoff branch B1/B2: audio should first be a guard/controller for video expert budget. The "listen-to-prune" idea maps cleanly to "listen-to-budget-video-experts".

### OmniSIFT

**What it does.** Modality-asymmetric token compression. It first prunes video using spatial/temporal redundancy, then uses pruned visual anchors to select audio tokens with a trained vision-guided audio selector. It adds about 4.85M parameters and fine-tunes on audio-video caption data.

**Threat.** Medium for direct audio compression, low for current video-only SCEB.

**Use for SCEB.** It warns that audio importance is context-dependent and may require cross-modal learned guidance. This supports the handoff warning: do not make audio expert budgeting P0.

### OmniSelect

**What it does.** Training-free query/modality-aware token compression. It uses AudioCLIP to estimate video-query and audio-query relevance, chooses among audio-centric, video-centric, and uniform pruning regimes, then prunes within temporal groups with adaptive ratios.

**Threat.** Medium for AV-SCEB naming, low for expert-budget SCEB.

**Use for SCEB.** Directly supports handoff B3: query-level regime should control video expert budget. But SCEB should emphasize that its control object is expert count per token-layer, not token retention.

## Implications For The Existing Handoff

### What To Keep

The handoff's conservative two-line strategy is still right:

- Main A: pure video-only expert budgeting / dynamic expert skipping.
- Main B: audio-aware guard/controller as extension, not the main bet.

But the naming and novelty language need to change.

### What To Change

Replace broad titles like:

```text
Structured Expert Budgeting for Efficient Omnimodal MoE Inference
Router-Guided Dynamic Expert Budgeting for Omnimodal MoE Inference
```

with narrower titles like:

```text
Post-hoc Router-Risk Expert Skipping for Frozen Omnimodal MoE Video Inference
Router-Boundary Risk Calibration for Expert Skipping in Frozen Omnimodal MoE
Training-Free Video-Region Expert Skipping for Frozen Audio-Video MoE Models
```

### What To Add To Experiments

The current A0-A5 experiments are necessary but not sufficient after MoDES / AnyExperts. Add the following:

| New Experiment | Why |
|---|---|
| MoDES-style local probability threshold baseline | Reviewers will expect comparison against local-router expert skipping |
| MoDES-style layer-calibrated score `alpha_l * pi_i` if feasible | Tests whether SCEB's router-boundary rule is actually better than layer KL calibration |
| Text/vision or audio/video dual-threshold baseline | Separates SCEB's protected-modality policy from generic dual-modality thresholding |
| No-calibration vs calibration study | Important differentiation: SCEB should be valuable if it avoids offline KL/frontier search |
| OmniZip-inspired audio guard for video top-m | Safe AV extension without touching audio expert budget |
| Query-regime guard from OmniSelect | Safer than claiming audio expert budgeting works |

### What Not To Claim

Do not claim:

- "first dynamic expert budget for multimodal MoE";
- "first training-free expert skipping for MLLM";
- "omnimodal MoE field is unoccupied";
- "audio can be expert-budgeted" before audio diagnostics;
- "mask-only latency is real compact-dispatch speedup" without MoE block and dispatch evidence.

## Revised Research Story

The strongest revised story is:

```text
Existing dynamic expert allocation methods either require training / architecture changes
(AnyExperts, BEAM), or use calibrated expert skipping for vision-language MLLMs (MoDES).
We instead study a stricter deployment setting: a frozen omnimodal audio-video MoE model,
where audio/text must be protected and only video-heavy regions are budgeted.

We show that expert redundancy is not just modality-level; it is localized to video token-layer
pairs with stable router boundaries. A simple router-boundary risk rule can preserve quality
while reducing expert compute, and audio can serve as a guard for video expert budgets rather
than a pruning target.
```

## Server Autoresearch Addendum

Priority order after this related-work review:

1. Keep current Vanilla / Global SCEB anchors.
2. Run fixed video-middle top-4/top-6 and answer-flip metrics.
3. Add MoDES-style baselines:
   - local router probability threshold;
   - optional layer-calibrated `alpha_l * pi_i`;
   - modality-specific thresholding.
4. Run signal complementarity: `p1-p2`, boundary only, both.
5. Run modality-stage probe with audio protected.
6. Only then test AV extensions:
   - audio salience guard for video min top-m;
   - query-level regime for video expert budget.

Decision rule:

```text
If SCEB does not beat MoDES-style thresholding on flip/accuracy-latency
under the same average active experts, do not push it as a paper novelty.
Instead frame it as a deployment-safe variant or internal optimization.
```
