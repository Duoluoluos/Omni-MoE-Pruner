# OmniZip × OmniDial on Qwen3-Omni: theory and experimental protocol

## Executive hypothesis

OmniZip and OmniDial act on different axes of Qwen3-Omni inference:

- **OmniZip** reduces the multimodal sequence length before the thinker transformer by pruning/merging audio and video tokens.
- **OmniDial** preserves the sequence but reduces the number of routed expert rows for selected tokens in MoE layers 12–31, using the frozen input-only confidence threshold.

If the retained-token ratio is `r_L` and the retained expert-row ratio conditional on a token is `r_E`, the routed-MLP work in the overlap should scale approximately as `r_L × r_E`, not as `r_L + r_E - 1`. Attention and encoder work respond only to OmniZip; routing, dispatch, grouped GEMM and fixed launch costs limit wall-clock multiplicativity.

## Expected interaction

### Compute and memory

1. **Routed expert FLOPs / rows: positive composability.** OmniZip removes rows before they reach every MoE layer; OmniDial reduces experts per remaining row. The combined routed work should therefore be close to the product of individual work ratios.
2. **Attention FLOPs: OmniZip-only.** For prefill attention, reducing sequence length yields roughly quadratic savings in the thinker attention term; OmniDial does not affect it.
3. **KV cache: OmniZip-only.** Fewer retained multimodal tokens reduce cache footprint; expert-budget reduction does not.
4. **Peak weight memory: mostly unchanged.** All expert weights remain resident. Activation/temporary memory can fall, but 30B model weights dominate peak allocation.
5. **Latency: sub-multiplicative relative to ideal compute.** Audio/video encoders, router computation, output head, Python control, kernel launches and decode are fixed or weakly affected. MCore GroupedMLP minimizes the dispatch/GEMM overhead that would otherwise hide OmniDial gains.

### Quality

Quality interaction is not guaranteed to be independent:

- **Potential antagonism:** OmniZip changes hidden-state and router distributions. A gate threshold calibrated on native sequences may reduce too many experts after compression.
- **Potential protection:** token compression removes redundant/noisy multimodal evidence, making surviving token routing more confident.
- **Required test:** keep the OmniDial threshold frozen. Recalibrating on OmniZip outputs would confound composability with retuning.

## Frozen 2×2 design

| Arm | OmniZip | OmniDial |
|---|---:|---:|
| Native | off | off |
| OmniZip | on | off |
| OmniDial | off | on, layers `[12,32)` |
| Stack | on | on, layers `[12,32)` |

Fixed settings:

- Qwen3-Omni-30B-A3B-Instruct, BF16, greedy generation, batch size 1, max 4 thinker tokens.
- OmniZip: audio pruning ratio 0.3, video pruning ratio 0.6, `g=3`, contextual ratio 0.05.
- OmniDial: CGCC top-4/top-8 policy, frozen `tau_reduce=0.08694529702904027`, layers 12–31.
- 32 preselected OmniVideoBench videos, 8 per archived input-length quartile; two-sample smoke first.
- Four arms share one resident model and one MCore expert conversion. Arm order is deterministically randomized within each sample.

## Metrics and interaction estimands

For metric `Y`, with `00=Native`, `10=OmniZip`, `01=OmniDial`, `11=Stack`:

- Additive interaction: `I_add = Y11 - Y10 - Y01 + Y00`.
- Efficiency multiplicative residual: `I_mul = (Y11/Y00) - (Y10/Y00)(Y01/Y00)`.
- For cost metrics, `I_mul ≈ 0` supports multiplicative composability; `<0` is super-multiplicative saving and `>0` is interference/fixed-overhead loss.

Primary measurements: paired accuracy, gold option log-probability, full-vocabulary JS to Native, answer flips, synchronized generation latency, retained tokens, MCore executed expert rows and peak allocated GPU memory. Bootstrap confidence intervals resample paired questions.

## Megatron infra use

This validation uses Megatron-Core `GroupedMLP` for every Qwen thinker MoE layer while retaining the original Qwen multimodal encoders, attention and KV-cache path. It provides:

- stable expert grouping and grouped GEMM for all four arms;
- identical expert backend between Native and compressed conditions;
- real executed-row accounting rather than theoretical top-k estimates;
- one model load and one expert conversion for randomized within-sample comparisons.

It is an expert-backend integration, not a claim that the full Qwen3-Omni generation stack has been ported to Megatron-Core.
