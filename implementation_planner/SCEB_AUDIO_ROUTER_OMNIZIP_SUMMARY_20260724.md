# SCEB Audio Router and OmniZip Summary

Date: 2026-07-24

This note summarizes the current evidence for:

1. adding audio-router conditioning to video-token SCEB across three benchmarks;
2. stacking the same audio-router-conditioned SCEB policy with OmniZip on OmniVideoBench.

Latency fields use mean prefill latency and mean model-side E2E/generation latency in milliseconds. Lower latency is better.

## Main Comparison

`Global SCEB` means the no-audio-router SCEB baseline. `ARC-SCEB` means audio-router-conditioned video SCEB using the full-set policy `arc_video_boundary_audio_floor6_full_v1`.

| Benchmark | Samples | Vanilla Acc | Global SCEB Acc | ARC-SCEB Acc | ARC vs Global Acc | Vanilla E2E | Global E2E | ARC E2E | ARC vs Global E2E | ARC vs Vanilla E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniVideoBench | 998+2/1000 | 39.38% | 39.68% | 40.08% | +0.40pp | 15632.0 | 14516.0 | 11276.3 | -22.32% | -27.86% |
| AVUT | 1734/1734 | 71.16% | 70.70% | 70.18% | -0.52pp | 4263.2 | 3698.7 | 3578.3 | -3.25% | -16.07% |
| WorldSense | 3172/3172 | 49.78% | 49.84% | 49.15% | -0.69pp | 6591.4 | 5988.1 | 5699.1 | -4.83% | -13.54% |

## Prefill and Budget

| Benchmark | Vanilla Prefill | Global SCEB Prefill | ARC-SCEB Prefill | ARC vs Global Prefill | Global Active top-m | ARC Active top-m | Global Assign Red. | ARC Assign Red. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniVideoBench | 12540.2 | 11586.2 | 8710.5 | -24.82% | 7.063 | 5.541 | 11.71% | 30.74% |
| AVUT | 3409.9 | 3372.3 | 3255.0 | -3.48% | 6.895 | 5.590 | 13.81% | 30.12% |
| WorldSense | 5277.3 | 5189.8 | 4902.8 | -5.53% | 6.895 | 5.602 | 13.81% | 29.97% |

## OmniZip Stacking

Full OmniZip-only completed on OmniVideoBench. OmniZip+ARC was interrupted when worker `4065939` left the queue, so the fair comparison is the paired subset `sample_index=0..335`.

| Run | Samples | Acc | Prefill mean / p50 | E2E mean / p50 | MoE prefill | Active top-m | Assign Red. |
|---|---:|---:|---:|---:|---:|---:|---:|
| OmniZip-only, full | 998+2/1000 | 39.68% | 4680.2 / 5071.9 | 5928.9 / 6390.2 | - | - | - |
| OmniZip-only, paired 0..335 | 336 | 37.80% | 5096.6 / 6110.8 | 6475.7 / 7501.7 | 2803.4 | - | - |
| OmniZip+ARC, paired 0..335 | 336 | 39.29% | 5044.4 / 6056.1 | 6392.7 / 7444.6 | 2759.0 | 5.437 | 32.04% |

| Comparison | Delta Acc | Delta Prefill | Delta E2E | Delta MoE prefill | Prediction Stability |
|---|---:|---:|---:|---:|---:|
| OmniZip+ARC vs OmniZip-only, paired 0..335 | +1.49pp | -1.03% | -1.28% | -1.58% | 310/336 same predictions, 92.26% |

## Takeaways

- Audio-router conditioning is clearly useful on OmniVideoBench: it improves accuracy over Global SCEB and gives a large extra latency reduction.
- On AVUT and WorldSense, audio-router conditioning gives additional latency reduction over Global SCEB but costs accuracy. The drop is within about 1pp, but this should be reported as a speed-accuracy tradeoff rather than a universal win.
- The audio-router policy substantially increases assignment reduction, from about 12-14% to about 30%, and lowers active top-m from about 6.9-7.1 to about 5.5-5.6.
- OmniZip stacking is promising but not finalized: on the fair paired subset, OmniZip+ARC is slightly faster and more accurate than OmniZip-only. The gain over OmniZip-only is small, so this is best framed as compatibility/stackability evidence, not a major extra speedup claim.
- A full OmniZip+ARC resume is still needed before making a final full-set stacking claim.

## Source Files

- Vanilla OmniVideoBench: `logs/omnivideobench_qwen3_vanilla/vanilla_full_2gpu_20260625_110343/full/summary.json`
- Global SCEB OmniVideoBench: `logs/omnivideobench_qwen3_budget_policy/policy_calib40_full1000_2gpu_20260623_111524/summary.json`
- ARC full-set: `logs/audio_router_conditioned_video_sceb_full_20260720/ARC_FULLSET_ANALYSIS.md`
- AVUT fair rerun: `logs/avut_qwen3_fair_rerun_20260706/AVUT_FAIR_RERUN_RESULTS.md`
- WorldSense fair rerun: `logs/worldsense_qwen3_fair_rerun_20260707/WORLDSENSE_FAIR_RERUN_RESULTS.md`
- OmniZip stack paired subset: `logs/omnizip_sceb_stack_omnivideobench_20260721/OMNIZIP_STACK_PAIRED_336.md`
- OmniZip stack full/interrupted summary: `logs/omnizip_sceb_stack_omnivideobench_20260721/OMNIZIP_STACK_ANALYSIS.md`
