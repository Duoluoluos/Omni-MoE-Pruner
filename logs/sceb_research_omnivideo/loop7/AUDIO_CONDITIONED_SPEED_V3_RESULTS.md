# Loop7 Audio-Conditioned Speed v3 Results

This report summarizes the measured Audio-Conditioned speed policy v3 run on
the 200-row OmniVideoBench audio-salient subset. The run completed on worker
3955069 with the policy file
`logs/omnivideobench_qwen3_research/policy_variants_20260709/loop7_audio_conditioned_speed_v3_policy.json`.

## Artifacts

| Artifact | Path |
|---|---|
| Smoke output | `logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/smoke2` |
| Limit-200 output | `logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/limit200` |
| Worker launcher/logs | `logs/omnivideobench_qwen3_research/loop7_audio_conditioned_speed_v3_20260709/launcher` |
| Policy report | `logs/sceb_research_omnivideo/loop7/AUDIO_CONDITIONED_SPEED_POLICY_V3.md` |

Smoke2 completed with 2 predictions and 50.00% accuracy. The full limit-200
run completed with 200 predictions and a 200-sample summary.

## Matched Accuracy

Rows below are matched on the Loop4 audio-salient subset. For v3 and
Audio-Guard v2, subset row ids are mapped back to original OmniVideoBench
indices before comparison.

| Method | Matched | Accuracy | Correct |
|---|---:|---:|---:|
| Audio-Conditioned speed v3 | 200 | 39.50% | 79/200 |
| Audio-Guard v2 | 200 | 41.50% | 83/200 |
| Global SCEB | 200 | 42.50% | 85/200 |
| Boundary-only SCEB | 200 | 41.50% | 83/200 |
| Vanilla top-8 | 200 | 42.00% | 84/200 |
| Loop3 Audio-as-Guard overlap | 39 | 43.59% | 17/39 |
| Boundary-Guarded overlap | 39 | 38.46% | 15/39 |

The speed policy loses 4 samples versus Audio-Guard v2 and 6 samples versus
Global SCEB on the matched 200 rows. That is larger than the ideal 1-2 sample
loss target, so v3 should not be promoted as the quality default. It is still
useful long-paper speed evidence because it produces a much more aggressive
video expert budget distribution while keeping audio tokens protected.

## Churn And Wrong-Answer Movement

| Source -> v3 | Matched | Flip rate | Source correct -> v3 wrong | Source wrong -> v3 correct |
|---|---:|---:|---:|---:|
| Audio-Guard v2 | 200 | 5.00% | 5 | 1 |
| Global SCEB | 200 | 5.50% | 6 | 0 |
| Boundary-only SCEB | 200 | 5.00% | 5 | 1 |
| Vanilla top-8 | 200 | 5.00% | 6 | 1 |
| Loop3 Audio-as-Guard overlap | 39 | 0.00% | 0 | 0 |
| Boundary-Guarded overlap | 39 | 5.13% | 0 | 2 |

The wrong-answer pattern is the main caution. The policy does not merely
produce harmless answer churn; it converts several previously correct answers
to wrong. For the paper, v3 belongs in the speed-frontier section rather than
the main robustness section.

## Speed And Expert-Budget Tradeoff

| Method | Samples | Accuracy | Prefill mean | Prefill median | Prefill p95 | Generation / E2E mean | Mean active experts | Top-4 | Top-6 | Top-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Audio-Conditioned speed v3 | 200 | 39.50% | 9,162 ms | 9,625 ms | 14,497 ms | 11,788 ms | 6.90 | 18.84% | 25.15% | 56.01% |
| Audio-Guard v2 | 200 | 41.50% | 9,078 ms | 9,633 ms | 14,297 ms | 11,720 ms | 7.38 | 5.09% | 25.64% | 69.27% |
| Global SCEB full anchor | 998 | 39.68% | 11,586 ms | 12,665 ms | 19,288 ms | 15,891 ms | 7.10 | 9.61% | 26.63% | 63.76% |
| Boundary-only full anchor | 998 | 39.78% | 9,976 ms | 11,059 ms | 16,606 ms | 15,489 ms | 6.02 | 29.06% | 43.07% | 27.87% |
| Vanilla top-8 full anchor | 998 | 39.38% | 12,540 ms | 13,865 ms | 19,234 ms | - | - | - | - | - |

The v3 compute shift is substantial against Audio-Guard v2: mean active experts
drop from 7.38 to 6.90, top-4 share rises from 5.09% to 18.84%, and top-8
share drops from 69.27% to 56.01%. However, wall-clock prefill and generation
latency on this 200-row subset are roughly flat against v2. This is expected
for the current mask-only implementation: dropped expert weights are zeroed and
renormalized, but the original expert loop still runs. v3 therefore gives
measured speed-policy evidence in effective expert budget terms, while compact
dispatch is still needed for a stronger wall-clock speed win.

## Task-Regime Results

| Task type | Rows | Accuracy | Mean active experts | Prefill mean | Generation / E2E mean | Top-4 / Top-6 / Top-8 |
|---|---:|---:|---:|---:|---:|---|
| background&music understanding | 47 | 34.04% | 8.00 | 8,245 ms | 10,634 ms | 0.0 / 0.0 / 100.0% |
| counting | 28 | 42.86% | 6.42 | 7,063 ms | 8,848 ms | 28.5 / 23.3 / 48.2% |
| causal reasoning | 20 | 50.00% | 6.44 | 7,819 ms | 10,103 ms | 27.8 / 23.2 / 49.0% |
| temporal understanding | 18 | 22.22% | 6.95 | 10,076 ms | 12,997 ms | 0.0 / 53.2 / 46.8% |
| reference reasoning | 15 | 33.33% | 6.98 | 8,206 ms | 10,798 ms | 0.0 / 52.3 / 47.7% |
| sentiment analysis | 15 | 60.00% | 6.42 | 11,490 ms | 15,616 ms | 28.5 / 23.2 / 48.3% |
| summarization | 12 | 58.33% | 6.43 | 9,853 ms | 12,307 ms | 27.8 / 23.2 / 49.1% |
| fine-grained perception | 10 | 20.00% | 6.43 | 10,217 ms | 13,055 ms | 27.3 / 23.3 / 49.4% |
| ego reasoning | 10 | 20.00% | 6.48 | 12,852 ms | 16,418 ms | 27.2 / 22.3 / 50.5% |
| relationship reasoning | 9 | 55.56% | 6.41 | 7,262 ms | 9,128 ms | 28.5 / 23.4 / 48.2% |
| hypothetical reasoning | 8 | 75.00% | 6.46 | 13,500 ms | 16,858 ms | 27.2 / 23.1 / 49.7% |
| spatial understanding | 5 | 20.00% | 7.00 | 9,882 ms | 12,560 ms | 0.0 / 51.3 / 48.7% |
| attribute comparison | 3 | 0.00% | 6.52 | 14,113 ms | 18,541 ms | 26.2 / 22.5 / 51.3% |

The task breakdown confirms the controller behaved as designed:
`background&music understanding` stayed full top-8, fragile AV rows avoided
top-4, and lower-risk rows moved strongly toward top-4/top-6. The observed
accuracy losses are concentrated enough that future v4 work should restore
more guard strength for temporal, reference, spatial, and ego rows.

## Decision

Keep v3 as speed-oriented Audio-Conditioned policy evidence, not as the main
quality policy. The result is useful because it demonstrates a controllable
audio-aware speed frontier:

- audio tokens remain protected;
- audio-core rows force full video top-8;
- lower-risk audio-salient rows move toward top-4/top-6 video budgets;
- effective expert compute drops materially;
- mask-only wall-clock latency does not yet realize the full compute reduction.

For the long paper, the clean conclusion is that audio-aware SCEB can trade
quality for expert-budget reduction, and compact dispatch is the next systems
step required to turn the effective-budget win into a wall-clock speed win.
