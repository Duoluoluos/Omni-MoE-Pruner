# Loop8 Audio-Guard v4 Results

This report summarizes the measured Audio-Guard v4 run on the 200-row
OmniVideoBench audio-salient subset. The limit-200 run completed on worker
3955069 with policy version `sceb_loop8_audio_guard_v4`.

## Artifacts

| Artifact | Path |
|---|---|
| Policy JSON | `logs/omnivideobench_qwen3_research/policy_variants_20260709/loop8_audio_guard_v4_policy.json` |
| Smoke output | `logs/omnivideobench_qwen3_research/loop8_audio_guard_v4_20260709/smoke2` |
| Limit-200 output | `logs/omnivideobench_qwen3_research/loop8_audio_guard_v4_20260709/limit200` |
| Candidate analysis | `logs/sceb_research_omnivideo/loop8/AUDIO_GUARD_V4_CANDIDATES.md` |
| Policy note | `logs/sceb_research_omnivideo/loop8/AUDIO_GUARD_POLICY_V4.md` |

Smoke2 completed with 2 predictions and 50.00% accuracy. Limit-200 completed
with 200 predictions, 200 profile rows, and a 200-sample summary.

## Matched Accuracy And Budget

Rows below use the same 200-row audio-salient subset for v4, v3, and v2. The
global, boundary-only, and vanilla rows are the matched anchors from prior
Loop4/Loop7 reports.

| Method | Matched | Accuracy | Correct | Mean active experts | Top-4 | Top-6 | Top-8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio-Guard v4 | 200 | 41.50% | 83/200 | 6.92 | 18.01% | 25.83% | 56.15% |
| Audio-Guard v2 | 200 | 41.50% | 83/200 | 7.38 | 5.09% | 25.64% | 69.27% |
| Loop7 speed v3 | 200 | 39.50% | 79/200 | 6.90 | 18.84% | 25.15% | 56.01% |
| Global SCEB | 200 | 42.50% | 85/200 | 7.04 | 13.27% | 22.08% | 64.65% |
| Boundary-only SCEB | 200 | 41.50% | 83/200 | 5.84 | 40.38% | 31.15% | 28.47% |
| Vanilla top-8 | 200 | 42.00% | 84/200 | n/a | n/a | n/a | n/a |

V4 meets the useful-success criterion. It recovers all four samples lost by
Loop7 speed v3 relative to Audio-Guard v2, while preserving almost the same
effective expert budget distribution as v3. It does not beat Global SCEB or
Vanilla on matched accuracy, so it should be presented as an AV-SCEB family
member rather than as the global default.

## Churn And Wrong Movement

Direct churn is available for v2 and v3 because all three runs use the same
subset order and sample indices.

| Source -> Audio-Guard v4 | Matched | Flip rate | Source correct -> v4 wrong | Source wrong -> v4 correct |
|---|---:|---:|---:|---:|
| Audio-Guard v2 -> v4 | 200 | 3.50% | 2 | 2 |
| Loop7 speed v3 -> v4 | 200 | 2.50% | 0 | 4 |

The v3-to-v4 movement is the key result: v4 produces four wrong-to-correct
recoveries and no correct-to-wrong losses against v3. Against v2, v4 is a
budget-improved tie: two regressions and two recoveries, ending at the same
83/200 total.

## Latency And Effective Budget

The implementation remains mask-only, so wall-clock speed is not expected to
track effective expert assignment perfectly.

| Method | Prefill mean | Prefill median | Prefill p95 | Generation / E2E mean | Mean active experts |
|---|---:|---:|---:|---:|---:|
| Audio-Guard v4 | 9,330 ms | 10,002 ms | 14,810 ms | 12,200 ms | 6.92 |
| Audio-Guard v2 | 9,078 ms | 9,633 ms | 14,297 ms | 11,720 ms | 7.38 |
| Loop7 speed v3 | 9,162 ms | 9,625 ms | 14,497 ms | 11,788 ms | 6.90 |

V4 improves the effective expert budget relative to v2, but it does not improve
wall-clock latency in this mask-only run. The correct interpretation is
therefore budget-policy evidence, not realized systems speedup. Compact
dispatch remains the required systems step for wall-clock gains.

## Task-Regime Results

| Task type | Rows | Accuracy | Correct | Mean active experts | Top-4 / Top-6 / Top-8 |
|---|---:|---:|---:|---:|---|
| background&music understanding | 47 | 34.04% | 16/47 | 8.00 | 0.0 / 0.0 / 100.0% |
| counting | 28 | 42.86% | 12/28 | 6.42 | 28.5 / 23.3 / 48.2% |
| causal reasoning | 20 | 50.00% | 10/20 | 6.44 | 27.8 / 23.2 / 49.0% |
| temporal understanding | 18 | 22.22% | 4/18 | 6.91 | 0.0 / 55.4 / 44.6% |
| reference reasoning | 15 | 40.00% | 6/15 | 6.94 | 0.0 / 54.2 / 45.8% |
| sentiment analysis | 15 | 60.00% | 9/15 | 6.42 | 28.5 / 23.2 / 48.3% |
| summarization | 12 | 58.33% | 7/12 | 6.43 | 27.8 / 23.2 / 49.1% |
| ego reasoning | 10 | 40.00% | 4/10 | 6.96 | 0.0 / 52.7 / 47.3% |
| fine-grained perception | 10 | 20.00% | 2/10 | 6.43 | 27.3 / 23.3 / 49.4% |
| relationship reasoning | 9 | 55.56% | 5/9 | 6.41 | 28.5 / 23.4 / 48.2% |
| hypothetical reasoning | 8 | 75.00% | 6/8 | 6.46 | 27.2 / 23.1 / 49.7% |
| spatial understanding | 5 | 40.00% | 2/5 | 6.96 | 0.0 / 53.2 / 46.8% |
| attribute comparison | 3 | 0.00% | 0/3 | 6.52 | 26.2 / 22.5 / 51.3% |

The policy behavior matches the design. Audio-core rows remain full top-8.
Temporal, reference, spatial, and ego rows avoid top-4 and use a top-6/top-8
mix. Lower-risk rows keep the v3 speed distribution with roughly 27-29% top-4
assignment share.

## Interpretation

Audio-Guard v4 is a successful guard-restored policy family member. It gives a
clearer audio-guard story than v3 because it restores fragile AV floors and
late-middle protection, and it gives a clearer budget story than v2 because it
keeps v3-like top-4/top-8 shares on lower-risk rows.

The main limitation is that v4 still trails the Global SCEB matched anchor by
two samples and the Vanilla matched anchor by one sample. The paper should not
claim that v4 is the best quality policy overall. The stronger claim is:

```text
Audio-conditioned guards can recover the quality loss of a speed-oriented AV
budget policy while preserving most of its effective expert-budget reduction.
```

For long-paper positioning, v4 should be the recommended Audio-Guard policy
family member for AV-SCEB evidence, while Global SCEB remains the main matched
quality anchor and compact dispatch remains future systems work for wall-clock
speed.
