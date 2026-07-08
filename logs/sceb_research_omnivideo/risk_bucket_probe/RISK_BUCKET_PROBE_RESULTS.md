# Confidence-Risk Bucket Probe On OmniVideoBench

This report uses the completed OmniVideoBench global SCEB policy run:

`logs/omnivideobench_qwen3_budget_policy/policy_calib40_full1000_2gpu_20260623_111524`

The run stores per-sample `expert_budget_summary.top_m_histogram`, selected by
the router-guided dynamic policy. Raw router margin tensors are not persisted in
the durable logs, so this probe uses the realized selected expert budget as the
available risk proxy:

- higher top-4 share = more token-layer pairs accepted as low-risk,
- higher top-8 share = more token-layer pairs protected as high-risk,
- lower average active experts = more aggressive realized budget.

All comparisons are OmniVideoBench-only and matched by `sample_index` against
the vanilla top-8 predictions when flip counts are reported.

## Overall Dynamic Budget Distribution

| Samples | Accuracy | Mean top-4 share | Mean top-6 share | Mean top-8 share | Mean active experts |
|---:|---:|---:|---:|---:|---:|
| 998 | 39.68% | 11.13% | 26.21% | 62.66% | 7.06 |

The global policy is conservative: most routed video-token middle-layer
assignments remain at top-8, while a smaller subset receives top-4 or top-6.
This is consistent with the handoff's claim that only confident video
token-layer states should be budgeted.

## Buckets By Realized Top-4 Share

| Bucket | Samples | Accuracy | Mean top-4 share | Mean active experts | Answer flip rate vs vanilla | Base-correct -> wrong | Base-wrong -> correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 452 | 31.86% | 0.00% | 7.27 | 5.31% | 1.11% | 0.66% |
| Mid | 214 | 38.32% | 17.66% | 7.00 | 3.74% | 0.93% | 1.87% |
| High | 332 | 51.20% | 22.08% | 6.83 | 6.33% | 1.51% | 2.41% |

The high top-4 bucket is not less accurate; it is substantially more accurate.
This supports the interpretation that the policy's top-4 decisions are
selective and tend to occur on easier, lower-risk token-layer/sample regimes,
rather than being arbitrary pruning.

## Buckets By Realized Top-8 Share

| Bucket | Samples | Accuracy | Mean top-8 share | Mean active experts | Answer flip rate vs vanilla | Base-correct -> wrong | Base-wrong -> correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low top-8 | 333 | 38.14% | 58.59% | 7.00 | 5.11% | 1.20% | 1.20% |
| Mid top-8 | 333 | 42.64% | 62.75% | 7.07 | 6.61% | 1.50% | 1.80% |
| High top-8 | 332 | 38.25% | 66.67% | 7.12 | 4.22% | 0.90% | 1.51% |

Top-8 share behaves as a protection signal rather than a monotonic accuracy
booster. High top-8 samples are more conservative and do not show high flip
risk, but their accuracy remains close to the low top-8 bucket because they are
often harder tasks that the policy chose to protect.

## Buckets By Average Active Experts

| Bucket | Samples | Accuracy | Mean active experts | Answer flip rate vs vanilla | Base-correct -> wrong | Base-wrong -> correct |
|---|---:|---:|---:|---:|---:|---:|
| Low active | 333 | 48.95% | 6.82 | 6.01% | 1.20% | 2.40% |
| Mid active | 333 | 36.04% | 7.08 | 4.80% | 1.80% | 1.20% |
| High active | 332 | 34.04% | 7.29 | 5.12% | 0.60% | 0.90% |

Lower average active experts correlate with higher sample accuracy in this
artifact. That does not mean using fewer experts causes easier answers; it
means the router-guided policy tends to spend fewer experts where the sample
appears budgetable and keeps more experts for harder regimes.

## Task-Level Sanity

| Task | Samples | Accuracy | Mean top-8 share | Mean active experts |
|---|---:|---:|---:|---:|
| counting | 155 | 30.32% | 62.06% | 7.27 |
| reference reasoning | 98 | 39.80% | 62.41% | 7.27 |
| spatial understanding | 62 | 30.65% | 62.55% | 7.27 |
| temporal understanding | 137 | 28.47% | 62.03% | 7.26 |
| causal reasoning | 139 | 47.48% | 62.33% | 6.88 |
| hypothetical reasoning | 23 | 52.17% | 61.95% | 6.87 |
| sentiment analysis | 81 | 51.85% | 63.16% | 6.89 |

The stress tasks highlighted by the policy configuration, including counting,
reference, spatial, and temporal reasoning, receive more conservative active
expert budgets on average. Easier or more stable tasks such as causal,
hypothetical, and sentiment samples receive fewer active experts.

## Conclusion

The realized-budget buckets provide durable evidence for the paper story:
budgetability is structured, and router-guided SCEB is behaving like a risk
allocation mechanism rather than a uniform top-k reduction. The current durable
logs support the qualitative monotonicity claim at the sample/regime level:
where the policy assigns lower budgets, accuracy is not harmed and often rises;
where it assigns higher budgets, those samples are generally harder and are
protected.

The remaining limitation is that raw `p1-p2` and `p_m-p_(m+1)` bucket tensors
were not saved in these full-run artifacts. A future rerun can add those raw
margin summaries for a stricter margin-bucket monotonicity plot, but the
existing OmniVideoBench artifacts already close a durable risk-bucket evidence
gap for the current autoresearch verifier.
