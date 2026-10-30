# Missing-scope results — 2026-09-30

Both suites completed: three new scopes × 1,000 questions per backbone, plus 32 unique-video diagnostics. Frozen hashes, complete-question coverage, ledger identity, accuracy and mean generation times were independently checked. Result status: verified within the stated protocol.

| Backbone | New scope | Accuracy (%) | Generation (s) |
|---|---|---:|---:|
| qwen | [0,20) | 38.10 | 13.364 |
| qwen | [28,48) | 39.50 | 13.598 |
| qwen | [0,48) | 37.60 | 14.183 |
| unimoe | [0,12) | 37.60 | 2.840 |
| unimoe | [7,19) | 38.80 | 2.851 |
| unimoe | [16,28) | 37.30 | 2.886 |

Accuracy uses all 1,000 questions; latency excludes the first five warmup questions. All arms are OmniDial. Qwen uses the existing HF backend; Uni uses the existing DeepSpeed backend.

Historical manuscript anchors remain unchanged: Qwen [12,32) 39.68% / 13.151 s; Uni all-layer 38.2% / 2.693 s. These were reused, not rerun. New gates were calibrated from input-only data; comparisons with historical anchors are descriptive, not paired.

Within the new Qwen arms, late scope has the highest observed accuracy (39.5%). Late minus all-layer accuracy is +1.9 pp, with a paired 95% CI of [-0.63, 4.46] pp. Its mean latency is 0.585 s lower. On the 32-video diagnostic subset, all-layer prefill routed rows are 77.52% of Native, but all-layer inference is slower than both new partial arms: logical work savings do not establish speedup in this HF implementation.

Uni partial scope is executable and completes the full benchmark. Middle scope has the highest observed accuracy (38.8%); middle minus late is +1.5 pp, 95% CI [-0.81, 3.74] pp. This does not establish a statistically superior scope or preference over the historical all-layer point.

Across all new within-model scope contrasts, the accuracy-difference CIs include zero. Correct-option log-probability does distinguish Qwen late from early/all and Uni early/middle from late. These pointwise contrasts do not establish a uniquely optimal scope.

The 32-video branch probes do not show that shared experts alone preserve Native behavior: Uni routed-output zeroing changes 87.5% of answers, versus 12.5% for shared-output zeroing. Both are prefill-only Native-weight output interventions, separate from practical budget changes and timing.

TODO: integrate scope/quality/work/branch figures and table into the manuscript, preserving provenance; scope-width sweep and tighter matched-work/gate-transfer controls remain deferred. No paper metrics were changed in this progress check.

Operational note: both experiment processes have exited after complete summaries. Current GPU utilization could not be read because nvidia-smi reports an NVML-library error.
