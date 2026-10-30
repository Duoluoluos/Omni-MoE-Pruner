# Layer follow-up progress — 2026-10-05

Both September 30 runs completed successfully, before the old worker stopped.
Qwen completed at 2026-09-30 15:12:29 CST; Uni at 14:59:22 CST. Each has
32 committed question records, complete raw coverage and a verified summary.
Independent checks today passed frozen file hashes, raw/question identity,
input-reference identity, session preflights, all diagnostic means and work totals.

## Localized branch-removal probes (32 videos; not full-set accuracy)

| Model / branch removed | Early Δ gold log p / flip | Middle Δ gold log p / flip | Late Δ gold log p / flip |
|---|---:|---:|---:|
| Qwen routed | -6.392 / 96.875% | +0.078 / 65.625% | -0.703 / 71.875% |
| Uni routed | -5.460 / 100% | +0.267 / 78.125% | -0.856 / 62.5% |
| Uni shared | +0.034 / 15.625% | -0.018 / 3.125% | +0.004 / 3.125% |

Early routed removal has the largest correct-option log-probability loss in
both checkpoints. Middle-band mean log probability is more stable, yet many
answers change; this does not establish safely removable layers. Shared-output
removal changes fewer answers than routed-output removal on this subset.
These are extreme Native-weight prefill output probes, not OmniDial policy
accuracy, a shared-only compensation test, or timing evidence.

## Uni terminal-layer protection pilot

With identical gates, protecting layer 27 changes scope [16,28) to [16,27).
Correct answers: 8/32 (25.00%) → 10/32 (31.25%), a gain of two questions.
Gold-option mean log probability increases by 0.7195 versus the archived late
scope. Actual routed prefill rows rise from 10,820,908 to 10,919,792 (+0.9138%),
or 90.5104% → 91.3375% of the same Native total. Shared prefill work is unchanged.
This is a positive pilot signal worth full-1000 continuation; it is not eligible
for a full-benchmark Main Table cell. No full-set latency claim is inferred.

## Remaining work and resource state

- P0-A localized probes and the P0-B pilot: verified; figure integration pending.
- Uni [16,27) full extension: planned; reuse 32 predictions, infer remaining 968.
- Six Main Table gaps / five job groups: 0 launched, still planned; there was no
  automatic follow-up queue after the completed diagnostics.
- SSH proxy now reports that worker ed8cf8975c804c34 is not running. A currently
  running worker is needed before more inference can start.
- Project quota initially prevented saving this check. Author-authorized cleanup
  on October 5 released 246.43 MiB; directory creation, writing, fsync and atomic
  rename now pass. This report and the TODO are saved. The GPU worker remains
  stopped; neither completed inference run failed because of this quota issue.

Machine-readable independent check: completion_check_20261005.json.
