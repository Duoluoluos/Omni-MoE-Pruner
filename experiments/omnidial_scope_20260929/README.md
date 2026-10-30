# OmniDial missing-scope study — 2026-09-29

**Complete and verified (checked 2026-09-30).** Each backbone completed three
new scopes × 1,000 questions and 32-video diagnostics. Uni finished September 29
17:29; Qwen finished September 30 04:19 (Asia/Shanghai). Both inference processes
have exited. Current GPU utilization is unavailable due to an NVML query error.

Results: `results_progress_20260930.md`; independent validation:
`completion_check_20260930.json`; per-model metrics and CIs: `summary.json`.
`handoff.json` is the historical launch snapshot. Paper figures/table are integrated in the ACL manuscript; see its
`plan/layer_scope_followups.md` for planned targeted experiments.


Qwen paper scope is **[12,32)**. Its main-table point, 39.68% / 13.151 s,
is reused exactly. Uni-MoE all-layer 38.2% / 2.693 s is reused. Existing Native,
fixed-budget, Alloc-MoE and ASET points are not repeated.

| GPU | Runtime | New scopes | Evaluation |
|---|---|---|---|
| 0 | existing Qwen HF | [0,20), [28,48), [0,48) | 1,000 questions each |
| 1 | existing Uni DeepSpeed | [0,12), [7,19), [16,28) | 1,000 questions each |

Shared input-only diagnostic subset: 32 videos, eight per archived Uni input
length quartile. Native/branch/work instrumentation only on this subset and
preflight inputs; no full Native run. Width sweep deferred.

Exact old thresholds were deleted. The existing quantile rules are applied
once to eight input-only videos, with no answer-dependent tuning. Frozen gates
and hashes are saved before measured inference. Historical aggregates do not
provide paired contrasts, transferred CIs, or a same-session latency control.

Each model folder contains frozen protocol/inputs/source and reused table
snapshot. During execution it adds progress.json, run.log, environment.json,
gates.json, sessions/, predictions/, raw_results.jsonl, and finally summary.json.
Only atomic complete question files count toward progress. Partial questions
restart as whole three-arm blocks. Each launch rechecks its environment,
preflight and warmup; a lock rejects duplicate writers. A stale lock can be
recovered only after confirming its owning worker/process is gone.

Launch on the supplied worker: `bash experiments/omnidial_scope_20260929/launch.sh qwen`
or `.../launch.sh unimoe`. On worker replacement, restore the same environment
and launch the frozen runner with `--recover-lock` after the old worker is dead.
Do not edit frozen sources after measured output exists.

Purpose: scope quality/time/work comparison plus routed/shared branch diagnostics,
not an assumption that partial or all-layer scope wins. Raw diagnostics are
separate from timing. Full coverage and checks are required before paper updates.

Runtime restoration: `.runtime/uni_scope_env_20260929.tar.zst` contains the existing
Uni environment/code; worker `/home/tiger/uni_moe_venv` and `/home/tiger/Uni-MoE`
point into `/tmp/omnidial_runtime_20260929/`. These are worker-local and must be
restored on worker replacement. Shared experiment outputs remain persistent.
