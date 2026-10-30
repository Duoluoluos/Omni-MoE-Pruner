# Remaining OmniDial experiments — 2026-10-06

Two H20 lanes on worker `921134cf872ab4e8`; current supervisor PID is in
`supervisor.pid` and `queue.json` (the initial supervisor was PID 116421).
Live state: `queue.json`; initial verified background handoff: `handoff.json`.
Administrative repair and resumed generation check: `handoff_repaired_20261006.json`.

| GPU | Current task | Then |
|---|---|---|
| 0 | Qwen all [0,48), AVUT 1,733 | Qwen all WorldSense 3,170 |
| 1 | Uni partial [7,19), AVUT 1,733 | Uni partial WorldSense 3,170; partial Alloc-MoE + ASET on OVB 1,000; Uni [16,27), 968 new + 32 reused |

Each inference directory has a frozen manifest, inputs, source snapshots and
copied September 29 OmniDial gates. Predictions are one fsynced JSONL line per
complete question, so large runs do not require thousands of checkpoint files.
Only a torn final line can be recovered; duplicate indices and source/gate drift
are rejected. Per-session modality checks and warmup precede measurements.
No full historical Native, fixed-budget or completed scope run is repeated.

WorldSense's historical exclusion list was deleted. Header/first-AV-frame and
64-frame full-span input-only audits reject the corrupt video ZbktCcGp.mp4
(questions 2293/2294), retaining 3,170 of 3,172 questions, versus the historical
3,167. The unknown additional historical exclusions are not guessed. AVUT
retains the documented exclusion 1495, giving 1,733. Input audit records are
`*_media_check.json`. No evaluation labels enter inference or calibration.

`queue.py` adopts the initial AVUT PIDs and runs the remaining tasks sequentially
per GPU. After a complete run, `analyze.py` independently joins labels and verifies
coverage, frozen hashes, preflight and finite outputs. `backfill.py` rechecks it,
serializes Main Table/CSV/prose/TODO updates, and runs `make review`. Errors are
reported in `queue.json`, per-run failure records and backfill/build logs; an
incomplete run never fills a table cell. Separate tasks can continue after a
failure. There is no external publication or notification.

Administrative restart repair: both `queue.lock` and `paper_backfill.lock`
live under worker-local `/tmp/omnidial_remaining_20261006_locks/`. The shared
mount returned `ENOSYS` for `flock` during the initial backfill; supervisors,
backfills and any overlapping paper editor must use the same worker-local
paper lock. This is single-worker coordination, not a cross-host lock.
Progress reads retry briefly and retain their last state across transient
replace/read races. A verified full summary and an exited matching inference
process suffice for completion even if progress is stale or missing. Restart
adopts children only after checking their exact script and `--out` arguments,
rejects duplicate children, and retries missing/failed completed-run backfills
up to three times per supervisor. Prior failure logs and state history remain.
Backfill completion records bind the successful build to its raw data, summary
and administrative helper hashes; frozen run sources/protocols are unchanged.

Uni's 32 terminal pilot predictions are eligible for reuse only after the same
model/package/source/weight checks, input digest and two policy-output checks.
Their old timings are excluded from the new continuation's latency summary.
Baseline input caches use worker-local `/tmp` and the retained environment is
restored from `.runtime/uni_scope_env_20260929.tar.zst`; shared weights/data are
not duplicated. `PYTHONDONTWRITEBYTECODE=1` limits shared cache-file growth.

Already integrated: verified localized probes in Figure 4, bounded explanations,
and the exact terminal pilot appendix table. See the paper's
`build/followup_integration_validation_20261006.json` for compile/numeric checks.
