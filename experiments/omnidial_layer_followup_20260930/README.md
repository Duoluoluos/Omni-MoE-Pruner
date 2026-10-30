# Localized layer diagnostics — September 30, 2026

Purpose: explain the layer patterns already drawn in manuscript Figures 3–4.
The original full-1000 scopes, Native references, and global branch removals are
reused. No new environment or model implementation is introduced.

| GPU | Model / process | New evaluation | Output |
|---|---|---|---|
| 0 | Qwen HF / 437481 | routed-output removal on [0,20), [12,32), [28,48), 32 videos | `qwen_diagnostic32_v1/` |
| 1 | Uni DeepSpeed / 437482 | routed/shared removal on [0,12), [7,19), [16,28); OmniDial [16,27), 32 videos | `unimoe_diagnostic32_v1/` |

The main question is whether branch dependence is localized by depth and
whether preserving Uni's last MoE layer recovers behavior at a small routed-work
cost. Strong disruption at every depth or no recovery is also informative.
No expected numeric outcome is assumed. The localized Qwen [12,32) zero-output
probe is a new diagnostic, not a rerun of the historical OmniDial policy point.

`protocol.json` and `manifest.json` freeze each run; `source/` copies the proven
parent backend. `references.json` stores the 32 Native and late-scope references,
counts and input hashes. Original parent prediction hashes are in the protocol.
`gates.json` is copied unchanged. Each model checks its shortest/longest subset
inputs against stored Native answers, text and option log probabilities, with
maximum absolute tolerance 1e-5, plus embedding and no-op checks. All other
input digests must also match. Model/source/runtime/weight metadata must match
the parent environment before inference starts.

Raw outputs are checkpointed atomically after each new arm, then after the
question. `progress.json` reports live state; `failure.json` is only written on
failure. Only after all 32 videos finish does the analyzer open labels, validate
coverage and emit `summary.json` with mean log-probability changes, answer flips,
accuracy differences and Uni routed work. Probe times are not speed evidence.
Pilot summaries are not eligible for full-benchmark Main Table cells.

The Uni 1,000-question continuation is **not queued**. Review the pilot first;
if extended, retain its 32 completed policy outputs and define the remaining
968-question schedule and a disclosed per-session timing window. Do not rerun
Native, old scope points or branch probes over the full benchmark.

## Environment and resumption

Worker: `ed8cf8975c804c34.worker_0.job.wangqi-2248@merlin-ssh-proxy-cn.byted.org:22222`.
The two GPU UUIDs are frozen in `launch.sh`; ordinal device nodes changed on the
worker but the UUIDs still identify the two H20s.

System library links pointed to zero-byte 535.261.03 targets. The active kernel
uses 535.183.06. `/tmp/omnidial_driver_20260930` supplies task-local symlinks to
the existing matching CUDA/NVML libraries through `LD_LIBRARY_PATH`.
No system library, package or driver was edited. Existing Uni environment/code
remain under `/tmp/omnidial_runtime_20260929`, reachable through `/home/tiger` links.
Caches are local `/tmp`; ffmpeg is the previously validated shared binary.

`launch.sh` refuses duplicate launch PID files. For a genuine interrupted run,
first verify the recorded process has stopped, then invoke its frozen runner
with the same interpreter/environment, `--out <run>` and `--recover-lock` if a
stale lock remains. Do not run the preparation script again over existing runs.
Committed outputs are validated, unfinished arms resumed, and Native identity
checks repeated for the new session. The inference policy never reads labels.

Main Table integration in this turn uses **completed September 29 full-set
results**, not these new probes. Provenance:
`../../research_paper/acl_arr_oct2027/results/scope_main_table_provenance.json`.
Updated plan: `../../research_paper/acl_arr_oct2027/plan/layer_scope_followups.md`.
