# Omni-MoE-Pruner

This repository preserves OmniDial expert-budget configurations and inference
implementations for **Qwen3-Omni-30B-A3B-Instruct** and **UniMoE 2.0 Omni**.
The model-specific directories below contain frozen configuration files and
the source snapshots used by the archived runs.

## Configuration provenance

The complete retained cross-benchmark configurations use Qwen's all-layer
scope `[0,48)` and UniMoE's partial scope `[7,19)`. Layer intervals are
zero-based and exclude the upper endpoint. These are recorded operating
points, without a benchmark-independent claim of optimality.

The [scope-study archive](experiments/omnidial_scope_20260929/README.md)
records that the exact thresholds of the earlier main-table configurations
were deleted. Its retained gates were recalibrated on eight input-only videos.
Consequently, this checkout cannot guarantee exact reconstruction of those
historical "best" points. The directories below preserve the complete later
configurations, including their actual thresholds.

## Qwen3-Omni configurations and inference

Application scope: `[0,48)`; backend: `megatron_core_grouped_mlp`.
The retained gate is `tau_reduce = 0.08694529702904027`.
The gate's recorded calibration scope is `[12,32)`; the application scope is
specified separately in `arms.json` and `protocol.json`.

| Benchmark | Configuration directory | Frozen inference script |
|---|---|---|
| AVUT | [qwen_all_avut/](experiments/omnidial_remaining_20261006/qwen_all_avut/) | [source/run.py](experiments/omnidial_remaining_20261006/qwen_all_avut/source/run.py) |
| WorldSense | [qwen_all_worldsense/](experiments/omnidial_remaining_20261006/qwen_all_worldsense/) | [source/run.py](experiments/omnidial_remaining_20261006/qwen_all_worldsense/source/run.py) |

The AVUT configuration files are:

- [protocol.json](experiments/omnidial_remaining_20261006/qwen_all_avut/protocol.json): checkpoint, backend, scope, inputs, generation and measurement settings.
- [arms.json](experiments/omnidial_remaining_20261006/qwen_all_avut/arms.json): the `omnidial` arm with `{"kind":"cgcc","scope":[0,48]}`.
- [gates.json](experiments/omnidial_remaining_20261006/qwen_all_avut/gates.json): frozen gate and calibration provenance.
- [manifest.json](experiments/omnidial_remaining_20261006/qwen_all_avut/manifest.json): hashes of the frozen configuration and source files.

The corresponding WorldSense files are
[protocol.json](experiments/omnidial_remaining_20261006/qwen_all_worldsense/protocol.json),
[arms.json](experiments/omnidial_remaining_20261006/qwen_all_worldsense/arms.json),
[gates.json](experiments/omnidial_remaining_20261006/qwen_all_worldsense/gates.json), and
[manifest.json](experiments/omnidial_remaining_20261006/qwen_all_worldsense/manifest.json).

The AVUT source snapshot includes the Qwen runtime implementation:

- [qwen_runtime_backend.py](experiments/omnidial_remaining_20261006/qwen_all_avut/source/qwen_runtime_backend.py)
- [qwen_mcore_controller.py](experiments/omnidial_remaining_20261006/qwen_all_avut/source/qwen_mcore_controller.py)
- [qwen_mcore_experts.py](experiments/omnidial_remaining_20261006/qwen_all_avut/source/qwen_mcore_experts.py)
- [scope_study_backend.py](experiments/omnidial_remaining_20261006/qwen_all_avut/source/scope_study_backend.py)

The WorldSense directory preserves its own copies of these files.

## UniMoE configurations and inference

Application scope: `[7,19)`; backend: `unimoe_native_deepspeed`.
The retained gates are:

```json
{
  "tau_skip": 0.6786869187104074,
  "tau_hard": 0.2749494430140445,
  "tau_null": 0.4352393838167196
}
```

Their recorded calibration scope is `[0,28)`; the application scope is
specified separately in `arms.json` and `protocol.json`.

| Benchmark | Configuration directory | Frozen inference script |
|---|---|---|
| AVUT | [unimoe_partial_avut/](experiments/omnidial_remaining_20261006/unimoe_partial_avut/) | [source/run.py](experiments/omnidial_remaining_20261006/unimoe_partial_avut/source/run.py) |
| WorldSense | [unimoe_partial_worldsense/](experiments/omnidial_remaining_20261006/unimoe_partial_worldsense/) | [source/run.py](experiments/omnidial_remaining_20261006/unimoe_partial_worldsense/source/run.py) |

The AVUT configuration files are:

- [protocol.json](experiments/omnidial_remaining_20261006/unimoe_partial_avut/protocol.json): checkpoint, backend, scope, inputs, generation and measurement settings.
- [arms.json](experiments/omnidial_remaining_20261006/unimoe_partial_avut/arms.json): the `omnidial` arm with `{"kind":"cgcc","scope":[7,19]}`.
- [gates.json](experiments/omnidial_remaining_20261006/unimoe_partial_avut/gates.json): frozen gates and calibration provenance.
- [manifest.json](experiments/omnidial_remaining_20261006/unimoe_partial_avut/manifest.json): hashes of the frozen configuration and source files.
- [source/todo3_backend.py](experiments/omnidial_remaining_20261006/unimoe_partial_avut/source/todo3_backend.py): UniMoE adapter and budget controller.

The corresponding WorldSense files are
[protocol.json](experiments/omnidial_remaining_20261006/unimoe_partial_worldsense/protocol.json),
[arms.json](experiments/omnidial_remaining_20261006/unimoe_partial_worldsense/arms.json),
[gates.json](experiments/omnidial_remaining_20261006/unimoe_partial_worldsense/gates.json),
[manifest.json](experiments/omnidial_remaining_20261006/unimoe_partial_worldsense/manifest.json), and
[source/todo3_backend.py](experiments/omnidial_remaining_20261006/unimoe_partial_worldsense/source/todo3_backend.py).

Frozen keys such as `cgcc` retain their original names for reproducibility.

## OmniVideoBench scope-study archives

| Model | Configuration directory | Frozen inference script |
|---|---|---|
| Qwen3-Omni | [qwen_full1000_v2/](experiments/omnidial_scope_20260929/qwen_full1000_v2/) | [source/run_scope_study_v2.py](experiments/omnidial_scope_20260929/qwen_full1000_v2/source/run_scope_study_v2.py) |
| UniMoE | [unimoe_full1000_v2/](experiments/omnidial_scope_20260929/unimoe_full1000_v2/) | [source/run_scope_study_v2.py](experiments/omnidial_scope_20260929/unimoe_full1000_v2/source/run_scope_study_v2.py) |

Each directory retains `protocol.json`, `gates.json`, `manifest.json`,
`inputs.json`, `source/`, and `calibration/`. The archived launcher is
[launch.sh](experiments/omnidial_scope_20260929/launch.sh); its interpreter,
GPU and filesystem paths are specific to the original worker setup.

## Running the archived entry points

The cross-benchmark `source/run.py` scripts accept `--out <run-directory>`
and optional `--recover-lock`. The selected directory supplies the frozen
protocol, gates, arms, inputs and manifest. The runner verifies those hashes
before loading the model. Completed directories are analyzed again instead
of starting another inference pass.

Restore the recorded model, data and software environment before resuming a
run. The snapshots contain absolute paths and enforce environment checks;
a checkout in a different location needs a separately prepared run. Preserve
the frozen source and outputs when preparing new experiments.

- Qwen checkpoint location: `models/Qwen3-Omni-30B-A3B-Instruct/`.
  The cross-benchmark backend requires the recorded Megatron-Core grouped-MLP
  runtime. The repository also retains [qwen-omni-utils/src/qwen_omni_utils/](qwen-omni-utils/src/qwen_omni_utils/).
- UniMoE checkpoint location: `models/Uni-MoE-2.0-Omni/`.
  The adapter additionally needs the external UniMoE implementation and
  DeepSpeed environment. Archived worker paths are
  `/home/tiger/Uni-MoE/Uni-MoE-2` and `/home/tiger/uni_moe_venv/bin/python`;
  these paths were absent during this inventory. A local, Git-ignored runtime
  archive is `.runtime/uni_scope_env_20260929.tar.zst`.
- Model weights, environments and dataset bodies are kept outside Git.
  See [DATA_STORAGE.md](docs/DATA_STORAGE.md) and
  [data_inventory.json](docs/data_inventory.json) for the shared dataset layout.

The top-level historical scripts
`scripts/run_qwen_mcore_crossbench.py`, `scripts/run_qwen_crossbench_clean.py`,
and `scripts/run_mb_new2_cgskip_unimoe2.py` still contain standalone references
to the absent `research_paper/` workspace. Use the frozen entries above as
the configuration and source index; those historical CLIs require additional
records before standalone execution.

File presence, Git tracking, frozen manifest hashes and Python syntax were
checked for the listed bundles. GPU inference was not rerun for this inventory.
