# Environment and Data Reproduction

This document describes how to reproduce the current OmniZip/Qwen3-Omni evaluation environment on another machine, and how to download the model weights and datasets that are intentionally not tracked by Git.

## Current Status

The repository is not currently a lockfile-based `uv sync` project:

- There is no root-level `pyproject.toml` or `uv.lock`.
- `lmms-eval/` and `qwen-omni-utils/` each have their own packaging metadata.
- The current runtime uses a project-local `.venv`.
- `eval.sh` calls `${PROJECT_ROOT}/.venv/bin/accelerate`.
- `omnizip/` is imported from source through `PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}` rather than installed as a package.

Use `uv` as the fastest way to create the virtual environment and install packages, but treat the setup as `uv venv` + `uv pip install`, not `uv sync`.

## Path Contract

On every machine, keep this layout:

```text
OmniZip/
  .venv/
  eval_data/
    huggingface/
      videomme/
        data/
        subtitle/
    AVUTBenchmark/
    OmniVideoBench/
    ShortVid-Bench/
    WorldSense/
  models/
    Qwen3-Omni-30B-A3B-Instruct/
    Qwen2.5-Omni-7B/
```

Most scripts resolve paths relative to `PROJECT_ROOT`. From the repo root:

```bash
export PROJECT_ROOT="$(pwd)"
export HF_HOME="${PROJECT_ROOT}/eval_data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
```

If the new machine has a large data disk elsewhere, use symlinks while preserving the repo-relative paths:

```bash
mkdir -p /data/omnizip
ln -s /data/omnizip/eval_data "${PROJECT_ROOT}/eval_data"
ln -s /data/omnizip/models "${PROJECT_ROOT}/models"
```

Create the symlinks before downloading. The code should still see `models/...` and `eval_data/...` inside the repo.

## Environment Setup

Recommended baseline: Linux, NVIDIA GPU, CUDA-compatible PyTorch, Python 3.11. The current machine uses Python 3.11.2.

Install or confirm `uv`:

```bash
uv --version
```

Create the environment:

```bash
cd OmniZip
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel packaging ninja
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install \
  transformers==4.57.3 \
  accelerate==1.12.0 \
  datasets==5.0.0 \
  huggingface-hub==0.36.0 \
  hf_transfer \
  numpy==1.26.4 \
  av==15.1.0 \
  decord==0.6.0 \
  moviepy==2.2.1 \
  librosa==0.11.0 \
  soundfile==0.13.1 \
  pillow==11.3.0 \
  scikit-learn==1.7.2 \
  scipy==1.15.3 \
  opencv-python-headless \
  loguru \
  tqdm \
  jsonlines \
  pyyaml \
  sentencepiece \
  qwen-vl-utils
```

Install the local editable packages:

```bash
uv pip install -e qwen-omni-utils
uv pip install -e lmms-eval
```

Install FlashAttention last:

```bash
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

If `flash-attn` build fails, first verify that `torch` imports and matches the CUDA runtime on the machine:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
PY
```

## Hugging Face Login

Some datasets or models may require an accepted license or authenticated download.

```bash
hf auth login
hf auth whoami
```

Do not commit tokens. If a token is needed in a script, pass it through the environment:

```bash
export HF_TOKEN="..."
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
```

## Download Model Weights

Create the model directory:

```bash
mkdir -p "${PROJECT_ROOT}/models"
```

Download the Qwen3-Omni model used by `eval.sh` and profile scripts:

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir "${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct" \
  --max-workers 8
```

Optional, for the Qwen2.5 demo:

```bash
hf download Qwen/Qwen2.5-Omni-7B \
  --local-dir "${PROJECT_ROOT}/models/Qwen2.5-Omni-7B" \
  --max-workers 8
```

The default paths expected by this repo are:

```text
models/Qwen3-Omni-30B-A3B-Instruct
models/Qwen2.5-Omni-7B
```

Override at runtime only if needed:

```bash
PRETRAINED=/path/to/Qwen3-Omni-30B-A3B-Instruct bash eval.sh
```

## Download Datasets

Create data directories:

```bash
mkdir -p "${PROJECT_ROOT}/eval_data" "${HF_HOME}"
```

### Video-MME

`lmms-eval` expects Video-MME media at:

```text
${HF_HOME}/videomme/data
${HF_HOME}/videomme/subtitle
```

With the path contract above, that is:

```text
eval_data/huggingface/videomme/data
eval_data/huggingface/videomme/subtitle
```

Download directly into that directory:

```bash
hf download lmms-lab/Video-MME \
  --repo-type dataset \
  --local-dir "${HF_HOME}/videomme" \
  --max-workers 8
```

### OmniVideoBench

Used by `profile_omnivideobench_qwen3`:

```bash
hf download NJU-LINK/OmniVideoBench \
  --repo-type dataset \
  --local-dir "${PROJECT_ROOT}/eval_data/OmniVideoBench" \
  --max-workers 8
```

Expected files include:

```text
eval_data/OmniVideoBench/data.json
eval_data/OmniVideoBench/data.parquet
```

### AVUTBenchmark

Used by `profile_avut_qwen3`:

```bash
hf download tsinghua-ee/AVUTBenchmark \
  --repo-type dataset \
  --local-dir "${PROJECT_ROOT}/eval_data/AVUTBenchmark" \
  --max-workers 8
```

Expected files include:

```text
eval_data/AVUTBenchmark/AV_Human_data.json
eval_data/AVUTBenchmark/AV_Gemini_data.json
```

### ShortVid-Bench

```bash
hf download TencentARC/ShortVid-Bench \
  --repo-type dataset \
  --local-dir "${PROJECT_ROOT}/eval_data/ShortVid-Bench" \
  --max-workers 8
```

Expected file:

```text
eval_data/ShortVid-Bench/ShortVid-Bench-1k.json
```

### WorldSense

```bash
hf download honglyhly/WorldSense \
  --repo-type dataset \
  --local-dir "${PROJECT_ROOT}/eval_data/WorldSense" \
  --max-workers 8
```

Expected file:

```text
eval_data/WorldSense/worldsense_qa.json
```

## Quick Verification

Check imports:

```bash
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python - <<'PY'
import torch
import transformers
import lmms_eval
import qwen_omni_utils
import omnizip
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("lmms_eval ok")
print("qwen_omni_utils ok")
print("omnizip source import ok")
PY
```

Check required paths:

```bash
test -d models/Qwen3-Omni-30B-A3B-Instruct
test -f models/Qwen3-Omni-30B-A3B-Instruct/config.json
test -d eval_data/huggingface/videomme/data
test -f eval_data/OmniVideoBench/data.json
test -f eval_data/AVUTBenchmark/AV_Human_data.json
```

Run a small smoke eval:

```bash
EVAL_LIMIT=2 TASKS=videomme bash eval.sh
```

## Running With Non-Default Paths

Prefer keeping symlinks under the repo. If you must use external paths without symlinks:

```bash
export PROJECT_ROOT="$(pwd)"
export HF_HOME="/data/omnizip/eval_data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PRETRAINED="/data/omnizip/models/Qwen3-Omni-30B-A3B-Instruct"
bash eval.sh
```

For profile scripts, pass explicit paths:

```bash
python profile_avut_qwen3/run_avut_profile.py \
  --model-path "${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct" \
  --data-path "${PROJECT_ROOT}/eval_data/AVUTBenchmark/AV_Human_data.json" \
  --video-dir "${PROJECT_ROOT}/eval_data/AVUTBenchmark"

python profile_omnivideobench_qwen3/run_omnivideobench_profile.py \
  --model-path "${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct" \
  --data-path "${PROJECT_ROOT}/eval_data/OmniVideoBench/data.json" \
  --video-dir "${PROJECT_ROOT}/eval_data/OmniVideoBench"
```

## Notes

- `eval_data/`, `models/`, `logs/`, `.venv/`, and cache directories are intentionally ignored by Git.
- Expect large downloads. Current local sizes are roughly:
  - `models/Qwen3-Omni-30B-A3B-Instruct`: 66 GB
  - `models/Qwen2.5-Omni-7B`: 21 GB
  - `eval_data/huggingface/videomme`: 94 GB
  - `eval_data/OmniVideoBench`: 54 GB
  - `eval_data/AVUTBenchmark`: 24 GB
  - `eval_data/WorldSense`: 17 GB
  - `eval_data/ShortVid-Bench`: 12 GB
- If a download is interrupted, rerun the same `hf download ... --local-dir ...` command; it should resume/reuse existing files.
- Avoid creating a top-level directory named `videomme` in the repo root. `eval.sh` checks for this because it can shadow the `lmms-eval` task name.
