#!/usr/bin/env bash
# Run ms-swift's official Megatron-SWIFT LoRA recipe for Qwen3-Omni MoE.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${MODEL_PATH:-${project_root}/models/Qwen3-Omni-30B-A3B-Instruct}"
data_dir="${DATA_DIR:-${project_root}/ft_data}"
# Keep every dataset cache on the persistent workspace mount, never under the
# worker's ephemeral system-disk home.  MS-Swift stores local-JSONL Arrow and
# AddLengthPreprocessor caches under ModelScope (not Hugging Face), so both
# roots must be fixed for later arms to reuse this expensive preprocessing.
persistent_cache_root="${LORA_CACHE_ROOT:-${data_dir}/processed_cache}"
persistent_hf_cache_root="${LORA_HF_CACHE_ROOT:-${persistent_cache_root}/huggingface}"
# `modelscope.hub.utils.get_cache_dir()` treats MODELSCOPE_CACHE itself as its
# cache root.  The cache copied from the worker's default location is rooted at
# `.../modelscope/hub`, so preserve that final `hub` component here; otherwise
# MS-Swift looks in a distinct `.../modelscope/datasets` tree and repeats the
# expensive multimodal AddLength map.
persistent_modelscope_cache_root="${LORA_MODELSCOPE_CACHE_ROOT:-${persistent_cache_root}/modelscope/hub}"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
lora_venv="${LORA_VENV:-${runtime_root}/venv}"
run_name="${RUN_NAME:-omnivideo_100k_qwen3_omni_lora}"
# Keep the adapter checkpoints on the persistent workspace volume.  The
# temporary worker runtime is intentionally used only for the environment.
output_dir="${OUTPUT_DIR:-${project_root}/models/${run_name}}"
expert_model_parallel_size="${EXPERT_MODEL_PARALLEL_SIZE:-2}"
finetune="${FINETUNE:-true}"
add_version="${ADD_VERSION:-true}"
adapters="${ADAPTERS:-}"
mcore_adapter="${MCORE_ADAPTER:-}"
external_plugins="${EXTERNAL_PLUGINS:-}"
no_load_optim="${NO_LOAD_OPTIM:-false}"
no_load_rng="${NO_LOAD_RNG:-false}"
no_save_optim="${NO_SAVE_OPTIM:-true}"
no_save_rng="${NO_SAVE_RNG:-true}"
learning_rate="${LEARNING_RATE:-1e-4}"
min_learning_rate="${MIN_LEARNING_RATE:-1e-5}"
lr_warmup_fraction="${LR_WARMUP_FRACTION:-0.05}"
seed="${SEED:-42}"
data_seed="${DATA_SEED:-${seed}}"
max_steps="${MAX_STEPS:-}"

dataset_path="${TRAIN_DATASET:-${data_dir}/omnivideo_100k_lora.jsonl}"
val_dataset_path="${VAL_DATASET:-}"
split_dataset_ratio="${SPLIT_DATASET_RATIO:-0.001}"
# With global batch 16 a step contains four smoke-test-sized accumulation
# windows.  Saving every 50 steps gives a durable adapter roughly every
# 5--6 hours, rather than risking the worker lifetime with a 1,000-step save.
eval_steps="${EVAL_STEPS:-500}"
save_steps="${SAVE_STEPS:-50}"
save_total_limit="${SAVE_TOTAL_LIMIT:-2}"
logging_steps="${LOGGING_STEPS:-5}"
global_batch_size="${GLOBAL_BATCH_SIZE:-16}"
use_distributed_optimizer="${USE_DISTRIBUTED_OPTIMIZER:-true}"
# Activation-recompute + fusion knobs.  Defaults preserve the historical recipe
# exactly (full/uniform/1, rope fusion off); overridable for speed-memory tuning
# on workers with spare HBM.  Recompute is a numerically exact speed<->memory
# trade-off; changing it does not alter training numerics.
recompute_granularity="${RECOMPUTE_GRANULARITY:-full}"
recompute_method="${RECOMPUTE_METHOD:-uniform}"
recompute_num_layers="${RECOMPUTE_NUM_LAYERS:-1}"
apply_rope_fusion="${APPLY_ROPE_FUSION:-false}"
if [[ "${1:-}" == "--smoke" ]]; then
  dataset_path="${data_dir}/omnivideo_100k_lora_smoke.jsonl"
  val_dataset_path=""
  output_dir="${output_dir}_smoke"
  split_dataset_ratio="0.125"
  eval_steps=1
  save_steps=2
  logging_steps="${LOGGING_STEPS:-1}"
fi

if [[ ! -d "${model_path}" ]]; then
  printf 'Missing model directory: %s\n' "${model_path}" >&2
  exit 1
fi
if [[ ! -f "${dataset_path}" ]]; then
  printf 'Missing prepared dataset: %s\n' "${dataset_path}" >&2
  exit 1
fi
if [[ -n "${val_dataset_path}" && ! -f "${val_dataset_path}" ]]; then
  printf 'Missing validation dataset: %s\n' "${val_dataset_path}" >&2
  exit 1
fi
if [[ ! -x "${lora_venv}/bin/python" ]]; then
  printf 'Missing LoRA environment: %s\n' "${lora_venv}" >&2
  exit 1
fi

export PATH="${lora_venv}/bin:${PATH}"
ffmpeg_path="$("${lora_venv}/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
export PATH="$(dirname "${ffmpeg_path}"):${PATH}"
if ! command -v megatron >/dev/null; then
  printf 'Megatron-SWIFT CLI is unavailable in %s\n' "${lora_venv}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
mkdir -p "${persistent_hf_cache_root}/datasets" "${persistent_hf_cache_root}/hub" \
  "${persistent_hf_cache_root}/assets"
mkdir -p "${persistent_modelscope_cache_root}/datasets/map_cache"
export HF_HOME="${persistent_hf_cache_root}"
export HF_DATASETS_CACHE="${persistent_hf_cache_root}/datasets"
export HF_HUB_CACHE="${persistent_hf_cache_root}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_ASSETS_CACHE="${persistent_hf_cache_root}/assets"
export MODELSCOPE_CACHE="${persistent_modelscope_cache_root}"
printf 'Persistent Hugging Face cache: %s\n' "${persistent_hf_cache_root}"
printf 'Persistent ModelScope dataset cache: %s\n' "${persistent_modelscope_cache_root}"

# The worker's CUDA 12.6 tree contains a self-referential `cuda-12.6`
# symlink.  Transformer Engine recursively scans CUDA paths at import time,
# so preload the CUDA 12.8 libraries bundled with PyTorch and point its three
# probes at those small package directories.  This avoids the system-tree
# traversal and keeps all user-space CUDA libraries version-consistent.
python_purelib="$("${lora_venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
nvidia_lib_root="${python_purelib}/nvidia"
cuda_lib_components=(
  nvjitlink cusparse cusolver cudnn cublas cuda_runtime cuda_nvrtc curand
  cufft nccl cusparselt cufile cuda_cupti nvtx
)
cuda_lib_paths=()
for component in "${cuda_lib_components[@]}"; do
  candidate="${nvidia_lib_root}/${component}/lib"
  if [[ -d "${candidate}" ]]; then
    cuda_lib_paths+=("${candidate}")
  fi
done
for required_component in cudnn cuda_nvrtc curand; do
  if [[ ! -d "${nvidia_lib_root}/${required_component}/lib" ]]; then
    printf 'Missing PyTorch CUDA library directory: %s\n' \
      "${nvidia_lib_root}/${required_component}/lib" >&2
    exit 1
  fi
done
cuda_python_ld_library_path="$(IFS=:; printf '%s' "${cuda_lib_paths[*]}")"
export LD_LIBRARY_PATH="${cuda_python_ld_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDNN_HOME="${nvidia_lib_root}/cudnn"
export NVRTC_HOME="${nvidia_lib_root}/cuda_nvrtc"
export CURAND_HOME="${nvidia_lib_root}/curand"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# Mirrors modelscope/ms-swift/examples/megatron/multimodal/omni/moe.sh.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT:-0}"
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ "${1:-}" == "--check-env" ]]; then
  "${lora_venv}/bin/python" -c \
    'import decord, flash_attn, swift, torch, transformer_engine.pytorch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "swift": swift.__version__, "decord": decord.__version__})'
  megatron sft --help >/dev/null
  printf 'Qwen3-Omni LoRA runtime check passed.\n'
  exit 0
fi

resume_args=()
if [[ -n "${adapters}" && -n "${mcore_adapter}" ]]; then
  printf 'ADAPTERS and MCORE_ADAPTER are mutually exclusive.\n' >&2
  exit 2
fi
if [[ -n "${adapters}" ]]; then
  resume_args+=(
    --adapters "${adapters}"
    --no_load_optim "${no_load_optim}"
    --no_load_rng "${no_load_rng}"
  )
fi
if [[ -n "${mcore_adapter}" ]]; then
  # Unlike PEFT `--adapters` (a weight-only warm start), MCore adapter
  # checkpoints can restore iteration, optimizer, scheduler, and RNG state.
  resume_args+=(
    --mcore_adapter "${mcore_adapter}"
    --no_load_optim "${no_load_optim}"
    --no_load_rng "${no_load_rng}"
  )
fi
if [[ -n "${external_plugins}" ]]; then
  # Megatron-SWIFT accepts a list of plugin files. A colon-separated
  # environment variable keeps the launcher interface shell-safe while
  # allowing the single-GPU optimizer shim and ARC-SCEB patch to load together.
  IFS=':' read -r -a external_plugin_paths <<< "${external_plugins}"
  resume_args+=(--external_plugins "${external_plugin_paths[@]}")
fi
val_dataset_args=()
if [[ -n "${val_dataset_path}" ]]; then
  val_dataset_args+=(--val_dataset "${val_dataset_path}")
fi
train_iters_args=()
num_train_epochs_args=(--num_train_epochs 1)
if [[ -n "${max_steps}" ]]; then
  # Megatron-SWIFT uses train_iters rather than Hugging Face's max_steps.
  # Supplying num_train_epochs as well would overwrite the explicit cap in
  # MegatronSftArguments.init_iters, so the two modes are mutually exclusive.
  train_iters_args+=(--train_iters "${max_steps}")
  num_train_epochs_args=()
fi

# Build recompute args.  When granularity is "none"/"off"/empty, omit the flags
# entirely (Megatron rejects recompute_method/num_layers without an active
# granularity).  Otherwise pass the historical or overridden triplet.
recompute_args=()
case "${recompute_granularity,,}" in
  ""|none|off|disable|disabled)
    : ;;  # no activation recompute -> fastest, uses more HBM
  selective)
    recompute_args+=(--recompute_granularity selective) ;;
  *)
    recompute_args+=(
      --recompute_granularity "${recompute_granularity}"
      --recompute_method "${recompute_method}"
      --recompute_num_layers "${recompute_num_layers}"
    ) ;;
esac

exec megatron sft \
  --model "${model_path}" \
  --save_safetensors true \
  --merge_lora false \
  --dataset "${dataset_path}" \
  "${val_dataset_args[@]}" \
  --load_from_cache_file true \
  --seed "${seed}" \
  --data_seed "${data_seed}" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --sequence_parallel true \
  --packing true \
  --freeze_llm false \
  --freeze_vit true \
  --freeze_aligner true \
  --split_dataset_ratio "${split_dataset_ratio}" \
  --expert_model_parallel_size "${expert_model_parallel_size}" \
  --moe_permute_fusion true \
  --moe_grouped_gemm true \
  --moe_shared_expert_overlap true \
  --moe_aux_loss_coeff 1e-3 \
  --micro_batch_size 1 \
  --global_batch_size "${global_batch_size}" \
  --use_distributed_optimizer "${use_distributed_optimizer}" \
  "${recompute_args[@]}" \
  --apply_rope_fusion "${apply_rope_fusion}" \
  --finetune "${finetune}" \
  --cross_entropy_loss_fusion true \
  --lr "${learning_rate}" \
  --lr_warmup_fraction "${lr_warmup_fraction}" \
  --min_lr "${min_learning_rate}" \
  "${num_train_epochs_args[@]}" \
  "${train_iters_args[@]}" \
  --output_dir "${output_dir}" \
  --add_version "${add_version}" \
  --eval_steps "${eval_steps}" \
  --save_steps "${save_steps}" \
  --save_total_limit "${save_total_limit}" \
  --logging_steps "${logging_steps}" \
  --max_length 4096 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 8 \
  --no_save_optim "${no_save_optim}" \
  --no_save_rng "${no_save_rng}" \
  --gradient_accumulation_fusion false \
  --attention_backend flash \
  "${resume_args[@]}"
