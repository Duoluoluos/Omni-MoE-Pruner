PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PROJECT_ROOT
export HF_HOME="${PROJECT_ROOT}/eval_data/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"

# Safety check: lmms-eval interprets `--tasks <name>` as a directory if a
# folder with that name exists in the cwd (it then globs *.yaml from it).
# Make sure no stray dataset-cache folder named like one of our tasks lives
# in the project root, otherwise task selection silently produces an empty list.
for task_name in videomme videomme_long videomme_w_subtitle videomme_long_w_subtitle; do
    if [ -d "${PROJECT_ROOT}/${task_name}" ] && [ ! -f "${PROJECT_ROOT}/${task_name}/videomme.yaml" ]; then
        echo "ERROR: stray directory '${PROJECT_ROOT}/${task_name}' shadows lmms-eval task '${task_name}'."
        echo "       Move or remove this directory before running eval."
        exit 1
    fi
done

# Or export WRAPPER=None
export WRAPPER="${WRAPPER:-OmniZip}"
OMNIZIP_RHO_AUDIO="${OMNIZIP_RHO_AUDIO:-0.3}"
OMNIZIP_RHO_VIDEO="${OMNIZIP_RHO_VIDEO:-0.6}"
OMNIZIP_G="${OMNIZIP_G:-3}"
OMNIZIP_CONTEXTUAL_RATIO="${OMNIZIP_CONTEXTUAL_RATIO:-0.05}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-768}"
TASKS="${TASKS:-videomme}"

LIMIT_ARG=""
if [ -n "${EVAL_LIMIT:-}" ]; then
    LIMIT_ARG="--limit $EVAL_LIMIT"
fi

PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/models/Qwen3-Omni-30B-A3B-Instruct}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12347}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-./logs/}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PROJECT_ROOT}/.venv/bin/accelerate" launch --num_processes=1 --main_process_port="${MAIN_PROCESS_PORT}" -m lmms_eval \
    --model qwen3_omni \
    --model_args "pretrained=${PRETRAINED},attn_implementation=flash_attention_2,max_num_frames=${MAX_NUM_FRAMES},OMNIZIP_RHO_AUDIO=${OMNIZIP_RHO_AUDIO},OMNIZIP_RHO_VIDEO=${OMNIZIP_RHO_VIDEO},OMNIZIP_G=${OMNIZIP_G},OMNIZIP_CONTEXTUAL_RATIO=${OMNIZIP_CONTEXTUAL_RATIO}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    ${LIMIT_ARG} \
    --output_path "${EVAL_OUTPUT_PATH}"

# You can use for other benchmarks in lmms-eval
