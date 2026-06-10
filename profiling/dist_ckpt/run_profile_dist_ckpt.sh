#!/usr/bin/env bash
# Profile Megatron dist checkpoint save/load with Qwen3-0.6B model.
#
# This script follows the same approach as verl's stream trainer profiling:
# - Uses torchrun for multi-GPU distributed setup
# - Supports nsys profiling via ENABLE_NSYS_PROFILE
# - Configures NCCL for 4090-style nodes (no NVLink/P2P)
#
# Usage:
#   bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
#
# Or with nsys:
#   ENABLE_NSYS_PROFILE=1 bash profiling/dist_ckpt/run_profile_dist_ckpt.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/../coding-verl/.venv}"

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B-hf}"
CKPT_DIR="${CKPT_DIR:-/workspace/model/dist_ckpt_profile}"
NUM_ITERS="${NUM_ITERS:-3}"
SKIP_OPTIMIZER="${SKIP_OPTIMIZER:-1}"
TP="${TP:-1}"
PP="${PP:-1}"
DP="${DP:-}"
CACHE_STRUCTURE="${CACHE_STRUCTURE:-0}"
THREAD_COUNT="${THREAD_COUNT:-}"
VALIDATE_INTEGRITY="${VALIDATE_INTEGRITY:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

ENABLE_NSYS_PROFILE="${ENABLE_NSYS_PROFILE:-0}"
NSYS_PROFILE_STEPS="${NSYS_PROFILE_STEPS:-[1]}"
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-${CKPT_DIR}/nsys_profile}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
export VERL_FORCE_NO_TE_MCORE=1
export PYTHONPATH=/workspace/runtime_shims/no_te_mcore:${PYTHONPATH:-}

if [[ -d "${VENV_DIR}" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

PYTHON_SITE="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
NVIDIA_LIB_PATH="$(
  find "${PYTHON_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -
)"
if [[ -n "${NVIDIA_LIB_PATH}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
fi
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"

NSYS_PREFIX=""
if [[ "${ENABLE_NSYS_PROFILE}" == "1" || "${ENABLE_NSYS_PROFILE}" == "true" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "ENABLE_NSYS_PROFILE=1 but nsys not found" >&2
    exit 1
  fi
  mkdir -p "${NSYS_OUTPUT_DIR}"
  NSYS_PREFIX="nsys profile --trace=cuda,nvtx --output=${NSYS_OUTPUT_DIR}/profile_dist_ckpt --force-overwrite=true"
  echo "NSYS profiling enabled: output=${NSYS_OUTPUT_DIR}"
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "NUM_ITERS=${NUM_ITERS}"
echo "SKIP_OPTIMIZER=${SKIP_OPTIMIZER}"
echo "TP=${TP} PP=${PP}"
echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"

DP_ARG=""
if [[ -n "${DP}" ]]; then
  DP_ARG="--data-parallel-size ${DP}"
fi

THREAD_COUNT_ARG=""
if [[ -n "${THREAD_COUNT}" ]]; then
  THREAD_COUNT_ARG="--thread-count ${THREAD_COUNT}"
fi

cd "${PROJECT_DIR}"
mkdir -p "${CKPT_DIR}"

${NSYS_PREFIX} torchrun \
  --nproc_per_node="${N_GPUS_PER_NODE}" \
  profiling/dist_ckpt/profile_dist_ckpt.py \
  --model-path "${MODEL_PATH}" \
  --ckpt-dir "${CKPT_DIR}" \
  --num-iters "${NUM_ITERS}" \
  --skip-optimizer "${SKIP_OPTIMIZER}" \
  --tensor-model-parallel-size "${TP}" \
  --pipeline-model-parallel-size "${PP}" \
  ${DP_ARG} \
  --cache-structure "${CACHE_STRUCTURE}" \
  ${THREAD_COUNT_ARG} \
  --validate-integrity "${VALIDATE_INTEGRITY}"