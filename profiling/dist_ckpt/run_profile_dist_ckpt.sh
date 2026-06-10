#!/usr/bin/env bash
# Profile Megatron dist checkpoint save/load with Qwen3-0.6B model.
#
# This script follows verl's stream trainer profiling approach and adds:
# - Fine-grained sub-operation timing via monkey-patching
# - Comparison between fully-reshardable vs DP-reshardable optimizer state
# - GPU 34 selection for DP=2 profiling
#
# Usage:
#   # Default: fully_sharded_model_space (verl's choice) with optimizer
#   bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
#
#   # Compare fully-reshardable vs DP-reshardable
#   bash profiling/dist_ckpt/run_profile_dist_ckpt.sh --compare-reshard
#
#   # DP-reshardable only
#   OPTIM_SHARDING_TYPE=dp_reshardable bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
#
#   # Skip optimizer (model-only baseline)
#   SKIP_OPTIMIZER=1 bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
#
#   # With nsys
#   ENABLE_NSYS_PROFILE=1 bash profiling/dist_ckpt/run_profile_dist_ckpt.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/../coding-verl/.venv}"

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B-hf}"
CKPT_DIR="${CKPT_DIR:-/workspace/model/dist_ckpt_profile}"
NUM_ITERS="${NUM_ITERS:-3}"
SKIP_OPTIMIZER="${SKIP_OPTIMIZER:-0}"
OPTIM_SHARDING_TYPE="${OPTIM_SHARDING_TYPE:-}"  # empty = default (fully_sharded_model_space)
TP="${TP:-1}"
PP="${PP:-1}"
DP="${DP:-}"
CACHE_STRUCTURE="${CACHE_STRUCTURE:-1}"
THREAD_COUNT="${THREAD_COUNT:-}"
VALIDATE_INTEGRITY="${VALIDATE_INTEGRITY:-0}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
GPU_IDS="${GPU_IDS:-34,35}"  # Use GPU 34 and 35 for DP=2

ENABLE_NSYS_PROFILE="${ENABLE_NSYS_PROFILE:-0}"
NSYS_PROFILE_STEPS="${NSYS_PROFILE_STEPS:-[1]}"
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-${CKPT_DIR}/nsys_profile}"

COMPARE_RESHARD="${COMPARE_RESHARD:-0}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
export VERL_FORCE_NO_TE_MCORE=1
export PYTHONPATH=/workspace/runtime_shims/no_te_mcore:${PYTHONPATH:-}

# Select specific GPUs
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

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

DP_ARG=""
if [[ -n "${DP}" ]]; then
  DP_ARG="--data-parallel-size ${DP}"
fi

THREAD_COUNT_ARG=""
if [[ -n "${THREAD_COUNT}" ]]; then
  THREAD_COUNT_ARG="--thread-count ${THREAD_COUNT}"
fi

OPTIM_SHARDING_ARG=""
if [[ -n "${OPTIM_SHARDING_TYPE}" ]]; then
  OPTIM_SHARDING_ARG="--optimizer-sharding-type ${OPTIM_SHARDING_TYPE}"
fi

run_profile() {
  local shard_type="$1"
  local shard_arg="$2"
  local ckpt_subdir="$3"
  local json_output="$4"

  echo ""
  echo "=========================================="
  echo "  Running profile: optimizer_sharding=${shard_type}"
  echo "=========================================="
  echo ""

  cd "${PROJECT_DIR}"
  mkdir -p "${ckpt_subdir}"

  local full_ckpt_dir="${ckpt_subdir}/${shard_type}"

  ${NSYS_PREFIX} torchrun \
    --nproc_per_node="${N_GPUS_PER_NODE}" \
    profiling/dist_ckpt/profile_dist_ckpt.py \
    --model-path "${MODEL_PATH}" \
    --ckpt-dir "${full_ckpt_dir}" \
    --num-iters "${NUM_ITERS}" \
    --skip-optimizer "${SKIP_OPTIMIZER}" \
    ${shard_arg} \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    ${DP_ARG} \
    --cache-structure "${CACHE_STRUCTURE}" \
    ${THREAD_COUNT_ARG} \
    --validate-integrity "${VALIDATE_INTEGRITY}" \
    --output-json "${json_output}"
}

if [[ "${COMPARE_RESHARD}" == "1" ]]; then
  echo "Comparing fully_sharded_model_space vs dp_reshardable optimizer sharding..."

  RESULTS_DIR="${CKPT_DIR}/reshard_comparison"
  mkdir -p "${RESULTS_DIR}"

  # Run 1: fully_sharded_model_space (verl default)
  run_profile "fully_sharded_model_space" \
    "--optimizer-sharding-type fully_sharded_model_space" \
    "${RESULTS_DIR}" \
    "${RESULTS_DIR}/timing_fully_sharded.json"

  # Run 2: dp_reshardable (fastest, no inter-rank communication)
  run_profile "dp_reshardable" \
    "--optimizer-sharding-type dp_reshardable" \
    "${RESULTS_DIR}" \
    "${RESULTS_DIR}/timing_dp_reshardable.json"

  # Run 3: fully_reshardable (allows TP/PP reshard but slower due to all_gather)
  run_profile "fully_reshardable" \
    "--optimizer-sharding-type fully_reshardable" \
    "${RESULTS_DIR}" \
    "${RESULTS_DIR}/timing_fully_reshardable.json"

  echo ""
  echo "=========================================="
  echo "  RESHARD COMPARISON COMPLETE"
  echo "=========================================="
  echo "  Results saved to: ${RESULTS_DIR}/"
  echo "  - timing_fully_sharded.json"
  echo "  - timing_dp_reshardable.json"
  echo "  - timing_fully_reshardable.json"
  echo ""

  # Print comparison table (on rank 0 only)
  if [[ -f "${RESULTS_DIR}/timing_fully_sharded.json" ]]; then
    echo "  Comparison (extracting key timings):"
    python3 - <<'PY_COMPARE'
import json
import os

results_dir = os.environ.get("RESULTS_DIR", "/workspace/model/dist_ckpt_profile/reshard_comparison")
types = ["fully_sharded_model_space", "dp_reshardable", "fully_reshardable"]

all_data = {}
for t in types:
    path = f"{results_dir}/timing_{t}.json"
    if os.path.exists(path):
        with open(path) as f:
            all_data[t] = json.load(f)

if not all_data:
    print("  No JSON timing files found")
    exit(0)

# Key operations to compare
key_ops = [
    "save.checkpoint_write_total",
    "save.preprocess",
    "save.common_pt",
    "save.apply_saving_parallelization",
    "save.torch_dist_save",
    "save.mcore_to_pyt_translate",
    "load.checkpoint_read_total",
    "load.fp_wrapper_load",
    "load.exchange_by_distribution",
    "build_sharded_state_dict.optimizer_fully_sharded_model_space",
    "build_sharded_state_dict.optimizer_dp_reshardable",
    "build_sharded_state_dict.optimizer_fully_reshardable",
]

print(f"\n  {'Operation':<60} {'FS_model':>10} {'DP_reshard':>10} {'Fully_reshard':>10}")
print("  " + "-" * 90)

for op in key_ops:
    vals = {}
    for t in types:
        # Find matching key (may have different suffix per type)
        matching = [k for k in all_data[t] if k.startswith(op)]
        if matching:
            vals[t] = all_data[t][matching[0]]["mean"]
        else:
            vals[t] = "-"

    fs = f"{vals.get('fully_sharded_model_space', '-'):>10}" if isinstance(vals.get('fully_sharded_model_space'), float) else f"{vals.get('fully_sharded_model_space', '-'):>10}"
    dp = f"{vals.get('dp_reshardable', '-'):>10}" if isinstance(vals.get('dp_reshardable'), float) else f"{vals.get('dp_reshardable', '-'):>10}"
    fr = f"{vals.get('fully_reshardable', '-'):>10}" if isinstance(vals.get('fully_reshardable'), float) else f"{vals.get('fully_reshardable', '-'):>10}"

    if isinstance(vals.get('fully_sharded_model_space'), float):
        fs = f"{vals['fully_sharded_model_space']:.4f}"
    if isinstance(vals.get('dp_reshardable'), float):
        dp = f"{vals['dp_reshardable']:.4f}"
    if isinstance(vals.get('fully_reshardable'), float):
        fr = f"{vals['fully_reshardable']:.4f}"

    print(f"  {op:<60} {fs:>10} {dp:>10} {fr:>10}")
PY_COMPARE
  fi

else
  # Single run with specified sharding type
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "NUM_ITERS=${NUM_ITERS}"
  echo "SKIP_OPTIMIZER=${SKIP_OPTIMIZER}"
  echo "OPTIM_SHARDING_TYPE=${OPTIM_SHARDING_TYPE:-default(fully_sharded_model_space)}"
  echo "TP=${TP} PP=${PP} DP=${DP:-auto}"
  echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "CACHE_STRUCTURE=${CACHE_STRUCTURE}"

  cd "${PROJECT_DIR}"
  mkdir -p "${CKPT_DIR}"

  JSON_OUTPUT="${CKPT_DIR}/timing_results.json"
  if [[ -n "${OPTIM_SHARDING_TYPE}" ]]; then
    JSON_OUTPUT="${CKPT_DIR}/timing_${OPTIM_SHARDING_TYPE}.json"
  fi

  ${NSYS_PREFIX} torchrun \
    --nproc_per_node="${N_GPUS_PER_NODE}" \
    profiling/dist_ckpt/profile_dist_ckpt.py \
    --model-path "${MODEL_PATH}" \
    --ckpt-dir "${CKPT_DIR}" \
    --num-iters "${NUM_ITERS}" \
    --skip-optimizer "${SKIP_OPTIMIZER}" \
    ${OPTIM_SHARDING_ARG} \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size "${PP}" \
    ${DP_ARG} \
    --cache-structure "${CACHE_STRUCTURE}" \
    ${THREAD_COUNT_ARG} \
    --validate-integrity "${VALIDATE_INTEGRITY}" \
    --output-json "${JSON_OUTPUT}"
fi