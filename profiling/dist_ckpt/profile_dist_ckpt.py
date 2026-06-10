"""Profile Megatron distributed checkpoint save and load.

This script profiles the key Megatron dist_checkpointing APIs:
  - dist_checkpointing.save()  (distributed checkpoint save)
  - dist_checkpointing.load()  (distributed checkpoint load)

It follows the approach used in verl's stream trainer profiling:
  - Timing each sub-operation with time.perf_counter()
  - Using NVTX markers for nsight profiling
  - Supporting FullyParallelSaveStrategyWrapper / FullyParallelLoadStrategyWrapper
  - Caching strategy objects across iterations
  - Optional skip of optimizer state for focused profiling

The model is initialized from an HF Qwen3-0.6B checkpoint using mbridge,
and the profiling runs multiple save/load iterations.

Usage (via torchrun):
  torchrun --nproc_per_node=2 profiling/dist_ckpt/profile_dist_ckpt.py \
    --model-path /workspace/models/Qwen3-0.6B-hf \
    --ckpt-dir /workspace/model/dist_ckpt_profile \
    --num-iters 3 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1

Or via the shell wrapper:
  bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import time
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ──────────────────────────────────────────────────
# Timing helpers (mirroring verl's observability.py)
# ──────────────────────────────────────────────────

_TIMING_RESULTS: dict[str, list[float]] = {}


@contextmanager
def record_timing(name: str, nvtx_color: str | None = None):
    """Record wall-clock time for an operation, with optional NVTX marker."""
    start = time.perf_counter()
    try:
        if nvtx_color and hasattr(torch.cuda, "nvtx_range_push"):
            torch.cuda.nvtx_range_push(name, color=nvtx_color)
        yield
    finally:
        elapsed = time.perf_counter() - start
        if nvtx_color and hasattr(torch.cuda, "nvtx_range_pop"):
            torch.cuda.nvtx_range_pop()
        _TIMING_RESULTS.setdefault(name, []).append(elapsed)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info("[rank=%d] %s: %.4f s", rank, name, elapsed)


def print_timing_summary():
    """Print a summary table of all timing results."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return

    print("\n" + "=" * 70)
    print("  DIST CHECKPOINT PROFILING SUMMARY")
    print("=" * 70)
    print(f"  {'Operation':<50} {'Mean (s)':>10} {'Std (s)':>10} {'Count':>6}")
    print("-" * 70)

    for name, times in sorted(_TIMING_RESULTS.items()):
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0.0
        print(f"  {name:<50} {mean:>10.4f} {std:>10.4f} {len(times):>6}")

    print("=" * 70 + "\n")


def aggregate_timing_across_ranks():
    """Aggregate timing results across all ranks (max across ranks like verl)."""
    if not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()
    all_results: dict[str, list[list[float]]] = {}

    # Each rank serializes its timing dict
    local_data = {name: times for name, times in _TIMING_RESULTS.items()}
    local_json = json.dumps(local_data).encode()

    # Gather all rank data
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_json)

    if torch.distributed.get_rank() != 0:
        return

    # Aggregate: take max across ranks for each operation/iteration
    print("\n" + "=" * 70)
    print("  DIST CHECKPOINT PROFILING SUMMARY (MAX ACROSS RANKS)")
    print("=" * 70)
    print(f"  {'Operation':<50} {'Max Mean (s)':>12} {'Max Std (s)':>12} {'Count':>6}")
    print("-" * 70)

    per_op_per_iter: dict[str, list[float]] = {}
    for rank_data in gathered:
        rank_dict = json.loads(rank_data)
        for name, times in rank_dict.items():
            for i, t in enumerate(times):
                key = f"{name}_iter{i}"
                per_op_per_iter.setdefault(key, []).append(t)

    for name, times in sorted(_TIMING_RESULTS.items()):
        n_iters = len(times)
        # For each iteration, take the max across ranks
        max_times = []
        for i in range(n_iters):
            key = f"{name}_iter{i}"
            max_times.append(max(per_op_per_iter.get(key, [0.0])))

        mean = sum(max_times) / len(max_times)
        std = (sum((t - mean) ** 2 for t in max_times) / len(max_times)) ** 0.5 if len(max_times) > 1 else 0.0
        print(f"  {name:<50} {mean:>12.4f} {std:>12.4f} {len(max_times):>6}")

    print("=" * 70 + "\n")


# ──────────────────────────────────────────────────
# Strategy helpers (mirroring verl's dist_checkpointing.py)
# ──────────────────────────────────────────────────

_SAVE_STRATEGY_CACHE: dict[tuple[Any, ...], Any] = {}
_LOAD_STRATEGY_CACHE: dict[tuple[Any, ...], Any] = {}


def get_save_strategy(parallelization_group, cache_structure=False, thread_count=None):
    """Build or retrieve a cached save strategy, mirroring verl's approach."""
    from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper

    cache_key = _strategy_cache_key(parallelization_group, thread_count, cache_structure, "save")
    if cache_structure and cache_key in _SAVE_STRATEGY_CACHE:
        logger.info("Using cached save strategy for key=%s", cache_key)
        return _SAVE_STRATEGY_CACHE[cache_key]

    save_strategy = get_default_save_sharded_strategy("torch_dist")
    if thread_count is not None and hasattr(save_strategy, "thread_count"):
        save_strategy.thread_count = thread_count

    if cache_structure and hasattr(save_strategy, "use_cached_ckpt_structure"):
        save_strategy.use_cached_ckpt_structure = True

    wrapped = FullyParallelSaveStrategyWrapper(
        save_strategy, parallelization_group, do_cache_distribution=cache_structure
    )

    if cache_structure:
        _SAVE_STRATEGY_CACHE[cache_key] = wrapped
        logger.info("Cached save strategy for key=%s", cache_key)

    return wrapped


def get_load_strategy(ckpt_dir, parallelization_group, cache_structure=False, thread_count=None):
    """Build or retrieve a cached load strategy, mirroring verl's approach."""
    from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
    import inspect

    cache_key = _strategy_cache_key_load(ckpt_dir, parallelization_group, thread_count, cache_structure)
    if cache_structure and cache_key in _LOAD_STRATEGY_CACHE:
        logger.info("Using cached load strategy for key=%s", cache_key)
        return _LOAD_STRATEGY_CACHE[cache_key]

    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    if thread_count is not None and hasattr(load_strategy, "thread_count"):
        load_strategy.thread_count = thread_count

    if cache_structure and hasattr(load_strategy, "use_cached_ckpt_structure"):
        load_strategy.use_cached_ckpt_structure = True

    # Build wrapper kwargs - handle do_cache_distribution if supported
    kwargs = {}
    try:
        load_wrapper_params = inspect.signature(FullyParallelLoadStrategyWrapper).parameters
    except (TypeError, ValueError):
        load_wrapper_params = {}
    if cache_structure and "do_cache_distribution" in load_wrapper_params:
        kwargs["do_cache_distribution"] = cache_structure

    wrapped = FullyParallelLoadStrategyWrapper(
        load_strategy, parallelization_group, **kwargs
    )

    if cache_structure:
        _LOAD_STRATEGY_CACHE[cache_key] = wrapped
        logger.info("Cached load strategy for key=%s", cache_key)

    return wrapped


def _strategy_cache_key(group, thread_count, cache_structure, direction):
    try:
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
    except Exception:
        ranks = id(group)
    return ("torch_dist", ranks, thread_count, cache_structure, direction)


def _strategy_cache_key_load(ckpt_dir, group, thread_count, cache_structure):
    try:
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
    except Exception:
        ranks = id(group)
    return ("torch_dist", os.path.abspath(ckpt_dir), ranks, thread_count, cache_structure)


# ──────────────────────────────────────────────────
# Model initialization via mbridge
# ──────────────────────────────────────────────────

def initialize_model_from_hf(model_path, tp_size, pp_size):
    """Initialize a Megatron model from an HF checkpoint using mbridge.

    This mirrors the flow in verl's megatron_workers.py:
    1. Create bridge from HF config via AutoBridge.from_pretrained
    2. Override attention backend to unfused, disable TE
    3. Build Megatron model via bridge.get_model()
    4. Load HF weights via bridge.load_weights()
    """
    from megatron.core import parallel_state as mpu
    from mbridge import AutoBridge

    with record_timing("init.bridge_from_hf", nvtx_color="blue"):
        bridge = AutoBridge.from_pretrained(model_path)
        # Override for unfused attention, no TE, no sequence parallel
        # (matching verl's DAPO smoke config, but use_transformer_engine
        # is not a TransformerConfig parameter in mcore 0.13.0)
        bridge.set_extra_args(
            attention_backend="unfused",
            sequence_parallel=False,
        )

    with record_timing("init.build_megatron_model", nvtx_color="blue"):
        # Build Megatron model - returns a list of model chunks (PP stages)
        model_chunks = bridge.get_model(
            bf16=True,
            wrap_with_ddp=False,
        )

    with record_timing("init.load_hf_weights", nvtx_color="blue"):
        # Load HF weights into the Megatron model
        bridge.load_weights(model_chunks, model_path)

    logger.info(
        "Initialized Megatron model from %s: tp=%d pp=%d, chunks=%d",
        model_path, tp_size, pp_size, len(model_chunks),
    )

    return model_chunks, bridge


# ──────────────────────────────────────────────────
# Checkpoint sub-operations (mirroring verl's runtime.py)
# ──────────────────────────────────────────────────

def build_sharded_state_dict(model_chunks, optimizer=None, skip_optimizer=False):
    """Build the sharded state dict for save, with timing.

    Mirrors verl's _generate_megatron_sharded_training_state_dict and
    handoff.export.build_sharded_state.
    """
    from megatron.core.dist_checkpointing import LocalNonpersistentObject

    state_dict = {}

    with record_timing("build_sharded_state_dict.model", nvtx_color="yellow"):
        for i, chunk in enumerate(model_chunks):
            chunk_for_sd = getattr(chunk, "module", chunk)
            if hasattr(chunk_for_sd, "sharded_state_dict"):
                model_sd = chunk_for_sd.sharded_state_dict()
            else:
                model_sd = chunk_for_sd.state_dict()

            # Remove LocalNonpersistentObject entries
            model_sd = {
                k: v for k, v in model_sd.items()
                if not isinstance(v, LocalNonpersistentObject)
            }
            state_dict[f"model{i}"] = model_sd

    if optimizer is not None and not skip_optimizer:
        with record_timing("build_sharded_state_dict.optimizer", nvtx_color="yellow"):
            optim_state_dict = optimizer.sharded_state_dict(state_dict)
            state_dict["optimizer"] = optim_state_dict

    return state_dict


def profile_save(state_dict, ckpt_path, save_strategy, validate_integrity=True):
    """Profile the dist checkpoint save operation.

    Mirrors verl's handoff.export.checkpoint_save.
    """
    from megatron.core import dist_checkpointing

    with record_timing("save.checkpoint_write", nvtx_color="red"):
        async_request = dist_checkpointing.save(
            state_dict,
            ckpt_path,
            sharded_strategy=save_strategy,
            async_sharded_save=False,
            validate_access_integrity=validate_integrity,
        )

    if async_request is not None:
        logger.warning("Got async save request, finalizing synchronously")
        async_request.finalize()

    return async_request


def profile_load(sharded_state_dict, ckpt_dir, load_strategy):
    """Profile the dist checkpoint load operation.

    Mirrors verl's handoff.import.checkpoint_load.
    """
    from megatron.core import dist_checkpointing

    with record_timing("load.checkpoint_read", nvtx_color="green"):
        loaded_state_dict = dist_checkpointing.load(
            sharded_state_dict,
            ckpt_dir,
            sharded_strategy=load_strategy,
        )

    return loaded_state_dict


def apply_loaded_state(model_chunks, loaded_state_dict):
    """Apply loaded state dict back to the model.

    Mirrors verl's handoff.import.apply_state.
    """
    with record_timing("load.apply_state", nvtx_color="green"):
        for i, chunk in enumerate(model_chunks):
            chunk_for_sd = getattr(chunk, "module", chunk)
            model_sd = loaded_state_dict.get(f"model{i}", {})
            if hasattr(chunk_for_sd, "load_state_dict"):
                chunk_for_sd.load_state_dict(model_sd)


# ──────────────────────────────────────────────────
# Main profiling logic
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile Megatron dist checkpoint save/load")
    parser.add_argument("--model-path", type=str, default="/workspace/models/Qwen3-0.6B-hf",
                        help="Path to HF Qwen3-0.6B model")
    parser.add_argument("--ckpt-dir", type=str, default="/workspace/model/dist_ckpt_profile",
                        help="Directory for checkpoint save/load")
    parser.add_argument("--num-iters", type=int, default=3,
                        help="Number of save/load iterations to profile")
    parser.add_argument("--skip-optimizer", type=int, default=1,
                        help="Skip optimizer state (0=no, 1=yes) - for focused profiling")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=None,
                        help="Data parallel size (computed from world_size / tp / pp if not set)")
    parser.add_argument("--cache-structure", type=int, default=0,
                        help="Cache checkpoint structure across iterations (0=no, 1=yes)")
    parser.add_argument("--thread-count", type=int, default=None,
                        help="Thread count for save/load strategy")
    parser.add_argument("--validate-integrity", type=int, default=1,
                        help="Validate access integrity (0=no, 1=yes)")
    args = parser.parse_args()

    skip_optimizer = bool(args.skip_optimizer)
    cache_structure = bool(args.cache_structure)
    validate_integrity = bool(args.validate_integrity)

    # ──────────────────────────────────────────────────
    # Initialize distributed environment
    # ──────────────────────────────────────────────────
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)

    tp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    dp_size = args.data_parallel_size or (world_size // (tp_size * pp_size))

    logger.info(
        "rank=%d local_rank=%d world_size=%d tp=%d pp=%d dp=%d",
        rank, local_rank, world_size, tp_size, pp_size, dp_size,
    )

    # ──────────────────────────────────────────────────
    # Initialize Megatron parallel state
    # ──────────────────────────────────────────────────
    with record_timing("init.parallel_state", nvtx_color="blue"):
        from megatron.core import parallel_state as mpu

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )

        logger.info(
            "Parallel state initialized: tp_rank=%d pp_rank=%d dp_rank=%d",
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            mpu.get_data_parallel_rank(),
        )

    # ──────────────────────────────────────────────────
    # Initialize model from HF checkpoint via mbridge
    # ──────────────────────────────────────────────────
    model_chunks, bridge = initialize_model_from_hf(args.model_path, tp_size, pp_size)

    # Optionally create a Megatron optimizer
    optimizer = None
    if not skip_optimizer:
        with record_timing("init.optimizer", nvtx_color="blue"):
            # Collect all model parameters for optimizer
            model_params = []
            for chunk in model_chunks:
                model_params.extend(chunk.parameters())
            optimizer = torch.optim.AdamW(model_params, lr=1e-6)

    # ──────────────────────────────────────────────────
    # Get the parallelization group for checkpointing (DP group)
    # ──────────────────────────────────────────────────
    dp_group = mpu.get_data_parallel_group(with_context_parallel=True)

    # ──────────────────────────────────────────────────
    # Run profiling iterations
    # ──────────────────────────────────────────────────
    ckpt_base_dir = args.ckpt_dir

    for iter_idx in range(args.num_iters):
        logger.info("=== Iteration %d/%d ===", iter_idx + 1, args.num_iters)

        # Clean up previous checkpoint
        iter_ckpt_dir = os.path.join(ckpt_base_dir, f"iter_{iter_idx}")
        if os.path.exists(iter_ckpt_dir):
            shutil.rmtree(iter_ckpt_dir, ignore_errors=True)
        os.makedirs(iter_ckpt_dir, exist_ok=True)

        gc.collect()
        torch.cuda.empty_cache()

        # ── Build sharded state dict (mirrors verl's handoff.export.build_sharded_state) ──
        state_dict = build_sharded_state_dict(
            model_chunks, optimizer=optimizer, skip_optimizer=skip_optimizer
        )

        # ── Get save strategy ──
        with record_timing("save.get_strategy", nvtx_color="red"):
            save_strategy = get_save_strategy(
                dp_group,
                cache_structure=cache_structure,
                thread_count=args.thread_count,
            )

        # ── Profile SAVE (mirrors verl's handoff.export.checkpoint_save) ──
        profile_save(state_dict, iter_ckpt_dir, save_strategy, validate_integrity=validate_integrity)

        logger.info("Iteration %d: SAVE completed to %s", iter_idx + 1, iter_ckpt_dir)

        # ── Get load strategy ──
        with record_timing("load.get_strategy", nvtx_color="green"):
            load_strategy = get_load_strategy(
                iter_ckpt_dir,
                dp_group,
                cache_structure=cache_structure,
                thread_count=args.thread_count,
            )

        # ── Build load template (mirrors verl's handoff.import.build_load_template) ──
        load_template = build_sharded_state_dict(
            model_chunks, optimizer=optimizer, skip_optimizer=skip_optimizer
        )

        # ── Profile LOAD (mirrors verl's handoff.import.checkpoint_load) ──
        loaded_state_dict = profile_load(load_template, iter_ckpt_dir, load_strategy)

        # ── Apply loaded state (mirrors verl's handoff.import.apply_state) ──
        apply_loaded_state(model_chunks, loaded_state_dict)

        logger.info("Iteration %d: LOAD completed from %s", iter_idx + 1, iter_ckpt_dir)

    # ──────────────────────────────────────────────────
    # Print results
    # ──────────────────────────────────────────────────
    print_timing_summary()
    aggregate_timing_across_ranks()

    # ──────────────────────────────────────────────────
    # Clean up
    # ──────────────────────────────────────────────────
    mpu.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()