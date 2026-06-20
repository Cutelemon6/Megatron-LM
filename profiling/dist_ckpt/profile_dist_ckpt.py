"""Profile Megatron distributed checkpoint save and load with fine-grained sub-operation timing.

This script profiles the key Megatron dist_checkpointing APIs with fine-grained breakdown:
  - dist_checkpointing.save()  → preprocess, common, parallelize, translate, planner_write, finalize
  - dist_checkpointing.load()  → verify, common, preprocess, validate, strategy_load, apply_factory

Additionally, it compares:
  1. Fully-reshardable vs DP-reshardable optimizer state (sharding_type)
  2. Strategy caching effects (cache_structure=True vs False)

The model is initialized from an HF Qwen3-0.6B checkpoint using mbridge (random weights).
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


def _patch_mcore_no_te():
    """Patch Megatron's gpt_layer_specs to avoid requiring Transformer Engine."""
    try:
        from megatron.core.models.gpt import gpt_layer_specs as specs_mod
        if not hasattr(specs_mod, "_profile_no_te_patched"):
            _orig_get_spec = specs_mod.get_gpt_decoder_block_spec
            def _get_spec_no_te(config, use_transformer_engine=True, *args, **kwargs):
                return _orig_get_spec(config, use_transformer_engine=False, *args, **kwargs)
            specs_mod.get_gpt_decoder_block_spec = _get_spec_no_te
            specs_mod._profile_no_te_patched = True
            logger.info("Patched mcore gpt_layer_specs to use_transformer_engine=False")
    except ImportError:
        pass


# ──────────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────────

_TIMING_RESULTS: dict[str, list[float]] = {}


@contextmanager
def record_timing(name: str, nvtx_color: str | None = None):
    """Record wall-clock time for an operation, with optional NVTX marker."""
    start = time.perf_counter()
    try:
        if nvtx_color and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
        yield
    finally:
        elapsed = time.perf_counter() - start
        if nvtx_color and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
        _TIMING_RESULTS.setdefault(name, []).append(elapsed)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info("[rank=%d] %s: %.4f s", rank, name, elapsed)


def print_timing_summary():
    """Print a summary table of all timing results (rank 0 only)."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return

    print("\n" + "=" * 90)
    print("  DIST CHECKPOINT PROFILING SUMMARY")
    print("=" * 90)
    print(f"  {'Operation':<65} {'Mean (s)':>10} {'Std (s)':>10} {'Count':>6}")
    print("-" * 90)

    for name, times in sorted(_TIMING_RESULTS.items()):
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0.0
        print(f"  {name:<65} {mean:>10.4f} {std:>10.4f} {len(times):>6}")

    print("=" * 90 + "\n")


def aggregate_timing_across_ranks():
    """Aggregate timing results across all ranks (max across ranks)."""
    if not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()
    local_data = {name: times for name, times in _TIMING_RESULTS.items()}
    local_json = json.dumps(local_data).encode()

    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_json)

    if torch.distributed.get_rank() != 0:
        return

    print("\n" + "=" * 90)
    print("  DIST CHECKPOINT PROFILING SUMMARY (MAX ACROSS RANKS)")
    print("=" * 90)
    print(f"  {'Operation':<65} {'Max Mean (s)':>12} {'Max Std (s)':>12}")
    print("-" * 90)

    per_op_per_iter: dict[str, list[float]] = {}
    for rank_data in gathered:
        rank_dict = json.loads(rank_data)
        for name, times in rank_dict.items():
            for i, t in enumerate(times):
                key = f"{name}_iter{i}"
                per_op_per_iter.setdefault(key, []).append(t)

    for name, times in sorted(_TIMING_RESULTS.items()):
        n_iters = len(times)
        max_times = []
        for i in range(n_iters):
            key = f"{name}_iter{i}"
            max_times.append(max(per_op_per_iter.get(key, [0.0])))
        mean = sum(max_times) / len(max_times)
        std = (sum((t - mean) ** 2 for t in max_times) / len(max_times)) ** 0.5 if len(max_times) > 1 else 0.0
        print(f"  {name:<65} {mean:>12.4f} {std:>12.4f}")

    print("=" * 90 + "\n")


def save_timing_json(output_path: str):
    """Save timing results to JSON file (rank 0 only)."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return

    results = {}
    for name, times in sorted(_TIMING_RESULTS.items()):
        mean = sum(times) / len(times)
        std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5 if len(times) > 1 else 0.0
        results[name] = {"mean": mean, "std": std, "times": times, "count": len(times)}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved timing results to %s", output_path)


# ──────────────────────────────────────────────────
# Strategy helpers (from verl's dist_checkpointing.py)
# ──────────────────────────────────────────────────

_SAVE_STRATEGY_CACHE: dict[tuple[Any, ...], Any] = {}
_LOAD_STRATEGY_CACHE: dict[tuple[Any, ...], Any] = {}


def get_save_strategy(parallelization_group, cache_structure=False, thread_count=None):
    from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper

    cache_key = _strategy_cache_key(parallelization_group, thread_count, cache_structure, "save")
    if cache_structure and cache_key in _SAVE_STRATEGY_CACHE:
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
    return wrapped


def get_load_strategy(ckpt_dir, parallelization_group, cache_structure=False, thread_count=None):
    from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
    import inspect

    cache_key = _strategy_cache_key_load(ckpt_dir, parallelization_group, thread_count, cache_structure)
    if cache_structure and cache_key in _LOAD_STRATEGY_CACHE:
        return _LOAD_STRATEGY_CACHE[cache_key]

    load_strategy = get_default_load_sharded_strategy(ckpt_dir)
    if thread_count is not None and hasattr(load_strategy, "thread_count"):
        load_strategy.thread_count = thread_count
    if cache_structure and hasattr(load_strategy, "use_cached_ckpt_structure"):
        load_strategy.use_cached_ckpt_structure = True

    kwargs = {}
    try:
        load_wrapper_params = inspect.signature(FullyParallelLoadStrategyWrapper).parameters
    except (TypeError, ValueError):
        load_wrapper_params = {}
    if cache_structure and "do_cache_distribution" in load_wrapper_params:
        kwargs["do_cache_distribution"] = cache_structure

    wrapped = FullyParallelLoadStrategyWrapper(load_strategy, parallelization_group, **kwargs)
    if cache_structure:
        _LOAD_STRATEGY_CACHE[cache_key] = wrapped
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
# Monkey-patching for fine-grained timing
# ──────────────────────────────────────────────────

_PATCHED = False


def install_timing_patches():
    """Monkey-patch dist_checkpointing internals for fine-grained timing.

    The patches inject timing around key sub-operations WITHOUT changing
    the internal logic. The original functions are called unchanged, just
    wrapped with record_timing.

    Save flow captured:
      save.preprocess          → serialization.save_preprocess()
      save.common_pt           → common.save_common()
      save.apply_saving_parallelization → FullyParallelSaveStrategyWrapper.apply_saving_parallelization()
      save.torch_dist_save     → TorchDistSaveShardedStrategy.save()
        save.replace_keys       → _replace_state_dict_keys_with_sharded_keys()
        save.mcore_to_pyt       → mcore_to_pyt_state_dict()
      save.metadata_finalize   → save_config + barrier (in serialization.save())

    Load flow captured:
      load.verify              → serialization.verify_checkpoint()
      load.common_pt           → common.load_common()
      load.preprocess          → serialization.load_preprocess()
      load.apply_loading_parallelization → FullyParallelLoadStrategyWrapper.apply_loading_parallelization()
      load.torch_dist_load     → TorchDistLoadShardedStrategy.load()
      load.exchange_by_distribution → exchange_utils.exchange_by_distribution()
      load.exchange_objects    → exchange_utils.exchange_loaded_objects_gather_object()
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # ── Save side ──
    import megatron.core.dist_checkpointing.serialization as ser_mod
    import megatron.core.dist_checkpointing.strategies.common as common_mod
    import megatron.core.dist_checkpointing.strategies.fully_parallel as fp_mod
    import megatron.core.dist_checkpointing.strategies.torch as torch_mod

    # serialization.save_preprocess
    _orig_save_preprocess = ser_mod.save_preprocess
    def _patched_save_preprocess(*args, **kwargs):
        with record_timing("save.preprocess", nvtx_color="red"):
            return _orig_save_preprocess(*args, **kwargs)
    ser_mod.save_preprocess = _patched_save_preprocess

    # common.save_common
    _orig_save_common = common_mod.save_common
    def _patched_save_common(*args, **kwargs):
        with record_timing("save.common_pt", nvtx_color="red"):
            return _orig_save_common(*args, **kwargs)
    common_mod.save_common = _patched_save_common

    # FullyParallelSaveStrategyWrapper.apply_saving_parallelization
    _orig_apply_saving = fp_mod.FullyParallelSaveStrategyWrapper.apply_saving_parallelization
    def _patched_apply_saving(self, *args, **kwargs):
        with record_timing("save.apply_saving_parallelization", nvtx_color="red"):
            return _orig_apply_saving(self, *args, **kwargs)
    fp_mod.FullyParallelSaveStrategyWrapper.apply_saving_parallelization = _patched_apply_saving

    # FullyParallelSaveStrategyWrapper.save (the wrapper entrypoint)
    _orig_fp_save = fp_mod.FullyParallelSaveStrategyWrapper.save
    def _patched_fp_save(self, *args, **kwargs):
        # apply_saving_parallelization is already patched above
        # We just record the total save wrapper time (which includes apply + base)
        with record_timing("save.fp_wrapper_save", nvtx_color="red"):
            return _orig_fp_save(self, *args, **kwargs)
    fp_mod.FullyParallelSaveStrategyWrapper.save = _patched_fp_save

    # TorchDistSaveShardedStrategy.save
    _orig_tds_save = torch_mod.TorchDistSaveShardedStrategy.save
    def _patched_tds_save(self, *args, **kwargs):
        with record_timing("save.torch_dist_save", nvtx_color="red"):
            return _orig_tds_save(self, *args, **kwargs)
    torch_mod.TorchDistSaveShardedStrategy.save = _patched_tds_save

    # _replace_state_dict_keys_with_sharded_keys
    _orig_replace_keys = torch_mod._replace_state_dict_keys_with_sharded_keys
    def _patched_replace_keys(*args, **kwargs):
        with record_timing("save.replace_keys_for_sharding", nvtx_color="red"):
            return _orig_replace_keys(*args, **kwargs)
    torch_mod._replace_state_dict_keys_with_sharded_keys = _patched_replace_keys

    # mcore_to_pyt_state_dict
    _orig_mcore_to_pyt = torch_mod.mcore_to_pyt_state_dict
    def _patched_mcore_to_pyt(*args, **kwargs):
        with record_timing("save.mcore_to_pyt_translate", nvtx_color="red"):
            return _orig_mcore_to_pyt(*args, **kwargs)
    torch_mod.mcore_to_pyt_state_dict = _patched_mcore_to_pyt

    # ── Load side ──
    # serialization.verify_checkpoint
    _orig_verify = ser_mod.verify_checkpoint
    def _patched_verify(*args, **kwargs):
        with record_timing("load.verify_checkpoint", nvtx_color="green"):
            return _orig_verify(*args, **kwargs)
    ser_mod.verify_checkpoint = _patched_verify

    # common.load_common
    _orig_load_common = common_mod.load_common
    def _patched_load_common(*args, **kwargs):
        with record_timing("load.common_pt", nvtx_color="green"):
            return _orig_load_common(*args, **kwargs)
    common_mod.load_common = _patched_load_common

    # serialization.load_preprocess
    _orig_load_preprocess = ser_mod.load_preprocess
    def _patched_load_preprocess(*args, **kwargs):
        with record_timing("load.preprocess", nvtx_color="green"):
            return _orig_load_preprocess(*args, **kwargs)
    ser_mod.load_preprocess = _patched_load_preprocess

    # FullyParallelLoadStrategyWrapper.apply_loading_parallelization
    _orig_apply_loading = fp_mod.FullyParallelLoadStrategyWrapper.apply_loading_parallelization
    def _patched_apply_loading(self, *args, **kwargs):
        with record_timing("load.apply_loading_parallelization", nvtx_color="green"):
            return _orig_apply_loading(self, *args, **kwargs)
    fp_mod.FullyParallelLoadStrategyWrapper.apply_loading_parallelization = _patched_apply_loading

    # FullyParallelLoadStrategyWrapper.load
    _orig_fp_load = fp_mod.FullyParallelLoadStrategyWrapper.load
    def _patched_fp_load(self, *args, **kwargs):
        # This internally calls: apply_loading_parallelization (already patched)
        # then base_strategy.load + exchange. We record the total wrapper time.
        with record_timing("load.fp_wrapper_load", nvtx_color="green"):
            return _orig_fp_load(self, *args, **kwargs)
    fp_mod.FullyParallelLoadStrategyWrapper.load = _patched_fp_load

    # TorchDistLoadShardedStrategy.load
    _orig_tds_load = torch_mod.TorchDistLoadShardedStrategy.load
    def _patched_tds_load(self, *args, **kwargs):
        load_name = "load.torch_dist_load_unknown"
        try:
            from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor
            from megatron.core.dist_checkpointing.dict_utils import nested_values

            sharded_state_dict = args[0] if args else kwargs.get("sharded_state_dict", {})
            values = list(nested_values(sharded_state_dict))
            has_tensors = any(isinstance(value, ShardedTensor) for value in values)
            has_objects = any(isinstance(value, ShardedObject) for value in values)
            if has_tensors and has_objects:
                load_name = "load.torch_dist_load_mixed"
            elif has_tensors:
                load_name = "load.torch_dist_load_tensors"
            elif has_objects:
                load_name = "load.torch_dist_load_objects"
            else:
                load_name = "load.torch_dist_load_empty"
        except Exception:
            pass
        with record_timing("load.torch_dist_load", nvtx_color="green"):
            with record_timing(load_name, nvtx_color="green"):
                return _orig_tds_load(self, *args, **kwargs)
    torch_mod.TorchDistLoadShardedStrategy.load = _patched_tds_load

    # exchange_by_distribution. FullyParallelLoadStrategyWrapper imports this
    # symbol directly, so patch both the source module and the local reference.
    from megatron.core.dist_checkpointing import exchange_utils as exchange_mod
    _orig_exchange_by_dist = fp_mod.exchange_by_distribution
    def _patched_exchange_by_dist(*args, **kwargs):
        with record_timing("load.exchange_by_distribution", nvtx_color="green"):
            return _orig_exchange_by_dist(*args, **kwargs)
    fp_mod.exchange_by_distribution = _patched_exchange_by_dist
    exchange_mod.exchange_by_distribution = _patched_exchange_by_dist

    # exchange_loaded_objects_gather_object. Patch the local fully_parallel
    # reference as well, otherwise training load misses this timer.
    _orig_exchange_obj = fp_mod.exchange_loaded_objects_gather_object
    def _patched_exchange_obj(*args, **kwargs):
        with record_timing("load.exchange_objects", nvtx_color="green"):
            return _orig_exchange_obj(*args, **kwargs)
    fp_mod.exchange_loaded_objects_gather_object = _patched_exchange_obj
    exchange_mod.exchange_loaded_objects_gather_object = _patched_exchange_obj

    logger.info("Installed fine-grained timing patches for dist_checkpointing save/load")


# ──────────────────────────────────────────────────
# Model initialization
# ──────────────────────────────────────────────────

def initialize_model_from_hf(model_path, tp_size, pp_size):
    from mbridge import AutoBridge

    with record_timing("init.bridge_from_hf", nvtx_color="blue"):
        bridge = AutoBridge.from_pretrained(model_path)
        bridge.set_extra_args(attention_backend="unfused", sequence_parallel=False)

    with record_timing("init.build_megatron_model", nvtx_color="blue"):
        model_chunks = bridge.get_model(bf16=True, wrap_with_ddp=False)

    logger.info("Initialized Megatron model (random weights): tp=%d pp=%d, chunks=%d", tp_size, pp_size, len(model_chunks))
    return model_chunks, bridge


# ──────────────────────────────────────────────────
# Sharded state dict construction
# ──────────────────────────────────────────────────

def build_sharded_state_dict(model_chunks, optimizer=None, skip_optimizer=False, optimizer_sharding_type=None):
    """Build the sharded state dict for save, with fine-grained timing."""
    from megatron.core.dist_checkpointing import LocalNonpersistentObject

    state_dict = {}

    with record_timing("build_sharded_state_dict.model", nvtx_color="yellow"):
        for i, chunk in enumerate(model_chunks):
            chunk_for_sd = getattr(chunk, "module", chunk)
            if hasattr(chunk_for_sd, "sharded_state_dict"):
                model_sd = chunk_for_sd.sharded_state_dict()
            else:
                model_sd = chunk_for_sd.state_dict()
            model_sd = {k: v for k, v in model_sd.items() if not isinstance(v, LocalNonpersistentObject)}
            state_dict[f"model{i}"] = model_sd

    if optimizer is not None and not skip_optimizer:
        with record_timing("build_sharded_state_dict.optimizer_materialize", nvtx_color="yellow"):
            _materialize_megatron_optimizer_state(optimizer)

        optim_kwargs = {}
        if optimizer_sharding_type:
            optim_kwargs["metadata"] = {"distrib_optim_sharding_type": optimizer_sharding_type}

        sharding_desc = optimizer_sharding_type or "default"
        with record_timing(f"build_sharded_state_dict.optimizer_{sharding_desc}", nvtx_color="yellow"):
            optim_state_dict = optimizer.sharded_state_dict(
                state_dict, is_loading=False, **optim_kwargs
            )
            state_dict["optimizer"] = optim_state_dict

    return state_dict


def _materialize_megatron_optimizer_state(optimizer):
    """Materialize optimizer state (lazy init -> actual state)."""
    if hasattr(optimizer, "inner_optimizers"):
        for inner_opt in optimizer.inner_optimizers:
            if hasattr(inner_opt, "get_parameter_state_dp_reshardable"):
                inner_opt.get_parameter_state_dp_reshardable()
    elif hasattr(optimizer, "get_parameter_state_dp_reshardable"):
        optimizer.get_parameter_state_dp_reshardable()


def apply_loaded_state(model_chunks, loaded_state_dict):
    """Apply loaded state dict to model."""
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
    parser = argparse.ArgumentParser(description="Profile Megatron dist checkpoint save/load (fine-grained)")
    parser.add_argument("--model-path", type=str, default="/workspace/models/Qwen3-0.6B-hf")
    parser.add_argument("--ckpt-dir", type=str, default="/workspace/model/dist_ckpt_profile")
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--skip-optimizer", type=int, default=0,
                        help="Skip optimizer state (0=no, 1=yes)")
    parser.add_argument("--optimizer-sharding-type", type=str, default=None,
                        choices=["fully_sharded_model_space", "fully_reshardable", "dp_reshardable"],
                        help="Optimizer sharding type for DistributedOptimizer. "
                             "fully_sharded_model_space: each param is separate ShardedTensor (verl default). "
                             "fully_reshardable: gather on DP rank 0 during save (allows TP/PP reshard). "
                             "dp_reshardable: no inter-rank communication, DP-parallel only.")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--cache-structure", type=int, default=1,
                        help="Cache checkpoint structure across iterations (recommended)")
    parser.add_argument("--thread-count", type=int, default=None)
    parser.add_argument("--validate-integrity", type=int, default=0,
                        help="Validate access integrity (0=no for speed, 1=yes)")
    parser.add_argument("--use-distributed-optimizer", type=int, default=1,
                        help="Use DistributedOptimizer (ZeRO) for optimizer state")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save timing results as JSON")
    args = parser.parse_args()

    skip_optimizer = bool(args.skip_optimizer)
    cache_structure = bool(args.cache_structure)
    validate_integrity = bool(args.validate_integrity)
    use_distributed_optimizer = bool(args.use_distributed_optimizer)

    # ── Distributed init ──
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    _patch_mcore_no_te()

    tp_size = args.tensor_model_parallel_size
    pp_size = args.pipeline_model_parallel_size
    dp_size = args.data_parallel_size or (world_size // (tp_size * pp_size))

    logger.info("rank=%d local_rank=%d world_size=%d tp=%d pp=%d dp=%d",
                rank, local_rank, world_size, tp_size, pp_size, dp_size)

    # ── Parallel state + RNG ──
    with record_timing("init.parallel_state", nvtx_color="blue"):
        from megatron.core import parallel_state as mpu
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        rng_tracker = get_cuda_rng_tracker()
        rng_tracker.add("model-parallel-rng", 1234 + rank)

    # ── Install timing patches (must be before any checkpoint calls) ──
    install_timing_patches()

    # ── Model init ──
    model_chunks, bridge = initialize_model_from_hf(args.model_path, tp_size, pp_size)

    # ── Optimizer ──
    optimizer = None
    if not skip_optimizer:
        with record_timing("init.optimizer", nvtx_color="blue"):
            from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
            optim_config = OptimizerConfig(
                optimizer="adam",
                lr=1e-6,
                adam_beta1=0.9,
                adam_beta2=0.999,
                use_distributed_optimizer=use_distributed_optimizer,
                bf16=True,
            )
            optimizer = get_megatron_optimizer(model_chunks, config=optim_config)

            # Run one dummy step to initialize optimizer state
            with record_timing("init.optimizer_step", nvtx_color="blue"):
                dummy_loss = sum(p.sum() for p in model_chunks[0].parameters())
                dummy_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    # ── DP group ──
    dp_group = mpu.get_data_parallel_group(with_context_parallel=True)

    # Print config
    sharding_desc = args.optimizer_sharding_type or "default(fully_sharded_model_space)"
    logger.info("Config: optimizer_sharding=%s, cache_structure=%s, skip_optimizer=%s, dp=%d",
                sharding_desc, cache_structure, skip_optimizer, dp_size)

    # ── Run profiling iterations ──
    ckpt_base_dir = args.ckpt_dir

    for iter_idx in range(args.num_iters):
        logger.info("=== Iteration %d/%d ===", iter_idx + 1, args.num_iters)

        iter_ckpt_dir = os.path.join(ckpt_base_dir, f"iter_{iter_idx}")
        if os.path.exists(iter_ckpt_dir):
            shutil.rmtree(iter_ckpt_dir, ignore_errors=True)
        os.makedirs(iter_ckpt_dir, exist_ok=True)

        gc.collect()
        torch.cuda.empty_cache()

        # ── Build sharded state dict ──
        state_dict = build_sharded_state_dict(
            model_chunks, optimizer=optimizer, skip_optimizer=skip_optimizer,
            optimizer_sharding_type=args.optimizer_sharding_type,
        )

        # ── Get save strategy ──
        with record_timing("save.get_strategy", nvtx_color="red"):
            save_strategy = get_save_strategy(dp_group, cache_structure=cache_structure, thread_count=args.thread_count)

        # ── SAVE ──
        with record_timing("save.checkpoint_write_total", nvtx_color="red"):
            from megatron.core import dist_checkpointing
            dist_checkpointing.save(
                state_dict, iter_ckpt_dir,
                sharded_strategy=save_strategy,
                validate_access_integrity=validate_integrity,
            )
        logger.info("Iteration %d: SAVE completed", iter_idx + 1)

        # ── Get load strategy ──
        with record_timing("load.get_strategy", nvtx_color="green"):
            load_strategy = get_load_strategy(iter_ckpt_dir, dp_group, cache_structure=cache_structure, thread_count=args.thread_count)

        # ── Build load template ──
        load_template = build_sharded_state_dict(
            model_chunks, optimizer=optimizer, skip_optimizer=skip_optimizer,
            optimizer_sharding_type=args.optimizer_sharding_type,
        )

        # ── LOAD ──
        with record_timing("load.checkpoint_read_total", nvtx_color="green"):
            from megatron.core import dist_checkpointing
            loaded_state_dict = dist_checkpointing.load(
                load_template, iter_ckpt_dir,
                sharded_strategy=load_strategy,
                validate_access_integrity=validate_integrity,
            )
        apply_loaded_state(model_chunks, loaded_state_dict)
        logger.info("Iteration %d: LOAD completed", iter_idx + 1)

    # ── Results ──
    print_timing_summary()
    aggregate_timing_across_ranks()

    if args.output_json:
        save_timing_json(args.output_json)

    # ── Cleanup ──
    mpu.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()