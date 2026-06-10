# Megatron Dist Checkpoint Profiling

Profile Megatron's distributed checkpoint `save()` and `load()` APIs using a Qwen3-0.6B model.

This profiling approach mirrors verl's stream trainer checkpoint handoff profiling:
- Timing each sub-operation with `time.perf_counter()`
- NVTX markers for nsight profiling
- `FullyParallelSaveStrategyWrapper` / `FullyParallelLoadStrategyWrapper` support
- Strategy caching across iterations
- `--skip-optimizer` flag for focused model-only profiling (like `VERL_STREAM_HANDOFF_PROFILE_SKIP_OPTIMIZER_STATE`)

## Files

| File | Description |
|------|-------------|
| `profile_dist_ckpt.py` | Main profiling script (launched via torchrun) |
| `run_profile_dist_ckpt.sh` | Shell wrapper with env setup and nsys support |
| `README.md` | This file |

## Quick Start (Remote Container)

1. SSH to the remote container:
   ```bash
   ssh -p 30321 xuyimeng@221.130.15.74
   docker exec -it -u 1006:1006 -e HOME=/workspace 388c7a56d751 bash
   ```

2. Activate the venv:
   ```bash
   source /workspace/coding-verl/.venv/bin/activate
   ```

3. Run the profiling:
   ```bash
   cd /workspace/Megatron-LM
   bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
   ```

## Key Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/workspace/models/Qwen3-0.6B-hf` | HF model path |
| `CKPT_DIR` | `/workspace/model/dist_ckpt_profile` | Checkpoint output dir |
| `NUM_ITERS` | `3` | Number of save/load iterations |
| `SKIP_OPTIMIZER` | `1` | Skip optimizer state (focused profiling) |
| `TP` | `1` | Tensor parallel size |
| `PP` | `1` | Pipeline parallel size |
| `DP` | (auto) | Data parallel size |
| `CACHE_STRUCTURE` | `0` | Cache checkpoint structure across iterations |
| `ENABLE_NSYS_PROFILE` | `0` | Enable nsys profiling |
| `N_GPUS_PER_NODE` | `2` | Number of GPUs |

## Profile Operations

The script measures timing for these sub-operations (matching verl's stream trainer observability):

| Timing Label | NVTX Color | Description |
|--------------|-----------|-------------|
| `init.parallel_state` | blue | Initialize Megatron parallel state |
| `init.bridge_from_hf` | blue | Create mbridge from HF config |
| `init.build_megatron_model` | blue | Build Megatron model via mbridge |
| `build_sharded_state_dict.model` | yellow | Build model sharded state dict |
| `build_sharded_state_dict.optimizer` | yellow | Build optimizer sharded state dict |
| `save.get_strategy` | red | Get/create save strategy |
| `save.checkpoint_write` | red | dist_checkpointing.save() call |
| `load.get_strategy` | green | Get/create load strategy |
| `load.checkpoint_read` | green | dist_checkpointing.load() call |
| `load.apply_state` | green | Apply loaded state to model |

## Profiling Results (Qwen3-0.6B, 2 GPUs, TP=1 PP=1 DP=2, model-only)

First successful run with 2 iterations, skipping optimizer state:

```
======================================================================
  DIST CHECKPOINT PROFILING SUMMARY (MAX ACROSS RANKS)
======================================================================
  Operation                                          Max Mean (s)  Max Std (s)  Count
----------------------------------------------------------------------
  build_sharded_state_dict.model                           0.0063       0.0005      4
  init.bridge_from_hf                                      0.0023       0.0000      1
  init.build_megatron_model                                0.3935       0.0000      1
  init.parallel_state                                      0.1344       0.0000      1
  load.apply_state                                         0.0052       0.0000      2
  load.checkpoint_read                                     0.9609       0.0463      2
  load.get_strategy                                        0.0003       0.0000      2
  save.checkpoint_write                                    5.3845       3.9801      2
  save.get_strategy                                        0.0038       0.0035      2
======================================================================
```

Key observations:
- **Save dominates**: `save.checkpoint_write` takes ~5.4s (first iter ~9.4s, second iter ~1.4s)
  - First iteration is slower due to strategy computation and disk I/O
  - Second iteration benefits from strategy caching
- **Load is faster**: `load.checkpoint_read` takes ~1.0s consistently
- **State dict construction is fast**: `build_sharded_state_dict.model` ~6ms
- **Strategy lookup is cheap**: `save.get_strategy` ~4ms, `load.get_strategy` ~0.3ms
- **Apply state is fast**: `load.apply_state` ~5ms

## Notes

- The model uses **random weights** (not loaded from HF checkpoint) since we're profiling
  the dist save/load APIs only. Weight values are irrelevant for this measurement.
- The profiling runs on the local repo's Megatron code (shadowing the pip-installed megatron-core),
  with compatibility patches for `ModelType.encoder_and_decoder` and `TENorm`.
- NCCL is configured for RTX 4090 (no NVLink/P2P): `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1`