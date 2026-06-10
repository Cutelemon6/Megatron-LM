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
| `init.load_hf_weights` | blue | Load HF weights into model |
| `build_sharded_state_dict.model` | yellow | Build model sharded state dict |
| `build_sharded_state_dict.optimizer` | yellow | Build optimizer sharded state dict |
| `save.get_strategy` | red | Get/create save strategy |
| `save.checkpoint_write` | red | dist_checkpointing.save() call |
| `load.get_strategy` | green | Get/create load strategy |
| `load.checkpoint_read` | green | dist_checkpointing.load() call |
| `load.apply_state` | green | Apply loaded state to model |

## Output

After running, the script prints a summary table like:

```
======================================================================
  DIST CHECKPOINT PROFILING SUMMARY (MAX ACROSS RANKS)
======================================================================
  Operation                                          Max Mean (s)   Max Std (s)  Count
----------------------------------------------------------------------
  build_sharded_state_dict.model                          0.0123        0.0012      3
  save.checkpoint_write                                   1.4567        0.1234      3
  load.checkpoint_read                                    0.8901        0.0567      3
  load.apply_state                                        0.0045        0.0003      3
======================================================================
```