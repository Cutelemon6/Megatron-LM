# Megatron Dist Checkpoint Save/Load Profiling Analysis

## Fine-Grained Sub-Operation Breakdown

The profiling script uses monkey-patching to inject timing around key internal
functions of `dist_checkpointing.save()` and `dist_checkpointing.load()`.

### Save Flow (serialization.py → strategies)

```
dist_checkpointing.save()
  ├── save.preprocess           (save_preprocess: factory + validation)
  ├── save.common_pt            (save_common: rank0 writes common.pt)
  ├── save.fp_wrapper_save      (FullyParallelSaveStrategyWrapper.save)
  │   ├── save.apply_saving_parallelization  (ALL_GATHER_OBJECT metadata exchange)
  │   └── save.torch_dist_save  (TorchDistSaveShardedStrategy.save)
  │       ├── save.replace_keys_for_sharding (_replace_state_dict_keys_with_sharded_keys)
  │       ├── save.mcore_to_pyt_translate    (mcore_to_pyt_state_dict: MCore→PyT conversion)
  │       ├── [MCoreSavePlanner: create_local_plan + create_global_plan]
  │       └── [FileSystemWriter: actual disk I/O]
  └── save.metadata_finalize    (save_config + barrier)
```

### Load Flow (serialization.py → strategies)

```
dist_checkpointing.load()
  ├── load.verify_checkpoint     (verify checkpoint directory)
  ├── load.common_pt            (load_common: rank0 reads common.pt)
  ├── load.preprocess           (load_preprocess: factory + extraction)
  ├── [extract_sharded_base + validate_integrity]
  ├── load.fp_wrapper_load      (FullyParallelLoadStrategyWrapper.load)
  │   ├── load.apply_loading_parallelization  (ALL_GATHER_OBJECT metadata exchange)
  │   ├── load.torch_dist_load  (TorchDistLoadShardedStrategy.load: disk read)
  │   ├── load.exchange_by_distribution (NCCL ALL-TO-ALL: cross-rank tensor exchange)
  │   └── load.exchange_objects (GLOO GATHER: cross-rank object exchange)
  └── [apply_factory_merges]
```

## Why Save is Slow (Previous Results: ~5.4s)

The coarse `save.checkpoint_write` timing (~5.4s) wraps the entire save() call.
Based on code analysis, the bottleneck has **3 main sources**:

1. **apply_saving_parallelization (~1-2s on first iter)**:
   - ALL_GATHER_OBJECT exchanges ShardedTensor metadata across DP ranks
   - For fully_sharded_model_space optimizer: ~300 params × 3 states = 900 ShardedTensor items
   - Greedy distribution computation to assign shards to ranks
   - **Mitigated by cache_structure=True**: second iter uses cached distribution

2. **mcore_to_pyt_state_dict (~0.5-1s)**:
   - CPU-bound conversion: each ShardedTensor → DTensor with shard mapping
   - For fully_sharded_model_space: 900 individual tensor conversions
   - **dp_reshardable advantage**: only ~3 ShardedTensors per bucket (few conversions)

3. **Disk I/O (FileSystemWriter, ~2-3s)**:
   - PyTorch DCP FileSystemWriter writes tensor data to local disk
   - For fully_sharded_model_space: 900 small files (one per param state)
   - For dp_reshardable: few large contiguous buffer files (better I/O throughput)
   - Thread_count=2 can help (parallel writes within a rank)

## Optimizer Sharding Type Comparison

### fully_sharded_model_space (verl default)
- Each optimizer param state (exp_avg, exp_avg_sq) is a separate ShardedTensor
- Allows TP/PP reshard (cross TP/PP groups load)
- **Save overhead**: metadata exchange + per-param conversion + many small files
- **Load overhead**: metadata exchange + ALL-TO-ALL data redistribution

### dp_reshardable (FASTEST)
- Optimizer state stored in internal DistOpt bucket format (flat contiguous buffers)
- Each DP rank saves its shard independently, **no inter-rank communication**
- **Save advantage**: NO apply_saving_parallelization, few large writes
- **Load advantage**: NO metadata exchange, each rank reads its own shard directly
- **Limitation**: Only supports DP reshard, cannot change TP/PP configuration
- **When applicable**: phase1 and phase2 have the same TP/PP (verl DAPO scenario)

### fully_reshardable
- Gather all optimizer data on DP rank 0 during save (ALL_GATHER)
- reshape into model param-like sizes for full reshardability
- **Save overhead**: ALL_GATHER of ~4x model data volume
- **Load overhead**: ALL ranks read full data, then flatten+trim
- **Worst I/O**: rank 0 writes everything, other ranks idle

## Quantitative Comparison (Expected, Qwen3-0.6B, DP=2)

| Metric | fully_sharded_model_space | dp_reshardable | fully_reshardable |
|--------|--------------------------|----------------|-------------------|
| **Save time (1st iter)** | ~9s | ~3s | ~12s |
| **Save time (cached)** | ~1.4s | ~0.8s | ~8s |
| **Load time** | ~1.0s | ~0.3s | ~1.5s |
| **Save communication** | metadata exchange | none | ALL_GATHER 4x model |
| **Load communication** | metadata + ALL-TO-ALL | none | each rank reads full |
| **File count** | ~900 small files | ~3 large files | ~300 files on rank0 |
| **TP/PP reshard** | ✅ | ❌ | ✅ |

## DP Reshard Acceleration Strategy

### Key Insight
In verl's stream trainer, phase1 and phase2 **use the same TP/PP configuration**
(e.g., TP=1, PP=1 for Qwen3-0.6B). This means `dp_reshardable` is applicable
and would be significantly faster than `fully_sharded_model_space`.

### Proposed Changes to verl

1. **Detect same TP/PP → use dp_reshardable**:
   ```python
   # In stream_trainer runtime.py:
   if source_plan.tp_size == target_plan.tp_size and \
      source_plan.pp_size == target_plan.pp_size:
       optimizer_metadata = {"distrib_optim_sharding_type": "dp_reshardable"}
   else:
       optimizer_metadata = self._FULLY_RESHARDABLE_OPTIMIZER_METADATA
   ```

2. **Skip FullyParallel wrappers for dp_reshardable**:
   - `dp_reshardable` doesn't need `apply_saving_parallelization` (no metadata exchange)
   - `dp_reshardable` doesn't need `exchange_by_distribution` on load (no cross-rank exchange)
   - Each rank independently saves/loads its DP shard
   - This eliminates both the metadata exchange AND the data redistribution overhead

3. **Thread count optimization**:
   - Set `thread_count=2` for FileSystemWriter (parallel writes within a rank)
   - verl already supports this via `VERL_MEGATRON_DIST_CKPT_THREAD_COUNT`

4. **Cache structure across iterations**:
   - verl already uses `cache_structure=True` (caches distribution + plan)
   - For dp_reshardable, caching is simpler (fewer ShardedTensors to track)

### Expected Speedup

With dp_reshardable + same TP/PP, the save/load time breakdown would be:

```
Save (dp_reshardable, cached):
  save.preprocess:          ~0.01s   (unchanged)
  save.common_pt:           ~0.01s   (unchanged)
  save.apply_saving_parallelization: SKIPPED ✅ (no communication)
  save.mcore_to_pyt:        ~0.05s   (few tensors, very fast)
  save.disk_write:          ~0.5s    (few large files, good throughput)
  save.metadata_finalize:   ~0.01s   (unchanged)
  Total:                    ~0.6s    vs ~1.4s (43% faster)

Load (dp_reshardable):
  load.verify:              ~0.001s  (unchanged)
  load.common_pt:           ~0.001s  (unchanged)
  load.preprocess:          ~0.01s   (unchanged)
  load.apply_loading_parallelization: SKIPPED ✅ (no communication)
  load.disk_read:           ~0.2s    (few large files, good throughput)
  load.exchange_by_distribution: SKIPPED ✅ (no communication)
  Total:                    ~0.3s    vs ~1.0s (70% faster)
```

### Implementation Risk

- **Must ensure same TP/PP**: If phase2 changes TP/PP, dp_reshardable will fail
  to load correctly. Need a runtime check before choosing sharding type.
- **Must ensure same DP group composition**: DP reshard only works if
  the checkpoint was saved with the same or compatible DP world size.
  Changing DP world size is supported (dp_reshardable handles padding).

## Running the Profile

### Model-only baseline (skip optimizer)
```bash
CUDA_VISIBLE_DEVICES=2,3 \
  bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
```

### With optimizer, single sharding type
```bash
CUDA_VISIBLE_DEVICES=2,3 OPTIM_SHARDING_TYPE=dp_reshardable \
  bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
```

### Full reshard comparison (3 sharding types)
```bash
CUDA_VISIBLE_DEVICES=2,3 COMPARE_RESHARD=1 \
  bash profiling/dist_ckpt/run_profile_dist_ckpt.sh
```

**Note**: If CUDA driver issues occur, restart the container:
```bash
docker stop verl-xuyimeng && docker start verl-xuyimeng
```