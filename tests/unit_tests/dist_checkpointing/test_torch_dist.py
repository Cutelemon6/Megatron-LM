# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Tests for PyTorch DCP based checkpoint format. """

import io
import os
import pickle
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from megatron.training.checkpointing import _dist_ckpt_load_workers
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class TestDistCkptWorkerConfig:
    def test_load_workers_falls_back_to_save_workers(self):
        args = SimpleNamespace(dist_ckpt_workers=32, dist_ckpt_load_workers=None)

        assert _dist_ckpt_load_workers(args) == 32

    def test_load_workers_can_override_save_workers(self):
        args = SimpleNamespace(dist_ckpt_workers=32, dist_ckpt_load_workers=16)

        assert _dist_ckpt_load_workers(args) == 16


class TestAsyncPreloadCopy:
    def test_cpu_tensor_copy_ignores_pinned_d2h(self):
        tensor = torch.arange(8, dtype=torch.float32)

        copied = FileSystemWriterAsync._copy_tensor_to_cpu(
            tensor, non_blocking=True, use_pinned_d2h=True
        )

        assert copied.device.type == 'cpu'
        assert torch.equal(copied, tensor)

    def test_cpu_tensor_copy_reports_fallback_copy_stats(self):
        tensor = torch.arange(8, dtype=torch.float32)
        profile_stats = {}

        copied = FileSystemWriterAsync._copy_tensor_to_cpu(
            tensor,
            non_blocking=True,
            use_pinned_d2h=True,
            profile_stats=profile_stats,
        )

        assert copied.device.type == 'cpu'
        assert profile_stats['fallback_copy_items'] == 1
        assert profile_stats.get('pinned_copy_items', 0) == 0
        assert profile_stats['fallback_copy_elapsed'] >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
    def test_cuda_tensor_can_stage_to_pinned_cpu(self):
        tensor = torch.arange(8, dtype=torch.float32, device='cuda')

        copied = FileSystemWriterAsync._copy_tensor_to_cpu(
            tensor, non_blocking=True, use_pinned_d2h=True
        )
        torch.cuda.synchronize()

        assert copied.device.type == 'cpu'
        assert copied.is_pinned()
        assert torch.equal(copied, tensor.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
    def test_cuda_tensor_reports_pinned_copy_stats(self):
        tensor = torch.arange(8, dtype=torch.float32, device='cuda')
        profile_stats = {}

        copied = FileSystemWriterAsync._copy_tensor_to_cpu(
            tensor,
            non_blocking=True,
            use_pinned_d2h=True,
            profile_stats=profile_stats,
        )
        torch.cuda.synchronize()

        assert copied.device.type == 'cpu'
        assert copied.is_pinned()
        assert profile_stats['pinned_copy_items'] == 1
        assert profile_stats.get('fallback_copy_items', 0) == 0
        assert profile_stats['pinned_alloc_elapsed'] >= 0.0
        assert profile_stats['pinned_copy_enqueue_elapsed'] >= 0.0

    def test_preload_tensors_profiles_rss_before_and_after(self, monkeypatch, caplog):
        bytes_data = []
        tensor_data = [(object(), torch.arange(4, dtype=torch.float32))]
        write_buckets = [('payload.distcp', 'storage_key', (bytes_data, tensor_data))]
        memory_values = iter([200 * 1024 * 1024, 220 * 1024 * 1024])
        monkeypatch.setattr(filesystem_async, '_process_memory', lambda: next(memory_values))
        monkeypatch.setenv('MEGATRON_DIST_CKPT_WRITER_PROFILE', '1')

        with caplog.at_level('WARNING', logger=filesystem_async.__name__):
            result = FileSystemWriterAsync.preload_tensors(write_buckets, non_blocking=False, rank=7)

        assert result[0][2][1][0][1].device.type == 'cpu'
        assert 'rss_before_mb=200.0' in caplog.text
        assert 'rss_after_mb=220.0' in caplog.text


class TestAsyncWritePayloadRelease:
    def test_write_preloaded_data_clears_payload_lists_after_write(self, tmp_path, monkeypatch):
        bytes_data = [(object(), io.BytesIO(b'abc'))]
        tensor_data = [(object(), torch.arange(4, dtype=torch.float32))]
        write_bucket = (str(tmp_path / 'payload.distcp'), 'storage_key', (bytes_data, tensor_data))

        def fake_write_item(stream, data, write_item, storage_key):
            if isinstance(data, io.BytesIO):
                stream.write(data.getvalue())
                size_in_bytes = len(data.getvalue())
            else:
                stream.write(data.numpy().tobytes())
                size_in_bytes = data.numel() * data.element_size()
            return SimpleNamespace(size_in_bytes=size_in_bytes)

        monkeypatch.setattr(filesystem_async, '_write_item', fake_write_item)

        local_proc_idx, local_results = FileSystemWriterAsync.write_preloaded_data(
            [], 0, write_bucket, results_queue=None, count_queue=None, use_fsync=False
        )

        assert local_proc_idx == 0
        assert len(local_results) == 2
        assert bytes_data == []
        assert tensor_data == []

    def test_write_preloaded_data_profiles_rss_before_and_after(
        self, tmp_path, monkeypatch, caplog
    ):
        bytes_data = []
        tensor_data = [(object(), torch.arange(4, dtype=torch.float32))]
        write_bucket = (str(tmp_path / 'payload.distcp'), 'storage_key', (bytes_data, tensor_data))

        def fake_write_item(stream, data, write_item, storage_key):
            stream.write(data.numpy().tobytes())
            return SimpleNamespace(size_in_bytes=data.numel() * data.element_size())

        memory_values = iter([100 * 1024 * 1024, 120 * 1024 * 1024])
        monkeypatch.setattr(filesystem_async, '_write_item', fake_write_item)
        monkeypatch.setattr(filesystem_async, '_process_memory', lambda: next(memory_values))
        monkeypatch.setenv('MEGATRON_DIST_CKPT_WRITER_PROFILE', '1')

        with caplog.at_level('WARNING', logger=filesystem_async.__name__):
            FileSystemWriterAsync.write_preloaded_data(
                [], 0, write_bucket, results_queue=None, count_queue=None, use_fsync=False
            )

        assert 'rss_before_mb=100.0' in caplog.text
        assert 'rss_after_mb=120.0' in caplog.text


class TestCachedMetadata:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_cached_metadata(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel(2, 4)

        sharded_state_dict_non_cached = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        sharded_state_dict_cached = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', torch.ones(2, 4), replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
            ),
        }

        loaded_non_cached, loaded_cached = None, None
        md_non_cached, md_cached = None, None
        with TempNamedDir(tmp_path_dist_ckpt / 'ckpt_dir') as ckpt_dir:
            save(sharded_state_dict_non_cached, ckpt_dir, async_sharded_save=False)
            loaded_non_cached = load(sharded_state_dict_non_cached, ckpt_dir)
            md_path = ckpt_dir / '.metadata'
            with md_path.open('rb') as f:
                md_non_cached = pickle.load(f)

        save_strategy = deepcopy(TorchDistSaveShardedStrategy())
        save_strategy.use_cached_ckpt_structure = True
        # Run over 3 iterations with cached metadata enabled
        # The 3rd iteration will run with cached metadata
        # `ckpt_dir` at the 3rd iteration 2 will be maintained for comparison
        ckpt_dir = None
        for i in range(3):
            ckpt_dir = TempNamedDir(tmp_path_dist_ckpt / f'ckpt_dir_${i}_cached')
            save(
                sharded_state_dict_cached,
                ckpt_dir.__enter__(),
                save_strategy,
                async_sharded_save=False,
            )
            if i < 2:
                ckpt_dir.cleanup()
        loaded_cached = load(sharded_state_dict_cached, ckpt_dir.__enter__())
        md_path = ckpt_dir.__enter__() / '.metadata'

        with md_path.open('rb') as f:
            md_cached = pickle.load(f)

        # Check loaded state dict
        diffs = diff(loaded_non_cached, loaded_cached)

        assert not any(
            len(x) for x in diffs
        ), 'Cached metadata doesn\'t produce the same state_dict in loading'
        # Check metadata recorded in .metadata, torch.distributed.metadata.Metadata
        for field in fields(md_non_cached):
            if field.name not in ['storage_data', 'storage_meta']:
                diffs = diff(getattr(md_non_cached, field.name), getattr(md_cached, field.name))
                assert not any(
                    len(x) for x in diffs
                ), f'{field.name} is different in metadata from non-cached, cached metadata impls'
        ckpt_dir.cleanup()
        Utils.destroy_model_parallel()

    def test_threaded_load(self, tmp_path_dist_ckpt):
        Utils.initialize_model_parallel()

        def make_sharded_state_dict():
            return {
                'sd_keyA': ShardedTensor.from_rank_offsets(
                    'keyA', torch.ones(2, 4), replica_id=Utils.rank
                ),
                'sd_keyB': ShardedTensor.from_rank_offsets(
                    'keyB', torch.ones(3, 5, 7), replica_id=Utils.world_size - Utils.rank - 1
                ),
            }

        with TempNamedDir(tmp_path_dist_ckpt / 'threaded_load_ckpt_dir') as ckpt_dir:
            save(make_sharded_state_dict(), ckpt_dir, async_sharded_save=False)

            loaded_regular = load(make_sharded_state_dict(), ckpt_dir)
            loaded_threaded = load(
                make_sharded_state_dict(),
                ckpt_dir,
                TorchDistLoadShardedStrategy(cache_metadata=True, thread_count=2),
            )

            os.environ["MEGATRON_DIST_CKPT_READER_PROFILE"] = "1"
            try:
                loaded_profiled = load(
                    make_sharded_state_dict(),
                    ckpt_dir,
                    TorchDistLoadShardedStrategy(cache_metadata=True, thread_count=2),
                )
            finally:
                os.environ.pop("MEGATRON_DIST_CKPT_READER_PROFILE", None)

        diffs = diff(loaded_regular, loaded_threaded)
        assert not any(len(x) for x in diffs), 'Threaded load changed loaded state_dict values'
        diffs = diff(loaded_regular, loaded_profiled)
        assert not any(len(x) for x in diffs), 'Profiled threaded load changed loaded state_dict values'
        Utils.destroy_model_parallel()


class TestCPUTensors:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_cpu_tensors_dont_take_too_much_space(self, tmp_path_dist_ckpt):
        large_cuda_tensor = torch.ones(1_000_000, dtype=torch.float, device='cuda')
        large_cpu_tensor = torch.ones(1_000_000, dtype=torch.float)
        # Create small tensors which are a view of a large tensor
        sharded_state_dict = {
            'sd_keyA': ShardedTensor.from_rank_offsets(
                'keyA', large_cuda_tensor[:10], replica_id=Utils.rank
            ),
            'sd_keyB': ShardedTensor.from_rank_offsets(
                'keyB', large_cpu_tensor[:10], replica_id=Utils.rank
            ),
        }

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_cpu_tensors_dont_take_too_much_space'
        ) as ckpt_dir:
            save(sharded_state_dict, ckpt_dir)

            distcp_files = [(ckpt_dir / '__0_0.distcp')]
            for file in distcp_files:
                assert file.exists()
                file_size = file.stat().st_size
                assert file_size < 10_000, file.name
