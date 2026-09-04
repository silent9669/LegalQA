from unittest.mock import MagicMock, patch
import pytest

from src.task2.generation.memory import (
    cleanup_cuda_stage,
    snapshot_cuda_memory,
    TrainerMemoryCallback,
)


def test_cleanup_cuda_stage_cpu_safe():
    # Should run without error on non-CUDA or CUDA systems
    x = [1, 2, 3]
    cleanup_cuda_stage(x, devices=(0, 1))


def test_cleanup_cuda_stage_mocked_cuda():
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2

    with patch("src.task2.generation.memory.torch.cuda", mock_cuda):
        dummy_obj = {"large": "tensor"}
        cleanup_cuda_stage(dummy_obj, devices=(0, 1))

        assert mock_cuda.empty_cache.call_count >= 1


def test_snapshot_cuda_memory_mocked():
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mock_cuda.memory_allocated.side_effect = lambda dev: 2000 * 1024 * 1024 if dev == 0 else 500 * 1024 * 1024
    mock_cuda.memory_reserved.side_effect = lambda dev: 2200 * 1024 * 1024 if dev == 0 else 600 * 1024 * 1024
    mock_cuda.max_memory_allocated.side_effect = lambda dev: 2500 * 1024 * 1024 if dev == 0 else 700 * 1024 * 1024
    mock_cuda.mem_get_info.side_effect = lambda dev: (12000 * 1024 * 1024, 15000 * 1024 * 1024)

    with patch("src.task2.generation.memory.torch.cuda", mock_cuda):
        snap = snapshot_cuda_memory("test_probe", devices=(0, 1))

        assert snap["label"] == "test_probe"
        assert 0 in snap["devices"]
        assert 1 in snap["devices"]
        assert snap["devices"][0]["allocated_mb"] == pytest.approx(2000.0)
        assert snap["devices"][0]["reserved_mb"] == pytest.approx(2200.0)
        assert snap["devices"][0]["max_allocated_mb"] == pytest.approx(2500.0)


def test_trainer_memory_callback_records_steps():
    cb = TrainerMemoryCallback(log_every_n_steps=50)

    mock_state = MagicMock()
    mock_state.global_step = 50

    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mock_cuda.memory_allocated.return_value = 1000 * 1024 * 1024
    mock_cuda.memory_reserved.return_value = 1200 * 1024 * 1024
    mock_cuda.max_memory_allocated.return_value = 1500 * 1024 * 1024
    mock_cuda.mem_get_info.return_value = (10000 * 1024 * 1024, 15000 * 1024 * 1024)

    with patch("src.task2.generation.memory.torch.cuda", mock_cuda):
        cb.on_step_end(args=MagicMock(), state=mock_state, control=MagicMock())

        assert len(cb.history) == 1
        assert cb.history[0]["step"] == 50
        assert cb.history[0]["snapshot"]["devices"][0]["allocated_mb"] == pytest.approx(1000.0)

        # Step 51 should not record (not multiple of 50)
        mock_state.global_step = 51
        cb.on_step_end(args=MagicMock(), state=mock_state, control=MagicMock())
        assert len(cb.history) == 1
