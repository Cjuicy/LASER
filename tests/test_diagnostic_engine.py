from pathlib import Path

import pytest
import torch

from inference_engine.streaming_window_engine import StreamingWindowEngine


def _engine(tmp_path, **kwargs):
    return StreamingWindowEngine(
        torch.nn.Identity(), inference_device="cpu", dtype=torch.float32,
        process_device="cpu", intermediate_device="cpu", cache_root=str(tmp_path),
        benchmark_latency=False, depth_refine=False, **kwargs,
    )


def _cache(frame_start=0):
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.arange(frame_start, frame_start + 3)
    return {
        "camera_poses": poses,
        "local_points": torch.ones((3, 1, 1, 3)),
        "conf": torch.ones((3, 1, 1)),
        "images": torch.ones((3, 3, 1, 1)),
    }


def test_cache_policy_defaults_to_existing_full_behavior(tmp_path):
    engine = _engine(tmp_path)
    assert engine.cache_policy == "full"


def test_invalid_cache_policy_fails_before_delegate_is_moved(tmp_path):
    class Delegate(torch.nn.Module):
        def to(self, *args, **kwargs):
            raise AssertionError("delegate must not be touched")
    with pytest.raises(ValueError, match="cache_policy"):
        StreamingWindowEngine(Delegate(), "cpu", torch.float32, cache_root=str(tmp_path), cache_policy="bad")


def test_metrics_only_cache_writes_pose_shards_without_mutating_memory_cache(tmp_path):
    engine = _engine(tmp_path, cache_policy="metrics-only", window_size=3, overlap=1)
    engine.temp_cache_dir = Path(tmp_path)
    engine.prev_window_cache = _cache()
    engine._save_cache()

    saved = torch.load(tmp_path / "window_cache_0.pt", weights_only=False)
    assert set(saved) == {"camera_poses", "window_id", "frame_start"}
    assert saved["window_id"] == 0
    assert saved["frame_start"] == 0
    assert "local_points" in engine.prev_window_cache


def test_full_cache_keeps_existing_dictionary_fields(tmp_path):
    engine = _engine(tmp_path, cache_policy="full")
    engine.temp_cache_dir = Path(tmp_path)
    engine.prev_window_cache = _cache()
    engine._save_cache()
    saved = torch.load(tmp_path / "window_cache_0.pt", weights_only=False)
    assert set(saved) == set(engine.prev_window_cache)


def test_pose_summary_trims_later_overlap_and_returns_extrinsic(tmp_path):
    engine = _engine(tmp_path, cache_policy="metrics-only", window_size=3, overlap=1)
    engine.temp_cache_dir = Path(tmp_path)
    engine.prev_window_cache = _cache(0)
    engine._save_cache()
    engine.prev_window_cache = _cache(2)
    engine._save_cache()

    summary = engine.parse_pose_cache_summary(remove_cache=False)

    assert list(summary) == ["extrinsic"]
    assert summary["extrinsic"].shape == (1, 5, 4, 4)
    assert summary["extrinsic"][0, :, 0, 3].tolist() == [0, 1, 2, 3, 4]


def test_diagnostic_ids_are_required_only_when_sink_is_enabled(tmp_path):
    _engine(tmp_path / "off", diagnostic_pass=0)
    with pytest.raises(ValueError, match="diagnostic_run_id"):
        _engine(tmp_path / "on", diagnostic_sink=object(), diagnostic_pass=1, diagnostic_sequence_id="02")
