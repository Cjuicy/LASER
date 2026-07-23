from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine import streaming_window_engine_lc as lc_module
from inference_engine.streaming_window_engine import STOP_SIGNAL
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC
from inference_engine.utils.geometry import accumulate_sim3


def make_window(depth):
    local_points = torch.full((1, 2, 1, 1, 3), depth)
    camera_poses = torch.eye(4).repeat(1, 2, 1, 1)
    confidence = torch.ones((1, 2, 1, 1))
    return {
        "local_points": local_points,
        "camera_poses": camera_poses,
        "conf": confidence,
    }


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    return value


def clone_cache(cache):
    return {key: clone_value(value) for key, value in cache.items()}


def assert_sim3_close(actual, expected):
    actual_scale, actual_rotation, actual_translation = actual
    expected_scale, expected_rotation, expected_translation = expected
    assert torch.as_tensor(actual_scale).item() == pytest.approx(
        torch.as_tensor(expected_scale).item()
    )
    torch.testing.assert_close(actual_rotation, expected_rotation)
    torch.testing.assert_close(actual_translation, expected_translation)


def run_three_window_fixture(monkeypatch, tmp_path):
    engine = StreamingWindowEngineLC(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        window_size=2,
        overlap=1,
        depth_refine=True,
    )
    caches = []
    registration_sources = []
    registration_scales = iter((2.0, 4.0))
    refinement_scales = iter((3.0, 5.0))

    monkeypatch.setattr(
        lc_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda points: (None, torch.eye(3)[None]),
    )

    def fake_unproject(depth, intrinsic):
        return depth[..., None].repeat(1, 1, 1, 3)

    monkeypatch.setattr(
        lc_module,
        "unproject_depth_to_local_points",
        fake_unproject,
    )

    def fake_register(source_points, target_points, *args):
        registration_sources.append(source_points.clone())
        return (
            next(registration_scales),
            torch.eye(3),
            torch.zeros(3),
        )

    monkeypatch.setattr(
        lc_module,
        "register_adjacent_windows",
        fake_register,
    )

    def fake_refine(*args):
        return torch.full((2, 1, 1, 1), next(refinement_scales))

    monkeypatch.setattr(
        lc_module,
        "refine_depth_segments",
        fake_refine,
    )
    engine._build_segment_graph = lambda *args: object()
    engine._save_cache = lambda: caches.append(
        clone_cache(engine.prev_window_cache)
    )

    for depth in (1.0, 1.0, 1.0):
        engine.registration_queue.put((make_window(depth), 0.0))
    engine.registration_queue.put(STOP_SIGNAL)
    engine._registration_worker()
    return caches, registration_sources


def test_lc_worker_uses_corrected_previous_window_for_next_registration(
    monkeypatch,
    tmp_path,
):
    caches, registration_sources = run_three_window_fixture(
        monkeypatch,
        tmp_path,
    )

    assert len(caches) == 3
    torch.testing.assert_close(
        registration_sources[1],
        torch.full_like(registration_sources[1], 6.0),
    )


def test_lc_worker_applies_scale_mask_immediately(monkeypatch, tmp_path):
    caches, _ = run_three_window_fixture(monkeypatch, tmp_path)

    torch.testing.assert_close(
        caches[1]["local_points"],
        torch.full_like(caches[1]["local_points"], 6.0),
    )
    torch.testing.assert_close(
        caches[1]["scale_mask"],
        torch.full_like(caches[1]["scale_mask"], 3.0),
    )


def test_lc_cache_stores_absolute_and_relative_sim3(monkeypatch, tmp_path):
    caches, _ = run_three_window_fixture(monkeypatch, tmp_path)

    assert "sim3_abs" in caches[0]
    assert "sim3_edge" not in caches[0]
    assert_sim3_close(
        caches[0]["sim3_abs"],
        (1.0, torch.eye(3), torch.zeros(3)),
    )

    for index in (1, 2):
        recomposed = accumulate_sim3(
            caches[index - 1]["sim3_abs"],
            caches[index]["sim3_edge"],
        )
        assert_sim3_close(recomposed, caches[index]["sim3_abs"])
