from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine import streaming_window_engine_lc as lc_module
from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.segmentation import build_segmentation_strategy
from inference_engine.streaming_window_engine import STOP_SIGNAL
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC
from inference_engine.utils.geometry import accumulate_sim3
from pipeline.config import load_pipeline_config


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
    segmentation = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("segmentation.method=depth",),
    ).config.segmentation

    class FixturePropagator:
        def __init__(self):
            self.scales = iter((3.0, 5.0))

        def propagate(self, *args):
            return torch.full((2, 1, 1, 1), next(self.scales))

    engine = StreamingWindowEngineLC(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=build_segmentation_strategy(segmentation),
        anchor_propagator=FixturePropagator(),
        registration_confidence_keep_ratio=0.3,
        anchor_enabled=True,
        temporal_iou_threshold=0.3,
        process_device="cpu",
        cache_root=str(tmp_path),
        window_size=2,
        overlap=1,
    )
    caches = []
    registration_sources = []
    registration_scales = iter((2.0, 4.0))

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


def test_aggregate_applies_only_optimization_delta():
    original_local_points = torch.full((1, 1, 1, 3), 6.0)
    original_camera_poses = torch.eye(4)[None]
    original_camera_poses[0, 0, 3] = 1.0
    cache = {
        "local_points": original_local_points.clone(),
        "camera_poses": original_camera_poses.clone(),
        "conf": torch.ones((1, 1, 1)),
        "scale_mask": torch.full((1, 1, 1, 1), 3.0),
        "sim3_abs": (
            2.0,
            torch.eye(3),
            torch.tensor([1.0, 0.0, 0.0]),
        ),
    }
    optimized_absolute = [
        (
            4.0,
            torch.eye(3),
            torch.tensor([5.0, 0.0, 0.0]),
        ),
    ]

    result = StreamingWindowEngineLC.aggregate_caches(
        [cache],
        optimized_absolute,
    )

    torch.testing.assert_close(
        result["local_points"],
        torch.full_like(result["local_points"], 12.0),
    )
    torch.testing.assert_close(
        result["camera_poses"][0, 0, :3, 3],
        torch.tensor([5.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(cache["local_points"], original_local_points)
    torch.testing.assert_close(cache["camera_poses"], original_camera_poses)
    torch.testing.assert_close(
        cache["scale_mask"],
        torch.full_like(cache["scale_mask"], 3.0),
    )


def test_aggregate_defaults_to_identity_optimization_delta():
    window_sim3 = (
        2.0,
        torch.eye(3),
        torch.zeros(3),
    )
    cache = {
        "local_points": torch.full((1, 1, 1, 3), 6.0),
        "camera_poses": torch.eye(4)[None],
        "conf": torch.ones((1, 1, 1)),
        "scale_mask": torch.full((1, 1, 1, 1), 3.0),
        "sim3_abs": window_sim3,
    }

    result = StreamingWindowEngineLC.aggregate_caches([cache])

    torch.testing.assert_close(
        result["local_points"],
        torch.full_like(result["local_points"], 6.0),
    )
    assert_sim3_close(result["sim3_abs"][0], window_sim3)


def test_aggregate_rejects_transform_count_mismatch():
    cache = {
        "local_points": torch.ones((1, 1, 1, 3)),
        "camera_poses": torch.eye(4)[None],
        "conf": torch.ones((1, 1, 1)),
        "sim3_abs": (1.0, torch.eye(3), torch.zeros(3)),
    }

    with pytest.raises(
        ValueError,
        match="optimized transform count does not match cache count",
    ):
        StreamingWindowEngineLC.aggregate_caches([cache], [])


def test_apply_optimization_deltas_updates_cache_transform_contract():
    caches = [
        {
            "local_points": torch.ones((1, 1, 1, 3)),
            "camera_poses": torch.eye(4)[None],
            "conf": torch.ones((1, 1, 1)),
            "sim3_abs": (1.0, torch.eye(3), torch.zeros(3)),
        },
        {
            "local_points": torch.full((1, 1, 1, 3), 6.0),
            "camera_poses": torch.eye(4)[None],
            "conf": torch.ones((1, 1, 1)),
            "sim3_abs": (2.0, torch.eye(3), torch.zeros(3)),
            "sim3_edge": (2.0, torch.eye(3), torch.zeros(3)),
        },
    ]
    optimized_absolute = [
        (1.0, torch.eye(3), torch.zeros(3)),
        (4.0, torch.eye(3), torch.zeros(3)),
    ]

    adjusted = StreamingWindowEngineLC.apply_optimization_deltas(
        caches,
        optimized_absolute,
    )

    torch.testing.assert_close(
        adjusted[1]["local_points"],
        torch.full_like(adjusted[1]["local_points"], 12.0),
    )
    assert_sim3_close(
        adjusted[1]["sim3_abs"],
        optimized_absolute[1],
    )
    recomposed = accumulate_sim3(
        adjusted[0]["sim3_abs"],
        adjusted[1]["sim3_edge"],
    )
    assert_sim3_close(recomposed, adjusted[1]["sim3_abs"])
    torch.testing.assert_close(
        caches[1]["local_points"],
        torch.full_like(caches[1]["local_points"], 6.0),
    )
