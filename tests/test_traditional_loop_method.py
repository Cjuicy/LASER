import numpy as np
import pytest
import torch

from inference_engine.segmentation import SegmentationResult
from inference_engine.streaming_window_engine import STOP_SIGNAL
from loop_closure.methods import shared
from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopCandidate,
    LoopSolution,
    WindowCache,
)
from loop_closure.methods.traditional import (
    TraditionalLoopClosureStrategy,
    TraditionalWindowEngine,
)
from pipeline.config import LoopMethod, SegmentationMethod, load_pipeline_config


def identity_sim3(scale=1.0):
    return scale, torch.eye(3), torch.zeros(3)


class OneRegionStrategy:
    name = SegmentationMethod.DEPTH

    def segment(self, point_maps, confidence, images):
        return [
            SegmentationResult(
                labels=np.zeros(point_maps.shape[1:3], dtype=np.intp),
                diagnostics={"method": "depth", "region_count": 1},
            )
            for _ in point_maps
        ]


class ConstantAnchor:
    def __init__(self, scale=3.0):
        self.scale = scale

    def propagate(self, source_points, target_points, *args):
        return torch.full(
            (*target_points.shape[:-1], 1),
            self.scale,
        )


def make_window(depth=1.0):
    frames, height, width = 2, 1, 1
    points = torch.full((1, frames, height, width, 3), depth)
    return {
        "local_points": points,
        "camera_poses": torch.eye(4).repeat(1, frames, 1, 1),
        "conf": torch.ones((1, frames, height, width)),
        "images": torch.zeros((1, frames, 3, height, width)),
    }


def test_traditional_window_defers_sim3_and_anchor_scale_application(
    monkeypatch,
    tmp_path,
):
    from loop_closure.methods import traditional as traditional_module

    engine = TraditionalWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=OneRegionStrategy(),
        anchor_propagator=ConstantAnchor(scale=3.0),
        registration_confidence_keep_ratio=0.5,
        anchor_enabled=True,
        temporal_iou_threshold=0.3,
        window_size=2,
        overlap=1,
        cache_root=str(tmp_path),
        intermediate_device="cpu",
        process_device="cpu",
        benchmark_latency=False,
    )
    monkeypatch.setattr(
        traditional_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda points: (points[..., 2], torch.eye(3)[None]),
    )
    monkeypatch.setattr(
        traditional_module,
        "unproject_depth_to_local_points",
        lambda depth, intrinsic: depth[..., None].repeat(1, 1, 1, 3),
    )
    monkeypatch.setattr(
        traditional_module,
        "register_adjacent_windows",
        lambda *args: identity_sim3(scale=2.0),
    )
    caches = []
    engine._save_cache = lambda: caches.append(engine.prev_window_cache)

    engine.registration_queue.put((make_window(), 0.0))
    engine.registration_queue.put((make_window(), 0.0))
    engine.registration_queue.put(STOP_SIGNAL)
    engine._registration_worker()

    cache = caches[1]
    assert cache.loop_state["tag"] == "traditional"
    assert cache.loop_state["anchor_scale_applied"] is False
    assert torch.as_tensor(
        cache.loop_state["relative_sim3"][0]
    ).item() == pytest.approx(2.0)
    torch.testing.assert_close(
        cache.local_points,
        torch.ones_like(cache.local_points),
    )
    torch.testing.assert_close(
        cache.anchor_scale_mask,
        torch.full_like(cache.anchor_scale_mask, 3.0),
    )


def traditional_cache_fixture():
    first = WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=LoopMethod.TRADITIONAL,
        window_index=0,
        frame_start=0,
        frame_end=2,
        local_points=torch.ones((2, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        segmentation_labels=(
            np.zeros((1, 1), dtype=np.intp),
            np.zeros((1, 1), dtype=np.intp),
        ),
        anchor_scale_mask=None,
        loop_state={
            "tag": "traditional",
            "relative_sim3": identity_sim3(),
            "anchor_scale_applied": False,
        },
    )
    second = WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=LoopMethod.TRADITIONAL,
        window_index=1,
        frame_start=1,
        frame_end=3,
        local_points=torch.ones((2, 1, 1, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        segmentation_labels=(
            np.zeros((1, 1), dtype=np.intp),
            np.zeros((1, 1), dtype=np.intp),
        ),
        anchor_scale_mask=torch.full((2, 1, 1, 1), 3.0),
        loop_state={
            "tag": "traditional",
            "relative_sim3": identity_sim3(scale=2.0),
            "anchor_scale_applied": False,
        },
    )
    return (first, second)


def strategy_fixture(optimizer=None):
    optimizer_config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return TraditionalLoopClosureStrategy(
        optimizer_config=optimizer_config,
        registration_confidence_keep_ratio=0.3,
        optimizer=optimizer,
    )


def test_traditional_aggregation_applies_delayed_transforms_once():
    strategy = strategy_fixture()
    solution = LoopSolution(
        optimized_transforms=(identity_sim3(), identity_sim3(scale=2.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    result = strategy.aggregate(traditional_cache_fixture(), solution)
    assert result.payload["local_points"].shape[0] == 3
    torch.testing.assert_close(
        result.payload["local_points"][-1],
        torch.full_like(result.payload["local_points"][-1], 3.0),
    )
    for cache in traditional_cache_fixture():
        torch.testing.assert_close(
            cache.local_points,
            torch.ones_like(cache.local_points),
        )


def test_traditional_constraint_keeps_baseline_compute_sim3_ab(
    monkeypatch,
):
    from loop_closure.methods import traditional as traditional_module

    calls = []
    expected = identity_sim3(scale=2.0)

    def fake_compute(left, right):
        calls.append((left, right))
        return expected

    monkeypatch.setattr(
        traditional_module,
        "compute_sim3_ab",
        fake_compute,
    )
    monkeypatch.setattr(
        traditional_module,
        "register_adjacent_windows",
        lambda *args: identity_sim3(scale=2.0),
    )
    candidate = (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),)
    constraint = strategy_fixture().build_constraints(
        traditional_cache_fixture(),
        candidate,
    )[0]
    assert calls
    assert constraint.measurement is expected


def test_traditional_uses_positive_shared_confidence_ratio(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shared,
        "select_top_confidence_mask",
        lambda confidence, keep_ratio: calls.append(keep_ratio)
        or torch.ones_like(confidence, dtype=torch.bool),
    )
    from loop_closure.methods import traditional as traditional_module

    monkeypatch.setattr(
        traditional_module,
        "register_adjacent_windows",
        lambda *args: identity_sim3(),
    )
    strategy_fixture().build_constraints(
        traditional_cache_fixture(),
        (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
    )
    assert calls and set(calls) == {0.3}


def test_traditional_no_loop_does_not_invoke_optimizer():
    class FailingOptimizer:
        def optimize(self, *args):
            raise AssertionError("optimizer must not be called")

    caches = traditional_cache_fixture()
    solution = strategy_fixture(FailingOptimizer()).optimize(caches, [])
    assert solution.used_no_loop_path is True
    assert solution.optimized_transforms == tuple(
        cache.loop_state["relative_sim3"] for cache in caches
    )


def test_traditional_cache_payload_round_trips_with_method_tag():
    cache = traditional_cache_fixture()[1]
    restored = WindowCache.from_payload(
        cache.to_payload(),
        expected_method=LoopMethod.TRADITIONAL,
    )
    assert restored.loop_state["tag"] == "traditional"
    assert restored.loop_state["anchor_scale_applied"] is False
