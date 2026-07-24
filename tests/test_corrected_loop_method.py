import numpy as np
import pytest
import torch

from inference_engine.segmentation import SegmentationResult
from inference_engine.streaming_window_engine import STOP_SIGNAL
from inference_engine.utils.geometry import (
    accumulate_sim3,
    closed_form_inverse_sim3,
)
from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopCandidate,
    LoopConstraint,
    LoopSolution,
    WindowCache,
)
from loop_closure.methods.corrected import (
    CorrectedLoopClosureStrategy,
    CorrectedWindowEngine,
    build_local_loop_constraint,
)
from loop_closure.methods.registry import (
    LOOP_STRATEGIES,
    build_loop_strategy,
)
from pipeline.config import LoopMethod, SegmentationMethod, load_pipeline_config


def sim3(scale=1.0, translation=None):
    return (
        scale,
        torch.eye(3),
        torch.zeros(3)
        if translation is None
        else torch.as_tensor(translation, dtype=torch.float32),
    )


class OneRegionStrategy:
    name = SegmentationMethod.ATOMIC

    def segment(self, point_maps, confidence, images):
        return [
            SegmentationResult(
                labels=np.zeros(point_maps.shape[1:3], dtype=np.intp),
                diagnostics={"method": "atomic", "region_count": 1},
            )
            for _ in point_maps
        ]


class SequenceAnchor:
    def __init__(self, scales):
        self.scales = iter(scales)

    def propagate(self, source_points, target_points, *args):
        return torch.full(
            (*target_points.shape[:-1], 1),
            next(self.scales),
        )


def make_window(depth=1.0):
    points = torch.full((1, 2, 1, 1, 3), depth)
    return {
        "local_points": points,
        "camera_poses": torch.eye(4).repeat(1, 2, 1, 1),
        "conf": torch.ones((1, 2, 1, 1)),
        "images": torch.zeros((1, 2, 3, 1, 1)),
    }


def run_corrected_windows(monkeypatch, tmp_path, count=3):
    from loop_closure.methods import corrected as corrected_module

    engine = CorrectedWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=OneRegionStrategy(),
        anchor_propagator=SequenceAnchor((3.0, 5.0)),
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
        corrected_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda points: (points[..., 2], torch.eye(3)[None]),
    )
    monkeypatch.setattr(
        corrected_module,
        "unproject_depth_to_local_points",
        lambda depth, intrinsic: depth[..., None].repeat(1, 1, 1, 3),
    )
    registration_sources = []
    registration_scales = iter((2.0, 4.0))

    def fake_register(source_points, *args):
        registration_sources.append(source_points.clone())
        return sim3(next(registration_scales))

    monkeypatch.setattr(
        corrected_module,
        "register_adjacent_windows",
        fake_register,
    )
    caches = []
    engine._save_cache = lambda: (
        caches.append(engine.prev_window_cache),
        setattr(engine, "cache_id", engine.cache_id + 1),
    )
    for _ in range(count):
        engine.registration_queue.put((make_window(), 0.0))
    engine.registration_queue.put(STOP_SIGNAL)
    engine._registration_worker()
    return engine, caches, registration_sources


def test_corrected_window_uses_corrected_previous_window_for_registration(
    monkeypatch,
    tmp_path,
):
    _, _, registration_sources = run_corrected_windows(
        monkeypatch,
        tmp_path,
    )
    torch.testing.assert_close(
        registration_sources[-1],
        torch.full_like(registration_sources[-1], 6.0),
    )


def test_corrected_window_applies_anchor_scale_immediately(
    monkeypatch,
    tmp_path,
):
    _, caches, _ = run_corrected_windows(monkeypatch, tmp_path, count=2)
    cache = caches[1]
    assert cache.loop_state["anchor_scale_applied"] is True
    torch.testing.assert_close(
        cache.local_points,
        torch.full_like(cache.local_points, 6.0),
    )


def test_corrected_cache_has_absolute_and_relative_sim3(
    monkeypatch,
    tmp_path,
):
    _, caches, _ = run_corrected_windows(monkeypatch, tmp_path, count=2)
    state = caches[1].loop_state
    assert state["tag"] == "corrected"
    assert "sim3_abs" in state
    assert "sim3_edge" in state


def test_corrected_loop_measurement_is_local_coordinate_constraint():
    absolute_a = sim3(2.0)
    absolute_b = sim3(4.0)
    global_a = sim3()
    global_b = sim3()
    constraint = build_local_loop_constraint(
        absolute_a,
        absolute_b,
        global_a,
        global_b,
    )
    sequential = accumulate_sim3(
        closed_form_inverse_sim3(*absolute_a),
        absolute_b,
    )
    residual = accumulate_sim3(constraint, sequential)
    assert torch.as_tensor(residual[0]).item() == pytest.approx(1.0)
    torch.testing.assert_close(residual[1], torch.eye(3))
    torch.testing.assert_close(residual[2], torch.zeros(3))


def corrected_caches():
    first = WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=LoopMethod.CORRECTED,
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
            "tag": "corrected",
            "sim3_abs": sim3(),
            "anchor_scale_applied": True,
        },
    )
    second = WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=LoopMethod.CORRECTED,
        window_index=1,
        frame_start=1,
        frame_end=3,
        local_points=torch.full((2, 1, 1, 3), 6.0),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        segmentation_labels=(
            np.zeros((1, 1), dtype=np.intp),
            np.zeros((1, 1), dtype=np.intp),
        ),
        anchor_scale_mask=torch.full((2, 1, 1, 1), 3.0),
        loop_state={
            "tag": "corrected",
            "sim3_abs": sim3(2.0),
            "sim3_edge": sim3(2.0),
            "anchor_scale_applied": True,
        },
    )
    return first, second


def strategy_fixture(optimizer=None, constraint_estimator=None):
    config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return CorrectedLoopClosureStrategy(
        optimizer_config=config,
        registration_confidence_keep_ratio=0.3,
        optimizer=optimizer,
        constraint_estimator=constraint_estimator,
    )


def test_corrected_aggregation_applies_only_optimization_delta_once():
    caches = corrected_caches()
    solution = LoopSolution(
        optimized_transforms=(sim3(), sim3(4.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    result = strategy_fixture().aggregate(caches, solution)
    torch.testing.assert_close(
        result.payload["local_points"][-1],
        torch.full_like(result.payload["local_points"][-1], 12.0),
    )
    assert result.summary["max_abs_log_scale_delta"] == pytest.approx(
        np.log(2.0)
    )


def test_corrected_no_loop_returns_original_absolute_transforms():
    caches = corrected_caches()
    solution = strategy_fixture().optimize(caches, [])
    assert solution.used_no_loop_path is True
    assert solution.optimized_transforms == tuple(
        cache.loop_state["sim3_abs"] for cache in caches
    )


def test_corrected_optimizer_receives_one_edge_per_window_transition():
    class RecordingOptimizer:
        def __init__(self):
            self.edges = None

        def optimize(self, edges, constraints):
            self.edges = edges
            return edges

    optimizer = RecordingOptimizer()
    caches = corrected_caches()
    constraint = LoopConstraint(
        window_a=1,
        window_b=0,
        measurement=sim3(),
        candidate=LoopCandidate(2, 0, 0.8),
    )
    strategy_fixture(optimizer).optimize(caches, [constraint])
    assert len(optimizer.edges) == len(caches) - 1


def test_corrected_aggregate_rejects_cache_count_mismatch():
    with pytest.raises(ValueError, match="count"):
        strategy_fixture().aggregate(
            corrected_caches(),
            LoopSolution(
                optimized_transforms=(sim3(),),
                constraints=(),
                used_no_loop_path=True,
            ),
        )


def test_loop_registry_has_exactly_two_methods():
    assert set(LOOP_STRATEGIES) == {
        LoopMethod.TRADITIONAL,
        LoopMethod.CORRECTED,
    }
    assert build_loop_strategy(
        LoopMethod.CORRECTED,
        optimizer_config=load_pipeline_config(
            "configs/pipeline/test.yaml",
            ("loop.optimizer.implementation=python",),
        ).config.loop.optimizer,
        registration_confidence_keep_ratio=0.3,
    ).name is LoopMethod.CORRECTED
