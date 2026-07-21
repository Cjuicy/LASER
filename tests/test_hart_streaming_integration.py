from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_engine import streaming_window_engine as swe_module
from inference_engine import streaming_window_engine_lc as swe_lc_module
from inference_engine.inference_utils import register_adjacent_window_pose
from inference_engine.anchor_propagation.segmentation import _frame_from_layers
from inference_engine.anchor_propagation.types import (
    AnchorPropagationState,
    PropagationResult,
    RegistrationState,
    SegmentationWindow,
)
from inference_engine.streaming_window_engine import StreamingWindowEngine, STOP_SIGNAL
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC


def _engine(tmp_path, **kwargs):
    return StreamingWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        benchmark_latency=False,
        window_size=2,
        overlap=1,
        **kwargs,
    )


def _window(depth):
    points = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
    points[..., -1] = depth
    return {
        "local_points": points,
        "conf": torch.ones((2, 2, 3), dtype=torch.float32),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
    }


def _patch_registration_primitives(monkeypatch, captured_sources=None):
    monkeypatch.setattr(
        swe_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda points: (None, torch.eye(3)[None]),
    )

    def unproject(depth, intrinsic):
        points = torch.zeros((*depth.shape, 3), dtype=depth.dtype)
        points[..., -1] = depth
        return points

    monkeypatch.setattr(swe_module, "unproject_depth_to_local_points", unproject)

    def register(src_points, tgt_points, src_poses, tgt_poses, mask):
        if captured_sources is not None:
            captured_sources.append(src_points.detach().clone())
        return 1.0, torch.eye(3), torch.zeros(3)

    monkeypatch.setattr(swe_module, "register_adjacent_windows", register)
    monkeypatch.setattr(
        swe_module,
        "register_adjacent_window_pose",
        lambda *args: (torch.eye(3), torch.zeros(3)),
    )


def _run_worker(engine, windows):
    engine._save_cache = lambda: None
    for window in windows:
        engine.registration_queue.put((window, 0.0))
    engine.registration_queue.put(STOP_SIGNAL)
    engine._registration_worker()


def _fake_result(kwargs, *, window_scale=1.0, residual=1.0, support=False):
    shape = kwargs["current_base_points"].shape[:3]
    residual_map = np.broadcast_to(
        np.asarray(residual, dtype=np.float32), shape
    ).copy()
    support_map = np.broadcast_to(np.asarray(support, dtype=bool), shape).copy()
    next_state = AnchorPropagationState(
        local_residual_tail=residual_map[-1:].copy(),
        confidence_tail=np.ones(shape[-3:], dtype=np.float32)[-1:].copy(),
        segments_tail=(),
    )
    return PropagationResult(
        window_scale=float(window_scale),
        local_residual_mask=residual_map[..., None],
        pose_support_mask=support_map,
        next_state=next_state,
        diagnostics={"window_scale": float(window_scale)},
    )


def test_fixed_scale_pose_solver_preserves_registration_direction():
    previous_poses = torch.eye(4).repeat(2, 1, 1)
    current_poses = torch.eye(4).repeat(2, 1, 1)
    calls = []

    def solve(src_poses, tgt_poses, *, scale):
        calls.append((src_poses, tgt_poses, scale))
        return torch.eye(3), torch.ones(3)

    rotation, translation = register_adjacent_window_pose(
        previous_poses,
        current_poses,
        2.0,
        register_func=solve,
    )

    assert calls == [(current_poses, previous_poses, 2.0)]
    np.testing.assert_array_equal(rotation, torch.eye(3))
    np.testing.assert_array_equal(translation, torch.ones(3))


def _set_hart_tail_state(engine, *, residual, support):
    points = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    points[..., -1] = 3.0
    engine.registration_state = RegistrationState(
        final_base_points_tail=points,
        final_base_poses_tail=torch.eye(4)[None],
        pose_support_mask_tail=np.asarray(support, dtype=bool).reshape(1, 2, 3),
    )
    engine.anchor_propagation_state = AnchorPropagationState(
        local_residual_tail=np.asarray(residual, dtype=np.float32).reshape(1, 2, 3),
        confidence_tail=np.ones((1, 2, 3), dtype=np.float32),
        segments_tail=(),
    )
    return points


def test_registration_support_uses_base_times_residual(tmp_path):
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=2,
    )
    fallback = _set_hart_tail_state(
        engine,
        residual=[2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        support=[True, False, False, True, False, False],
    )

    points, mask, diagnostics = engine._select_hart_registration_input(
        fallback,
        torch.ones((1, 2, 3), dtype=torch.bool),
    )

    np.testing.assert_array_equal(
        points[..., -1],
        [[[6.0, 3.0, 3.0], [6.0, 3.0, 3.0]]],
    )
    np.testing.assert_array_equal(
        mask,
        [[[True, False, False], [True, False, False]]],
    )
    assert diagnostics == {
        "registration_pose_support_pixels": 2,
        "registration_pose_support_used": True,
        "registration_pose_support_fallback_count": 0,
    }


def test_registration_support_falls_back_to_unmodified_base(tmp_path):
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=3,
    )
    fallback = _set_hart_tail_state(
        engine,
        residual=[2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        support=[True, False, False, True, False, False],
    )
    mutual = torch.ones((1, 2, 3), dtype=torch.bool)

    points, mask, diagnostics = engine._select_hart_registration_input(
        fallback,
        mutual,
    )

    np.testing.assert_array_equal(points[..., -1], 3.0)
    np.testing.assert_array_equal(mask, mutual)
    assert diagnostics == {
        "registration_pose_support_pixels": 2,
        "registration_pose_support_used": False,
        "registration_pose_support_fallback_count": 1,
    }


def test_commit_hart_state_waits_for_final_base_and_pose(tmp_path):
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    final_base = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
    final_base[..., -1] = 7.0
    final_poses = torch.eye(4).repeat(2, 1, 1)
    final_poses[:, 0, 3] = torch.tensor([4.0, 5.0])
    next_state = AnchorPropagationState(
        local_residual_tail=np.ones((1, 2, 3), dtype=np.float32),
        confidence_tail=np.ones((1, 2, 3), dtype=np.float32),
        segments_tail=(),
    )
    result = PropagationResult(
        window_scale=1.0,
        local_residual_mask=np.ones((2, 2, 3, 1), dtype=np.float32),
        pose_support_mask=np.asarray(
            [
                [[False, False, False], [False, False, False]],
                [[True, False, False], [True, False, False]],
            ]
        ),
        next_state=next_state,
        diagnostics={},
    )

    engine._commit_hart_state(result, final_base, final_poses)

    np.testing.assert_array_equal(
        engine.registration_state.final_base_points_tail[..., -1], 7.0
    )
    np.testing.assert_array_equal(
        engine.registration_state.final_base_poses_tail[:, 0, 3], [5.0]
    )
    np.testing.assert_array_equal(
        engine.registration_state.pose_support_mask_tail,
        [[[True, False, False], [True, False, False]]],
    )
    assert engine.anchor_propagation_state is next_state


def test_none_route_runs_no_segmentation_or_local_refinement(monkeypatch, tmp_path):
    _patch_registration_primitives(monkeypatch)
    engine = _engine(tmp_path, depth_refine=True, anchor_propagation="none")
    engine._build_segment_graph = lambda *args: pytest.fail("legacy graph called")
    engine._build_hart_segments = lambda *args: pytest.fail("HART builder called")

    _run_worker(engine, [_window(3.0)])

    np.testing.assert_array_equal(engine.prev_window_cache["local_points"][..., -1], 3.0)


def test_legacy_route_keeps_graph_and_refine_contract(monkeypatch, tmp_path):
    _patch_registration_primitives(monkeypatch)
    calls = []
    engine = _engine(tmp_path, depth_refine=True)
    engine._build_segment_graph = lambda *args: calls.append("graph") or object()

    def refine(*args):
        calls.append("refine")
        return torch.ones((2, 2, 3, 1))

    monkeypatch.setattr(swe_module, "refine_depth_segments", refine)
    engine._build_hart_segments = lambda *args: pytest.fail("HART builder called")

    _run_worker(engine, [_window(3.0), _window(3.0)])

    assert calls == ["graph", "graph", "refine"]


def test_ordinary_hart_registration_uses_only_pose_support_residual(
    monkeypatch, tmp_path
):
    _patch_registration_primitives(monkeypatch)
    captured = []

    def register(src_points, tgt_points, src_poses, tgt_poses, mask):
        captured.append((src_points.detach().clone(), mask.detach().clone()))
        return 1.0, torch.eye(3), torch.zeros(3)

    monkeypatch.setattr(swe_module, "register_adjacent_windows", register)
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    engine._build_segment_graph = lambda *args: pytest.fail("legacy graph called")
    engine._build_hart_segments = lambda *args: SimpleNamespace()
    calls = []
    residual = np.asarray(
        [[[2.0, 4.0, 4.0], [2.0, 4.0, 4.0]]], dtype=np.float32
    )
    support = np.asarray(
        [[[True, False, False], [True, False, False]]], dtype=bool
    )

    class FakePropagator:
        def refine(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _fake_result(
                    kwargs, residual=residual, support=support
                )
            return _fake_result(kwargs)

    engine.hart_propagator = FakePropagator()

    _run_worker(engine, [_window(3.0), _window(3.0)])

    assert len(calls) == 2
    np.testing.assert_array_equal(
        captured[0][0][..., -1],
        [[[6.0, 12.0, 12.0], [6.0, 12.0, 12.0]]],
    )
    np.testing.assert_array_equal(
        captured[0][1],
        [[[True, False, False], [True, False, False]]],
    )
    np.testing.assert_array_equal(
        calls[1][
            "previous_registration_state"
        ].final_base_points_tail[..., -1],
        3.0,
    )


def test_ordinary_hart_changes_current_window_camera_scale(monkeypatch, tmp_path):
    _patch_registration_primitives(monkeypatch)
    final_scales = []

    def solve(previous_poses, current_poses, scale):
        final_scales.append(float(scale))
        return torch.eye(3), torch.zeros(3)

    monkeypatch.setattr(swe_module, "register_adjacent_window_pose", solve)
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    engine._build_hart_segments = lambda *args: SimpleNamespace()
    calls = []

    class FakePropagator:
        def refine(self, **kwargs):
            calls.append(kwargs)
            return _fake_result(
                kwargs,
                window_scale=2.0 if len(calls) == 2 else 1.0,
            )

    engine.hart_propagator = FakePropagator()
    second = _window(3.0)
    second["camera_poses"][:, 0, 3] = torch.tensor([1.0, 2.0])

    _run_worker(engine, [_window(3.0), second])

    assert final_scales == [2.0]
    np.testing.assert_array_equal(
        engine.prev_window_cache["camera_poses"][:, 0, 3],
        [2.0, 4.0],
    )
    assert (
        engine.prev_window_cache["hart_diagnostics"][
            "final_registration_scale"
        ]
        == 2.0
    )


def test_ordinary_hart_applies_public_scale_and_local_residual_once(
    monkeypatch, tmp_path
):
    _patch_registration_primitives(monkeypatch)
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    engine._build_hart_segments = lambda *args: SimpleNamespace()
    calls = []
    residual = np.asarray(
        [[[0.5, 1.0, 1.0], [0.5, 1.0, 1.0]]], dtype=np.float32
    )

    class FakePropagator:
        def refine(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                return _fake_result(
                    kwargs, window_scale=2.0, residual=residual
                )
            return _fake_result(kwargs)

    engine.hart_propagator = FakePropagator()

    _run_worker(engine, [_window(3.0), _window(3.0)])

    np.testing.assert_array_equal(
        engine.prev_window_cache["local_points"][..., 0, -1], 3.0
    )
    np.testing.assert_array_equal(
        engine.prev_window_cache["local_points"][..., 1:, -1], 6.0
    )
    np.testing.assert_array_equal(
        engine.registration_state.final_base_points_tail[..., -1], 6.0
    )


def test_real_hart_uniform_scale_changes_current_window(monkeypatch, tmp_path):
    _patch_registration_primitives(monkeypatch)
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    labels = np.zeros((2, 3), dtype=np.intp)
    frame = _frame_from_layers(labels, labels)

    def segments(base_points, *args):
        return SegmentationWindow(
            frames=tuple(frame for _ in range(base_points.shape[0])),
            segment_mode="depth",
        )

    engine._build_hart_segments = segments
    second = _window(3.0)
    second["camera_poses"][:, 0, 3] = torch.tensor([1.0, 2.0])

    _run_worker(engine, [_window(6.0), second])

    assert engine.prev_window_cache["hart_diagnostics"]["window_scale"] == 2.0
    np.testing.assert_array_equal(
        engine.prev_window_cache["camera_poses"][:, 0, 3], [2.0, 4.0]
    )
    np.testing.assert_array_equal(
        engine.prev_window_cache["local_points"][..., -1], 6.0
    )


def _cache(depth, sim3_scale, *, local_residual=None, legacy_mask=None):
    cache = {
        "local_points": torch.full((1, 1, 1, 3), depth),
        "camera_poses": torch.eye(4)[None],
        "sim3": (sim3_scale, torch.eye(3), torch.zeros(3)),
    }
    if local_residual is not None:
        cache["local_residual_mask"] = torch.full(
            (1, 1, 1, 1), local_residual
        )
    if legacy_mask is not None:
        cache["scale_mask"] = torch.full((1, 1, 1, 1), legacy_mask)
    return cache


def test_lc_hart_applies_optimized_global_scale_and_local_residual_once():
    aggregated = StreamingWindowEngineLC.aggregate_caches(
        [
            _cache(3.0, 1.0, local_residual=2.0),
            _cache(3.0, 2.0, local_residual=4.0),
        ]
    )

    points = aggregated["local_points"].squeeze(0)
    np.testing.assert_array_equal(points[0], 6.0)
    np.testing.assert_array_equal(points[1], 24.0)


def _patch_lc_registration_primitives(monkeypatch, captured_sources=None):
    monkeypatch.setattr(
        swe_lc_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda points: (None, torch.eye(3)[None]),
    )

    def unproject(depth, intrinsic):
        points = torch.zeros((*depth.shape, 3), dtype=depth.dtype)
        points[..., -1] = depth
        return points

    monkeypatch.setattr(
        swe_lc_module, "unproject_depth_to_local_points", unproject
    )

    def register(src_points, *args):
        if captured_sources is not None:
            captured_sources.append(src_points.detach().clone())
        return 2.0, torch.eye(3), torch.zeros(3)

    monkeypatch.setattr(swe_lc_module, "register_adjacent_windows", register)
    monkeypatch.setattr(
        swe_lc_module,
        "register_adjacent_window_pose",
        lambda *args: (torch.eye(3), torch.zeros(3)),
    )


def test_lc_worker_couples_window_scale_into_current_pairwise_sim3(
    monkeypatch, tmp_path
):
    _patch_lc_registration_primitives(monkeypatch)
    engine = StreamingWindowEngineLC(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        window_size=2,
        overlap=1,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    engine._save_cache = lambda: None
    engine._build_hart_segments = lambda *args: SimpleNamespace()
    calls = []

    class FakePropagator:
        def refine(self, **kwargs):
            calls.append(kwargs)
            return _fake_result(
                kwargs,
                window_scale=1.5 if len(calls) == 2 else 1.0,
                residual=2.0 if len(calls) == 2 else 1.0,
                support=len(calls) == 2,
            )

    engine.hart_propagator = FakePropagator()
    _run_worker(engine, [_window(3.0), _window(3.0)])

    np.testing.assert_array_equal(
        engine.prev_window_cache["local_points"][..., -1], 3.0
    )
    np.testing.assert_array_equal(calls[1]["current_base_points"][..., -1], 6.0)
    assert engine.prev_window_cache["sim3"][0] == pytest.approx(3.0)
    assert engine.registration_state.cumulative_sim3[0] == pytest.approx(3.0)
    np.testing.assert_array_equal(
        engine.registration_state.final_base_points_tail[..., -1], 9.0
    )
    np.testing.assert_array_equal(
        engine.prev_window_cache["local_residual_mask"], 2.0
    )
    assert (
        engine.prev_window_cache["hart_diagnostics"][
            "final_registration_scale"
        ]
        == pytest.approx(3.0)
    )


def test_lc_registration_uses_previous_raw_residual_on_support(
    monkeypatch, tmp_path
):
    captured_sources = []
    _patch_lc_registration_primitives(monkeypatch, captured_sources)
    engine = StreamingWindowEngineLC(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        window_size=2,
        overlap=1,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    engine._save_cache = lambda: None
    engine._build_hart_segments = lambda *args: SimpleNamespace()
    calls = []

    class FakePropagator:
        def refine(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                return _fake_result(
                    kwargs,
                    window_scale=1.5,
                    residual=2.0,
                    support=True,
                )
            return _fake_result(kwargs)

    engine.hart_propagator = FakePropagator()
    _run_worker(engine, [_window(3.0), _window(3.0), _window(3.0)])

    np.testing.assert_array_equal(captured_sources[0][..., -1], 3.0)
    np.testing.assert_array_equal(captured_sources[1][..., -1], 6.0)
    assert not torch.any(captured_sources[1][..., -1] == 9.0)


def test_lc_legacy_scale_mask_semantics_are_unchanged():
    aggregated = StreamingWindowEngineLC.aggregate_caches(
        [
            _cache(3.0, 1.0),
            _cache(3.0, 2.0, legacy_mask=4.0),
        ]
    )

    points = aggregated["local_points"].squeeze(0)
    np.testing.assert_array_equal(points[0], 3.0)
    np.testing.assert_array_equal(points[1], 12.0)
