from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_engine import streaming_window_engine as swe_module
from inference_engine import streaming_window_engine_lc as swe_lc_module
from inference_engine.anchor_propagation.types import PropagationResult
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


def _run_worker(engine, windows):
    engine._save_cache = lambda: None
    for window in windows:
        engine.registration_queue.put((window, 0.0))
    engine.registration_queue.put(STOP_SIGNAL)
    engine._registration_worker()


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


def test_hart_registration_reads_base_points_not_refined_output(
    monkeypatch, tmp_path
):
    captured_sources = []
    _patch_registration_primitives(monkeypatch, captured_sources)
    engine = _engine(
        tmp_path,
        depth_refine=False,
        anchor_propagation="hart",
        anchor_min_pixels=1,
    )
    engine._build_segment_graph = lambda *args: pytest.fail("legacy graph called")
    engine._build_hart_segments = lambda *args: SimpleNamespace()
    calls = []

    class FakePropagator:
        def refine(self, **kwargs):
            calls.append(kwargs)
            scale = 2.0 if len(calls) == 1 else 1.0
            shape = (*kwargs["current_base_points"].shape[:3], 1)
            return PropagationResult(
                local_scale_mask=np.full(shape, scale, dtype=np.float32),
                next_state=object(),
                diagnostics={"fake": len(calls)},
            )

    engine.hart_propagator = FakePropagator()

    _run_worker(engine, [_window(3.0), _window(3.0)])

    assert len(calls) == 2
    np.testing.assert_array_equal(captured_sources[0][..., -1], 3.0)
    np.testing.assert_array_equal(
        calls[1]["previous_registration_state"].base_points_tail[..., -1],
        3.0,
    )


def _cache(depth, sim3_scale, *, local_mask=None, legacy_mask=None):
    cache = {
        "local_points": torch.full((1, 1, 1, 3), depth),
        "camera_poses": torch.eye(4)[None],
        "sim3": (sim3_scale, torch.eye(3), torch.zeros(3)),
    }
    if local_mask is not None:
        cache["local_scale_mask"] = torch.full((1, 1, 1, 1), local_mask)
    if legacy_mask is not None:
        cache["scale_mask"] = torch.full((1, 1, 1, 1), legacy_mask)
    return cache


def test_lc_hart_applies_optimized_global_scale_and_local_residual_once():
    aggregated = StreamingWindowEngineLC.aggregate_caches(
        [
            _cache(3.0, 1.0, local_mask=2.0),
            _cache(3.0, 2.0, local_mask=4.0),
        ]
    )

    points = aggregated["local_points"].squeeze(0)
    np.testing.assert_array_equal(points[0], 6.0)
    np.testing.assert_array_equal(points[1], 24.0)


def test_lc_worker_keeps_raw_points_and_tracks_cumulative_base_scale(
    monkeypatch, tmp_path
):
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
    monkeypatch.setattr(
        swe_lc_module,
        "register_adjacent_windows",
        lambda *args: (2.0, torch.eye(3), torch.zeros(3)),
    )
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
            shape = (*kwargs["current_base_points"].shape[:3], 1)
            return PropagationResult(
                local_scale_mask=np.ones(shape, dtype=np.float32),
                next_state=object(),
                diagnostics={},
            )

    engine.hart_propagator = FakePropagator()
    _run_worker(engine, [_window(3.0), _window(3.0)])

    np.testing.assert_array_equal(engine.prev_window_cache["local_points"][..., -1], 3.0)
    np.testing.assert_array_equal(calls[1]["current_base_points"][..., -1], 6.0)
    assert engine.registration_state.cumulative_sim3[0] == 2.0
    assert "local_scale_mask" in engine.prev_window_cache


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
