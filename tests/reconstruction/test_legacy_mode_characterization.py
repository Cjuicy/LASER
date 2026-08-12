from __future__ import annotations

import numpy as np
import pytest
import torch

from inference_engine.prediction_cache.types import WindowSpec
from inference_engine.streaming_window_engine import STOP_SIGNAL
from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopSolution,
    WindowCache,
)
from loop_closure.methods.corrected import (
    CorrectedLoopClosureStrategy,
    CorrectedWindowEngine,
)
from loop_closure.methods.traditional import (
    TraditionalLoopClosureStrategy,
    TraditionalWindowEngine,
)
from mv_recon.paper_streaming import (
    PaperStreamingDependencies,
    reconstruct_incremental_point_maps,
)
from pipeline.config import (
    LoopMethod,
    ModelName,
    load_pipeline_config,
)

from .fixtures import (
    CHECKPOINT_DIGEST,
    PREDICTION_KEY,
    SPECS,
    LiteralProvider,
    OneRegionSegmenter,
    SequencedAnchor,
    identity_sim3,
    literal_window,
)


def _optimizer_config():
    return load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer


def _run_legacy_engine(engine, count: int):
    caches = []

    def save_cache():
        caches.append(engine.prev_window_cache)
        engine.cache_id += 1

    engine._save_cache = save_cache
    for index in range(count):
        engine.registration_queue.put(
            (
                WindowSpec(index=index, frame_start=index, frame_end=index + 2),
                literal_window(),
                0.0,
            )
        )
    engine.registration_queue.put(STOP_SIGNAL)
    engine._registration_worker()
    return tuple(caches)


def _build_engine(engine_type, tmp_path, anchor_scale: float):
    anchor = SequencedAnchor(scales=(anchor_scale, anchor_scale))
    engine = engine_type(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=OneRegionSegmenter(),
        anchor_propagator=anchor,
        registration_confidence_keep_ratio=0.5,
        anchor_enabled=True,
        temporal_iou_threshold=0.3,
        window_size=2,
        overlap=1,
        cache_root=str(tmp_path),
        intermediate_device="cpu",
        process_device="cpu",
        benchmark_latency=False,
        prediction_key=PREDICTION_KEY,
        model_name=ModelName.PI3,
        checkpoint_digest=CHECKPOINT_DIGEST,
    )
    return engine


def _legacy_cache(
    *,
    mode: LoopMethod,
    window_index: int,
    depth: float,
    anchor_scale: float | None,
    state: dict[str, object],
) -> WindowCache:
    return WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=mode,
        prediction_key=PREDICTION_KEY,
        model_name=ModelName.PI3,
        checkpoint_digest=CHECKPOINT_DIGEST,
        window_index=window_index,
        frame_start=window_index,
        frame_end=window_index + 2,
        local_points=torch.full((2, 1, 1, 3), depth),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        segmentation_labels=(
            np.zeros((1, 1), dtype=np.intp),
            np.zeros((1, 1), dtype=np.intp),
        ),
        anchor_scale_mask=(
            None
            if anchor_scale is None
            else torch.full((2, 1, 1, 1), anchor_scale)
        ),
        loop_state=state,
    )


def test_legacy_no_loop_uses_corrected_predecessor_for_next_registration():
    registration_sources = []
    registration_scales = iter((2.0, 3.0))

    def register(source_points, target_points, source_poses, target_poses, mask):
        del source_poses, target_poses, mask
        registration_sources.append(
            (
                float(source_points[-1, 0, 0, 2]),
                float(target_points[0, 0, 0, 2]),
            )
        )
        return identity_sim3(next(registration_scales))

    anchor = SequencedAnchor()
    result = reconstruct_incremental_point_maps(
        provider=LiteralProvider(),
        specs=SPECS,
        images=torch.zeros((4, 3, 1, 1)),
        segmenter=OneRegionSegmenter(),
        anchor_propagator=anchor,
        overlap=1,
        confidence_keep_ratio=0.5,
        temporal_iou_threshold=0.3,
        anchor_enabled=True,
        dependencies=PaperStreamingDependencies(
            register_adjacent_windows=register,
            apply_sim3_to_pose=(
                lambda poses, scale, rotation, translation: poses
            ),
            build_temporal_graphs=(
                lambda results, threshold: (tuple(results), threshold)
            ),
        ),
    )

    assert registration_sources == [(1.0, 1.0), (10.0, 1.0)]
    assert anchor.calls == [(1.0, 2.0), (10.0, 3.0)]
    assert result.local_points[:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        10.0,
        21.0,
    ]


def test_legacy_traditional_records_scale_before_final_application(
    monkeypatch,
    tmp_path,
):
    from loop_closure.methods import traditional as traditional_module

    engine = _build_engine(TraditionalWindowEngine, tmp_path, 3.0)
    monkeypatch.setattr(
        traditional_module,
        "register_adjacent_windows",
        lambda *arguments: identity_sim3(2.0),
    )
    caches = _run_legacy_engine(engine, count=2)
    solution = LoopSolution(
        optimized_transforms=(identity_sim3(), identity_sim3(2.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    result = TraditionalLoopClosureStrategy(
        optimizer_config=_optimizer_config(),
        registration_confidence_keep_ratio=0.5,
    ).aggregate(caches, solution)

    assert caches[1].local_points[:, 0, 0, 2].tolist() == [1.0, 1.0]
    assert caches[1].anchor_scale_mask[:, 0, 0, 0].tolist() == [3.0, 3.0]
    assert result.payload["local_points"][:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        3.0,
    ]


def test_legacy_corrected_applies_online_scale_and_final_delta_once(
    monkeypatch,
    tmp_path,
):
    from loop_closure.methods import corrected as corrected_module

    engine = _build_engine(CorrectedWindowEngine, tmp_path, 3.0)
    monkeypatch.setattr(
        corrected_module,
        "register_adjacent_windows",
        lambda *arguments: identity_sim3(2.0),
    )
    caches = _run_legacy_engine(engine, count=2)
    solution = LoopSolution(
        optimized_transforms=(identity_sim3(), identity_sim3(4.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    result = CorrectedLoopClosureStrategy(
        optimizer_config=_optimizer_config(),
        registration_confidence_keep_ratio=0.5,
    ).aggregate(caches, solution)

    assert caches[1].local_points[:, 0, 0, 2].tolist() == [6.0, 6.0]
    assert result.payload["local_points"][:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        pytest.approx(12.0),
    ]
