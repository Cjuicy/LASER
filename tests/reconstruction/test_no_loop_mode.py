from __future__ import annotations

import inspect

import pytest
import torch

from pipeline.artifacts import ReconstructionArtifact
from pipeline.config import (
    ConfidenceQuantileMethod,
    ReconstructionMode,
    load_pipeline_config,
)
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.no_loop import NoLoopReconstructionMode
from reconstruction.prediction_stream import iter_window_predictions
from tests.reconstruction.fixtures import (
    SPECS,
    LiteralProvider,
    OneRegionSegmenter,
    SequencedAnchor,
    identity_sim3,
)


def _context(anchor, predictions):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser_no_loop.yaml",
        (
            "window.size=2",
            "window.overlap=1",
            "model.process_device=cpu",
        ),
    ).config
    return ReconstructionContext(
        predictions=predictions,
        frame_ids=(0, 1, 2, 3),
        segmentation_strategy=OneRegionSegmenter(),
        anchor_propagator=anchor,
        segmentation_config=config.segmentation,
        anchor_config=config.anchor_propagation,
        registration_config=config.registration,
        window_config=config.window,
        reconstruction_mode=ReconstructionMode.NO_LOOP,
    )


def test_no_loop_dependencies_contain_no_loop_services():
    parameters = inspect.signature(NoLoopReconstructionMode).parameters
    assert "loop_detector" not in parameters
    assert "constraint_estimator" not in parameters
    assert "optimizer" not in parameters


def test_no_loop_matches_table_four_incremental_order():
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
    predictions = iter_window_predictions(
        LiteralProvider(),
        SPECS,
        torch.zeros((4, 3, 1, 1)),
        "cpu",
    )
    mode = NoLoopReconstructionMode(
        register_adjacent=register,
        apply_pose_sim3=lambda poses, scale, rotation, translation: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
    )

    artifact = mode.run(_context(anchor, predictions))

    assert isinstance(artifact, ReconstructionArtifact)
    assert artifact.reconstruction_mode is ReconstructionMode.NO_LOOP
    assert registration_sources == [(1.0, 1.0), (10.0, 1.0)]
    assert anchor.calls == [(1.0, 2.0), (10.0, 3.0)]
    assert artifact.local_points[:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        10.0,
        21.0,
    ]
    assert artifact.global_points[:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        10.0,
        21.0,
    ]


def test_no_loop_rejects_context_for_other_mode():
    context = _context(
        SequencedAnchor(),
        iter_window_predictions(
            LiteralProvider(),
            SPECS,
            torch.zeros((4, 3, 1, 1)),
            "cpu",
        ),
    )
    context = ReconstructionContext(
        **{
            **context.__dict__,
            "reconstruction_mode": ReconstructionMode.CORRECTED,
        }
    )
    with pytest.raises(ValueError, match="no_loop context"):
        NoLoopReconstructionMode().run(context)


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        (ConfidenceQuantileMethod.HIGHER, [False, False, False, True, True, True]),
        (ConfidenceQuantileMethod.NEAREST, [False, False, True, True, True, True]),
    ),
)
def test_depth_confidence_quantile_method_is_explicit(method, expected):
    from inference_engine.segmentation.confidence import (
        select_numpy_top_confidence_mask,
    )

    mask = select_numpy_top_confidence_mask(
        [0, 1, 2, 3, 4, 5],
        0.5,
        method=method,
    )
    assert mask.tolist() == expected
