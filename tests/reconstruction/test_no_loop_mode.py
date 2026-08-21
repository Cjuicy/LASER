from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from inference_engine.segmentation import DisabledWindowReferenceRefiner
from pipeline.artifacts import ReconstructionArtifact
from pipeline.config import (
    ConfidenceQuantileMethod,
    ReconstructionMode,
    load_pipeline_config,
)
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.no_loop import NoLoopReconstructionMode
from reconstruction.prediction_stream import iter_window_predictions
from reconstruction.shared import as_numpy, reference_intrinsic
from tests.reconstruction.fixtures import (
    SPECS,
    LiteralProvider,
    OneRegionSegmenter,
    SequencedAnchor,
    identity_sim3,
)


def _context(anchor, predictions, *, segmenter=None):
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
        segmentation_strategy=segmenter or OneRegionSegmenter(),
        window_reference_refiner=DisabledWindowReferenceRefiner(),
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


def test_no_loop_consumes_predictions_incrementally():
    events = []

    class RecordingSegmenter(OneRegionSegmenter):
        def segment(self, points, confidence, images):
            events.append("segment")
            return super().segment(points, confidence, images)

    source = iter_window_predictions(
        LiteralProvider(),
        SPECS,
        torch.zeros((4, 3, 1, 1)),
        "cpu",
    )

    def guarded_stream():
        for index, prediction in enumerate(source):
            if index:
                assert events[-1] == "segment"
            events.append(f"yield:{index}")
            yield prediction

    NoLoopReconstructionMode(
        register_adjacent=lambda *arguments: identity_sim3(),
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
    ).run(
        _context(
            SequencedAnchor(scales=(1.0, 1.0)),
            guarded_stream(),
            segmenter=RecordingSegmenter(),
        )
    )

    assert events == [
        "yield:0",
        "segment",
        "yield:1",
        "segment",
        "yield:2",
        "segment",
    ]


def test_no_loop_segments_refines_graphs_and_anchors_current_window_state():
    events = []
    calls = []

    class RecordingAnchor:
        def propagate(
            self,
            source_points,
            target_points,
            source_graphs,
            target_graphs,
            overlap,
        ):
            del source_points, source_graphs, target_graphs, overlap
            events.append("anchor")
            return torch.ones((*target_points.shape[:-1], 1))

    def segment_window(
        *,
        strategy,
        refiner,
        point_maps,
        camera_poses,
        confidence,
        images,
        reference_intrinsic,
    ):
        events.extend(("segment", "refine"))
        calls.append(
            (
                point_maps.clone(),
                camera_poses.clone(),
                confidence.clone(),
                reference_intrinsic,
                refiner,
            )
        )
        return strategy.segment(
            as_numpy(point_maps),
            as_numpy(confidence),
            as_numpy(images),
        )

    def apply_pose(poses, scale, rotation, translation):
        del rotation, translation
        adjusted = poses.clone()
        adjusted[:, 0, 3] = float(scale)
        return adjusted

    predictions = iter_window_predictions(
        LiteralProvider(),
        SPECS,
        torch.zeros((4, 3, 1, 1)),
        "cpu",
    )
    context = _context(RecordingAnchor(), predictions)
    mode = NoLoopReconstructionMode(
        register_adjacent=lambda *arguments: identity_sim3(
            (2.0, 3.0)[len(calls) - 1]
        ),
        apply_pose_sim3=apply_pose,
        build_graphs=lambda results, threshold: (
            events.append("graph") or (tuple(results), threshold)
        ),
        segment_window=segment_window,
    )

    mode.run(context)

    assert events == [
        "segment",
        "refine",
        "graph",
        "segment",
        "refine",
        "graph",
        "anchor",
        "segment",
        "refine",
        "graph",
        "anchor",
    ]
    assert [call[0][..., 2].unique().item() for call in calls] == [1.0, 2.0, 3.0]
    assert [call[1][:, 0, 3].tolist() for call in calls] == [
        [0.0, 0.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]
    expected_intrinsic = reference_intrinsic(
        LiteralProvider().get(SPECS[0], torch.zeros((2, 3, 1, 1)))
        ["local_points"].squeeze(0)
    )
    assert all(
        torch.equal(call[3], expected_intrinsic)
        for call in calls
    )
    assert all(call[4] is context.window_reference_refiner for call in calls)


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
