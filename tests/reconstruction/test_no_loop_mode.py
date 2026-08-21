from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_engine.segmentation import DisabledWindowReferenceRefiner
from pipeline.artifacts import ReconstructionArtifact, write_reconstruction_artifact
from pipeline.config import (
    ConfidenceQuantileMethod,
    ReconstructionMode,
    load_pipeline_config,
)
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.no_loop import NoLoopReconstructionMode
from reconstruction.prediction_stream import iter_window_predictions
from reconstruction.shared import (
    as_numpy,
    reference_intrinsic,
    segment_and_refine_window,
)
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


def _segment_only_window(
    *,
    strategy,
    refiner,
    point_maps,
    camera_poses,
    confidence,
    images,
    reference_intrinsic,
):
    del refiner, camera_poses, reference_intrinsic
    return strategy.segment(
        as_numpy(point_maps),
        as_numpy(confidence),
        as_numpy(images),
    )


def _artifact_files(path):
    return sorted(
        item.relative_to(path)
        for item in Path(path).rglob("*")
        if item.is_file()
    )


def _write_test_artifact(artifact, path):
    return write_reconstruction_artifact(
        artifact,
        path,
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
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


def test_no_loop_disabled_wrapper_matches_segment_only_artifact_contract(tmp_path):
    def run(segment_window):
        graph_inputs = []

        def build_graphs(results, threshold):
            graph_inputs.append(results)
            return tuple(results), threshold

        mode = NoLoopReconstructionMode(
            register_adjacent=lambda *arguments: identity_sim3(),
            apply_pose_sim3=lambda poses, *arguments: poses,
            build_graphs=build_graphs,
            segment_window=segment_window,
        )
        artifact = mode.run(
            _context(
                SequencedAnchor(scales=(1.0, 1.0)),
                iter_window_predictions(
                    LiteralProvider(),
                    SPECS,
                    torch.zeros((4, 3, 1, 1)),
                    "cpu",
                ),
            )
        )
        return artifact, graph_inputs

    baseline, baseline_graphs = run(_segment_only_window)
    wrapped, wrapped_graphs = run(segment_and_refine_window)

    assert baseline.schema_version == wrapped.schema_version == 1
    assert baseline.reconstruction_mode is wrapped.reconstruction_mode
    assert baseline.segmentation_method is wrapped.segmentation_method
    assert baseline.prediction_key == wrapped.prediction_key
    assert baseline.diagnostics == wrapped.diagnostics
    for name in ("local_points", "global_points", "camera_poses", "confidence"):
        assert torch.equal(getattr(baseline, name), getattr(wrapped, name))
    assert len(baseline_graphs) == len(wrapped_graphs)
    for baseline_results, wrapped_results in zip(
        baseline_graphs,
        wrapped_graphs,
    ):
        assert len(baseline_results) == len(wrapped_results)
        for baseline_result, wrapped_result in zip(
            baseline_results,
            wrapped_results,
        ):
            np.testing.assert_array_equal(
                baseline_result.labels,
                wrapped_result.labels,
            )
            assert dict(baseline_result.diagnostics) == dict(
                wrapped_result.diagnostics
            )

    baseline_dir = _write_test_artifact(baseline, tmp_path / "baseline")
    wrapped_dir = _write_test_artifact(wrapped, tmp_path / "wrapped")
    assert _artifact_files(baseline_dir) == _artifact_files(wrapped_dir)
    for relative in (
        Path("manifest.json"),
        Path("diagnostics.json"),
        Path("resolved_reconstruction.yaml"),
    ):
        baseline_file = baseline_dir / relative
        wrapped_file = wrapped_dir / relative
        if relative.suffix == ".json":
            assert json.loads(baseline_file.read_text()) == json.loads(
                wrapped_file.read_text()
            )
        else:
            assert baseline_file.read_text() == wrapped_file.read_text()


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
