from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from inference_engine.segmentation import DisabledWindowReferenceRefiner
from loop_closure.methods.corrected import CorrectedLoopProcessor
from loop_closure.types import LoopSolution
from pipeline.artifacts import write_reconstruction_artifact
from pipeline.config import ReconstructionMode, load_pipeline_config
from pipeline.manifest import ImageManifest
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.corrected import CorrectedReconstructionMode
from reconstruction.prediction_stream import iter_window_predictions
from reconstruction.residual_alignment import ResidualAlignmentResult
from reconstruction.shared import as_numpy, segment_and_refine_window
from tests.reconstruction.fixtures import (
    LiteralProvider,
    OneRegionSegmenter,
    SPECS,
    SequencedAnchor,
    identity_sim3,
)


class EmptyDetector:
    def __init__(self, events=None):
        self.events = events

    def detect(self, manifest, images):
        if self.events is not None:
            self.events.append("detect")
        assert len(manifest) == images.shape[0]
        return ()


class UnusedEvidence:
    def estimate(self, *arguments):
        raise AssertionError("empty candidates must not request evidence")


class FailingOptimizer:
    def optimize(self, *arguments):
        raise AssertionError("empty constraints must not invoke optimizer")


def _context(anchor, predictions, *, refiner=None, anchor_enabled=True):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml",
        (
            "reconstruction.mode=corrected",
            "window.size=2",
            "window.overlap=1",
            "model.process_device=cpu",
            "loop.optimizer.implementation=python",
            "anchor_propagation.enabled="
            f"{str(anchor_enabled).lower()}",
        ),
    ).config
    images = torch.zeros((4, 3, 1, 1))
    return ReconstructionContext(
        predictions=predictions,
        frame_ids=(0, 1, 2, 3),
        segmentation_strategy=OneRegionSegmenter(),
        window_reference_refiner=(
            refiner if refiner is not None else DisabledWindowReferenceRefiner()
        ),
        anchor_propagator=anchor,
        segmentation_config=config.segmentation,
        anchor_config=config.anchor_propagation,
        registration_config=config.registration,
        window_config=config.window,
        reconstruction_mode=ReconstructionMode.CORRECTED,
        image_manifest=ImageManifest(
            paths=tuple(Path(f"frame-{index}.png") for index in range(4))
        ),
        images=images,
    ), config.loop.optimizer


def _predictions(events=None):
    source = iter_window_predictions(
        LiteralProvider(),
        SPECS,
        torch.zeros((4, 3, 1, 1)),
        "cpu",
    )
    for prediction in source:
        if events is not None:
            events.append(f"window:{prediction.spec.index}")
        yield prediction


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


def _passthrough_residual(**values):
    return ResidualAlignmentResult(
        local_points=values["current_points"],
        camera_poses=values["current_poses"],
        sim3=identity_sim3(),
        correspondence_count=1,
        abs_log_scale=0.0,
        rotation_rad=0.0,
        translation_norm=0.0,
    )


def test_corrected_uses_corrected_window_as_next_registration_source():
    registration_sources = []
    registration_scales = iter((2.0, 3.0))

    def register(source_points, target_points, *unused):
        registration_sources.append(
            (
                float(source_points[-1, 0, 0, 2]),
                float(target_points[0, 0, 0, 2]),
            )
        )
        return identity_sim3(next(registration_scales))

    anchor = SequencedAnchor(scales=(5.0, 7.0))
    context, optimizer_config = _context(anchor, _predictions())
    mode = CorrectedReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        optimizer=FailingOptimizer(),
        register_adjacent=register,
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
        residual_align=_passthrough_residual,
    )

    artifact = mode.run(context)

    assert registration_sources == [(1.0, 1.0), (10.0, 1.0)]
    assert anchor.calls == [(1.0, 2.0), (10.0, 3.0)]
    assert artifact.local_points[:, 0, 0, 2].tolist() == [
        1.0,
        1.0,
        10.0,
        21.0,
    ]


def test_corrected_composes_residual_after_coarse_and_rebuilds_edge():
    coarse_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    coarse_translation = torch.tensor([1.0, 0.0, 0.0])
    coarse_sim3 = (2.0, coarse_rotation, coarse_translation)
    residual_rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    residual_translation = torch.tensor([0.0, 2.0, 0.0])
    residual_sim3 = (4.0, residual_rotation, residual_translation)
    expected_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    expected_translation = torch.tensor([4.0, 2.0, 0.0])
    context, optimizer_config = _context(
        SequencedAnchor(scales=(3.0,)),
        iter_window_predictions(
            LiteralProvider(),
            SPECS[:2],
            torch.zeros((3, 3, 1, 1)),
            "cpu",
        ),
    )
    context = ReconstructionContext(
        **{
            **context.__dict__,
            "frame_ids": (0, 1, 2),
            "image_manifest": ImageManifest(
                paths=tuple(Path(f"frame-{index}.png") for index in range(3))
            ),
            "images": torch.zeros((3, 3, 1, 1)),
        }
    )

    def residual_align(**values):
        assert values["current_points"][0, 0, 0, 2].item() == 6.0
        poses = values["current_poses"].clone()
        poses[:, 0, 3] += 9.0
        return ResidualAlignmentResult(
            local_points=4.0 * values["current_points"],
            camera_poses=poses,
            sim3=residual_sim3,
            correspondence_count=1,
            abs_log_scale=math.log(4.0),
            rotation_rad=0.0,
            translation_norm=9.0,
        )

    class InspectingProcessor(CorrectedLoopProcessor):
        def optimize(self, states, constraints):
            del constraints
            for transform in (states[1].sim3_abs, states[1].sim3_edge):
                assert torch.as_tensor(transform[0]).item() == 8.0
                torch.testing.assert_close(transform[1], expected_rotation)
                torch.testing.assert_close(transform[2], expected_translation)
            return LoopSolution(
                optimized_transforms=tuple(state.sim3_abs for state in states),
                constraints=(),
                used_no_loop_path=True,
            )

    mode = CorrectedReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        processor=InspectingProcessor(
            optimizer_config,
            optimizer=FailingOptimizer(),
            apply_pose_sim3=lambda poses, *arguments: poses,
        ),
        register_adjacent=lambda *arguments: coarse_sim3,
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
        residual_align=residual_align,
    )

    artifact = mode.run(context)

    assert mode.trace[1].local_points[-1, 0, 0, 2].item() == 24.0
    assert artifact.local_points[-1, 0, 0, 2].item() == 24.0
    assert artifact.camera_poses[-1, 0, 3].item() == 9.0
    assert artifact.diagnostics.mode_scalars["residual_applied_window_count"] == 1
    assert artifact.diagnostics.mode_scalars["residual_skipped_window_count"] == 1
    assert artifact.diagnostics.mode_scalars["refined_edge_count"] == 1


def test_corrected_disabled_wrapper_matches_segment_only_artifact_contract(
    tmp_path,
):
    def run(segment_window):
        graph_inputs = []

        def build_graphs(results, threshold):
            graph_inputs.append(results)
            return tuple(results), threshold

        context, optimizer_config = _context(
            SequencedAnchor(scales=(1.0, 1.0)),
            _predictions(),
        )
        mode = CorrectedReconstructionMode(
            detector=EmptyDetector(),
            evidence=UnusedEvidence(),
            optimizer_config=optimizer_config,
            optimizer=FailingOptimizer(),
            register_adjacent=lambda *arguments: identity_sim3(),
            apply_pose_sim3=lambda poses, *arguments: poses,
            build_graphs=build_graphs,
            segment_window=segment_window,
        )
        return mode.run(context), graph_inputs

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
            assert torch.equal(
                torch.as_tensor(baseline_result.labels),
                torch.as_tensor(wrapped_result.labels),
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


def test_corrected_segments_refines_graphs_and_anchors_adjusted_state():
    events = []
    calls = []
    intrinsic = torch.tensor(
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]
    )

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

    def build_graphs(results, threshold):
        events.append("graph")
        return tuple(results), threshold

    source = (replace(prediction, reference_intrinsic=intrinsic) for prediction in _predictions())
    context, optimizer_config = _context(RecordingAnchor(), source)
    mode = CorrectedReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        optimizer=FailingOptimizer(),
        register_adjacent=lambda *arguments: identity_sim3(
            (2.0, 3.0)[len(calls) - 1]
        ),
        apply_pose_sim3=apply_pose,
        build_graphs=build_graphs,
        segment_window=segment_window,
        residual_align=_passthrough_residual,
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
    assert all(call[3] is intrinsic for call in calls)
    assert all(call[4] is context.window_reference_refiner for call in calls)


def test_corrected_enabled_wrapper_forwards_missing_prediction_intrinsic():
    graph_inputs = []

    class RecordingEnabledRefiner:
        enabled = True

        def __init__(self):
            self.calls = []
            self.returned = []

        def refine(
            self,
            results,
            *,
            point_maps,
            camera_poses,
            confidence,
            reference_intrinsic,
        ):
            self.calls.append(
                (point_maps, camera_poses, confidence, reference_intrinsic)
            )
            refined = [
                type(result)(
                    result.labels.copy(),
                    {**dict(result.diagnostics), "refined": True},
                )
                for result in results
            ]
            self.returned.append(refined)
            return refined

    refiner = RecordingEnabledRefiner()

    def build_graphs(results, threshold):
        graph_inputs.append(results)
        return tuple(results), threshold

    context, optimizer_config = _context(
        SequencedAnchor(scales=(1.0, 1.0)),
        _predictions(),
        refiner=refiner,
    )
    mode = CorrectedReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        optimizer=FailingOptimizer(),
        register_adjacent=lambda *arguments: identity_sim3(),
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=build_graphs,
        segment_window=segment_and_refine_window,
    )

    artifact = mode.run(context)

    assert len(refiner.calls) == 3
    assert all(call[3] is None for call in refiner.calls)
    assert all(
        graph_result is refined
        for graph_result, refined in zip(graph_inputs, refiner.returned)
    )
    assert all(
        graph_result[0].labels is refined[0].labels
        for graph_result, refined in zip(graph_inputs, refiner.returned)
    )
    assert all(
        result.diagnostics["refined"]
        for window_results in graph_inputs
        for result in window_results
    )
    assert all(
        summary["refined"]
        for summary in artifact.diagnostics.segmentation_summaries
    )


def test_corrected_applies_optimized_original_delta_exactly_once():
    context, optimizer_config = _context(
        SequencedAnchor(scales=(3.0,)),
        iter_window_predictions(
            LiteralProvider(),
            SPECS[:2],
            torch.zeros((3, 3, 1, 1)),
            "cpu",
        ),
    )
    context = ReconstructionContext(
        **{
            **context.__dict__,
            "frame_ids": (0, 1, 2),
            "image_manifest": ImageManifest(
                paths=tuple(Path(f"frame-{index}.png") for index in range(3))
            ),
            "images": torch.zeros((3, 3, 1, 1)),
        }
    )

    class DeltaProcessor(CorrectedLoopProcessor):
        def optimize(self, states, constraints):
            assert states[1].sim3_abs[0] == 8.0
            return LoopSolution(
                optimized_transforms=(identity_sim3(), identity_sim3(16.0)),
                constraints=(),
                used_no_loop_path=False,
            )

    def residual_align(**values):
        return ResidualAlignmentResult(
            local_points=4.0 * values["current_points"],
            camera_poses=values["current_poses"],
            sim3=identity_sim3(4.0),
            correspondence_count=1,
            abs_log_scale=math.log(4.0),
            rotation_rad=0.0,
            translation_norm=0.0,
        )

    processor = DeltaProcessor(
        optimizer_config,
        optimizer=FailingOptimizer(),
        apply_pose_sim3=lambda poses, *arguments: poses,
    )
    mode = CorrectedReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        processor=processor,
        register_adjacent=lambda *arguments: identity_sim3(2.0),
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
        residual_align=residual_align,
    )

    artifact = mode.run(context)

    assert mode.trace[1].local_points[-1, 0, 0, 2].item() == 24.0
    assert artifact.local_points[-1, 0, 0, 2].item() == 48.0


def test_corrected_anchor_disabled_skips_residual_and_keeps_coarse_edges():
    def fail_if_called(**values):
        del values
        raise AssertionError("residual alignment must be disabled with anchors")

    context, optimizer_config = _context(
        SequencedAnchor(),
        _predictions(),
        anchor_enabled=False,
    )
    mode = CorrectedReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        optimizer=FailingOptimizer(),
        register_adjacent=lambda *arguments: identity_sim3(2.0),
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
        residual_align=fail_if_called,
    )

    artifact = mode.run(context)

    assert torch.as_tensor(mode.trace[1].sim3_abs[0]).item() == 2.0
    assert torch.as_tensor(mode.trace[1].sim3_edge[0]).item() == 2.0
    assert artifact.diagnostics.mode_scalars["residual_applied_window_count"] == 0
    assert artifact.diagnostics.mode_scalars["residual_skipped_window_count"] == 3


def test_corrected_detects_after_windows_and_applies_delta_last():
    events = []
    context, optimizer_config = _context(
        SequencedAnchor(scales=(1.0, 1.0)),
        _predictions(events),
    )

    class RecordingProcessor(CorrectedLoopProcessor):
        def optimize(self, states, constraints):
            events.append("optimize")
            return super().optimize(states, constraints)

        def aggregate(self, states, solution):
            events.append("apply_delta")
            return super().aggregate(states, solution)

    mode = CorrectedReconstructionMode(
        detector=EmptyDetector(events),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        processor=RecordingProcessor(
            optimizer_config,
            optimizer=FailingOptimizer(),
            apply_pose_sim3=lambda poses, *arguments: poses,
        ),
        register_adjacent=lambda *arguments: identity_sim3(),
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
    )

    mode.run(context)

    assert events == [
        "window:0",
        "window:1",
        "window:2",
        "detect",
        "optimize",
        "apply_delta",
    ]
