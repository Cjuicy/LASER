from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from inference_engine.segmentation import DisabledWindowReferenceRefiner
from reconstruction.shared import as_numpy
from loop_closure.types import LoopSolution
from pipeline.config import ReconstructionMode, load_pipeline_config
from pipeline.manifest import ImageManifest
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.traditional import (
    TraditionalLoopProcessor,
    TraditionalReconstructionMode,
)
from reconstruction.prediction_stream import iter_window_predictions
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


def _context(anchor, predictions, *, events=None):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml",
        (
            "reconstruction.mode=traditional",
            "window.size=2",
            "window.overlap=1",
            "model.process_device=cpu",
            "loop.optimizer.implementation=python",
        ),
    ).config
    images = torch.zeros((4, 3, 1, 1))
    return ReconstructionContext(
        predictions=predictions,
        frame_ids=(0, 1, 2, 3),
        segmentation_strategy=OneRegionSegmenter(),
        window_reference_refiner=DisabledWindowReferenceRefiner(),
        anchor_propagator=anchor,
        segmentation_config=config.segmentation,
        anchor_config=config.anchor_propagation,
        registration_config=config.registration,
        window_config=config.window,
        reconstruction_mode=ReconstructionMode.TRADITIONAL,
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


def test_traditional_defers_all_recorded_corrections_until_aggregate():
    registration_sources = []

    def register(source_points, target_points, *unused):
        registration_sources.append(
            (
                float(source_points[-1, 0, 0, 2]),
                float(target_points[0, 0, 0, 2]),
            )
        )
        return identity_sim3(2.0)

    anchor = SequencedAnchor(scales=(3.0, 3.0))
    context, optimizer_config = _context(anchor, _predictions())
    mode = TraditionalReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        optimizer=FailingOptimizer(),
        register_adjacent=register,
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
    )

    artifact = mode.run(context)

    assert registration_sources == [(1.0, 1.0), (1.0, 1.0)]
    assert mode.trace[1].local_points[:, 0, 0, 2].tolist() == [1.0, 1.0]
    assert mode.trace[1].anchor_scale_mask[:, 0, 0, 0].tolist() == [3.0, 3.0]
    assert artifact.local_points[:, 0, 0, 2].tolist() == [1.0, 1.0, 3.0, 6.0]


def test_traditional_segments_refines_graphs_and_anchors_untransformed_state():
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

    def build_graphs(results, threshold):
        events.append("graph")
        return tuple(results), threshold

    source = (replace(prediction, reference_intrinsic=intrinsic) for prediction in _predictions())
    context, optimizer_config = _context(RecordingAnchor(), source)
    mode = TraditionalReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        optimizer=FailingOptimizer(),
        register_adjacent=lambda *arguments: identity_sim3(),
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=build_graphs,
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
    assert all(torch.equal(call[0][..., 2], torch.ones((2, 1, 1))) for call in calls)
    assert all(torch.equal(call[1], torch.eye(4).repeat(2, 1, 1)) for call in calls)
    assert all(call[3] is intrinsic for call in calls)
    assert all(call[4] is context.window_reference_refiner for call in calls)


def test_traditional_detects_only_after_all_windows_are_recorded():
    events = []
    context, optimizer_config = _context(
        SequencedAnchor(scales=(1.0, 1.0)),
        _predictions(events),
    )

    class RecordingProcessor(TraditionalLoopProcessor):
        def optimize(self, states, constraints):
            events.append("optimize")
            return super().optimize(states, constraints)

        def aggregate(self, states, solution):
            events.append("aggregate")
            return super().aggregate(states, solution)

    processor = RecordingProcessor(
        optimizer_config,
        optimizer=FailingOptimizer(),
    )
    mode = TraditionalReconstructionMode(
        detector=EmptyDetector(events),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        processor=processor,
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
        "aggregate",
    ]


def test_traditional_processor_accepts_typed_relative_state():
    context, optimizer_config = _context(
        SequencedAnchor(scales=(1.0, 1.0)),
        _predictions(),
    )
    mode = TraditionalReconstructionMode(
        detector=EmptyDetector(),
        evidence=UnusedEvidence(),
        optimizer_config=optimizer_config,
        optimizer=FailingOptimizer(),
        register_adjacent=lambda *arguments: identity_sim3(),
        apply_pose_sim3=lambda poses, *arguments: poses,
        build_graphs=lambda results, threshold: (tuple(results), threshold),
    )

    mode.run(context)
    solution = LoopSolution(
        optimized_transforms=tuple(state.relative_sim3 for state in mode.trace),
        constraints=(),
        used_no_loop_path=True,
    )

    assert solution.optimized_transforms == tuple(
        state.relative_sim3 for state in mode.trace
    )
