from __future__ import annotations

from pathlib import Path

import torch

from loop_closure.methods.corrected import CorrectedLoopProcessor
from loop_closure.types import LoopSolution
from pipeline.config import ReconstructionMode, load_pipeline_config
from pipeline.manifest import ImageManifest
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.corrected import CorrectedReconstructionMode
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


def _context(anchor, predictions):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml",
        (
            "reconstruction.mode=corrected",
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
            assert states[1].sim3_abs[0] == 2.0
            return LoopSolution(
                optimized_transforms=(identity_sim3(), identity_sim3(4.0)),
                constraints=(),
                used_no_loop_path=False,
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
    )

    artifact = mode.run(context)

    assert mode.trace[1].local_points[-1, 0, 0, 2].item() == 6.0
    assert artifact.local_points[-1, 0, 0, 2].item() == 12.0


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
