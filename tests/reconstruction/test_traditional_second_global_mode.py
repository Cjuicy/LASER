from __future__ import annotations

import math
import numpy as np
from pathlib import Path
import pytest
import torch

from inference_engine.segmentation import DisabledWindowReferenceRefiner
from inference_engine.utils.geometry import (
    accumulate_sim3,
    apply_sim3_to_pose,
    closed_form_inverse_sim3,
    homogenize_points,
)
from loop_closure.methods.second_global import SecondGlobalLoopProcessor
from loop_closure.methods.traditional import TraditionalLoopProcessor
from loop_closure.types import LoopCandidate, LoopSolution
from pipeline.artifacts import StagedReconstructionArtifacts
from pipeline.config import ReconstructionMode, load_pipeline_config
from pipeline.manifest import ImageManifest
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.traditional import (
    TraditionalReconstructionMode,
    TraditionalWindowState,
)
from reconstruction.residual_alignment import ResidualAlignmentResult
from reconstruction.modes.traditional_second_global import (
    MaterializedTraditionalWindow,
    SecondGlobalWindowState,
    TraditionalSecondGlobalReconstructionMode,
    build_second_global_states,
    materialize_traditional_stage1_windows,
)
from reconstruction.prediction_stream import iter_window_predictions
from tests.reconstruction.fixtures import (
    LiteralProvider,
    OneRegionSegmenter,
    SPECS,
    SequencedAnchor,
)


def _identity_sim3(scale: float = 1.0):
    return scale, torch.eye(3), torch.zeros(3)


def _state(
    index: int,
    *,
    relative_scale: float = 1.0,
    anchor_scale: float | None = None,
) -> TraditionalWindowState:
    return TraditionalWindowState(
        window_index=index,
        frame_start=index,
        frame_end=index + 2,
        local_points=torch.ones((2, 1, 1, 3)),
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
        relative_sim3=_identity_sim3(relative_scale),
        segmentation_diagnostics=(),
    )


def _processor() -> TraditionalLoopProcessor:
    optimizer_config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    return TraditionalLoopProcessor(optimizer_config)


def test_stage1_materialization_exactly_reproduces_traditional_aggregate():
    states = (
        _state(0),
        _state(1, relative_scale=2.0, anchor_scale=3.0),
    )
    solution = LoopSolution(
        optimized_transforms=(_identity_sim3(), _identity_sim3(2.0)),
        constraints=(),
        used_no_loop_path=False,
    )
    expected = _processor().aggregate(states, solution)

    windows = materialize_traditional_stage1_windows(states, solution)
    local_points = torch.cat(
        (windows[0].local_points, windows[1].local_points[1:]),
        dim=0,
    )
    camera_poses = torch.cat(
        (windows[0].camera_poses, windows[1].camera_poses[1:]),
        dim=0,
    )
    confidence = torch.cat(
        (windows[0].confidence, windows[1].confidence[1:]),
        dim=0,
    )
    global_points = torch.einsum(
        "nij,nhwj->nhwi",
        camera_poses,
        homogenize_points(local_points),
    )[..., :3]

    assert torch.equal(local_points, expected.local_points)
    assert torch.equal(camera_poses, expected.camera_poses)
    assert torch.equal(confidence, expected.confidence)
    assert torch.equal(global_points, expected.global_points)
    assert torch.equal(states[1].local_points, torch.ones_like(states[1].local_points))


def test_stage1_materialization_rejects_state_solution_count_mismatch():
    states = (_state(0), _state(1))
    solution = LoopSolution(
        optimized_transforms=(_identity_sim3(),),
        constraints=(),
        used_no_loop_path=True,
    )

    with pytest.raises(ValueError, match="materialization.*count"):
        materialize_traditional_stage1_windows(states, solution)


def _residual_sim3(index: int):
    angle = 0.1 * index
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return 1.0 + index, rotation, torch.tensor([float(index), 0.0, 0.0])


def _assert_sim3_equal(actual, expected):
    assert torch.as_tensor(actual[0]).item() == pytest.approx(
        torch.as_tensor(expected[0]).item()
    )
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2])


def _materialized_window(index: int) -> MaterializedTraditionalWindow:
    return MaterializedTraditionalWindow(
        window_index=index,
        frame_start=index,
        frame_end=index + 2,
        local_points=torch.full((2, 1, 1, 3), float(index + 1)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1)),
        stage1_absolute_sim3=_identity_sim3(),
    )


def test_second_global_states_use_the_residual_refined_predecessor():
    calls = []

    def residual_align(**values):
        calls.append(values)
        index = len(calls)
        sim3 = _residual_sim3(index)
        return ResidualAlignmentResult(
            local_points=values["current_points"] * sim3[0],
            camera_poses=apply_sim3_to_pose(
                values["current_poses"],
                *sim3,
            ),
            sim3=sim3,
            correspondence_count=1,
            abs_log_scale=abs(math.log(float(sim3[0]))),
            rotation_rad=0.1 * index,
            translation_norm=float(index),
        )

    states, results = build_second_global_states(
        tuple(_materialized_window(index) for index in range(3)),
        overlap=1,
        confidence_keep_ratio=1.0,
        residual_align=residual_align,
    )

    assert calls[1]["previous_points"] is states[1].local_points
    assert calls[1]["previous_poses"] is states[1].camera_poses
    expected_edge = accumulate_sim3(
        closed_form_inverse_sim3(*states[1].sim3_abs),
        states[2].sim3_abs,
    )
    _assert_sim3_equal(states[2].sim3_edge, expected_edge)
    assert len(results) == len(states) - 1 == 2


def test_second_global_states_keep_one_window_at_identity():
    states, results = build_second_global_states(
        (_materialized_window(0),),
        overlap=1,
        confidence_keep_ratio=1.0,
        residual_align=lambda **values: (_ for _ in ()).throw(
            AssertionError(f"unexpected residual call: {values}")
        ),
    )

    assert states[0].sim3_edge is None
    _assert_sim3_equal(states[0].sim3_abs, _identity_sim3())
    assert results == ()


def _predictions():
    return iter_window_predictions(
        LiteralProvider(),
        SPECS,
        torch.zeros((4, 3, 1, 1)),
        "cpu",
    )


def _context(predictions, mode, anchor):
    config = load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml",
        (
            f"reconstruction.mode={mode.value}",
            "window.size=2",
            "window.overlap=1",
            "model.process_device=cpu",
            "loop.optimizer.implementation=python",
        ),
    ).config
    return (
        ReconstructionContext(
            predictions=predictions,
            frame_ids=(0, 1, 2, 3),
            segmentation_strategy=OneRegionSegmenter(),
            window_reference_refiner=DisabledWindowReferenceRefiner(),
            anchor_propagator=anchor,
            segmentation_config=config.segmentation,
            anchor_config=config.anchor_propagation,
            registration_config=config.registration,
            window_config=config.window,
            reconstruction_mode=mode,
            image_manifest=ImageManifest(
                paths=tuple(
                    Path(f"frame-{index}.png")
                    for index in range(4)
                )
            ),
            images=torch.zeros((4, 3, 1, 1)),
        ),
        config.loop.optimizer,
    )


class _OneCandidateDetector:
    def __init__(self):
        self.call_count = 0

    def detect(self, manifest, images):
        assert len(manifest) == images.shape[0] == 4
        self.call_count += 1
        return (LoopCandidate(frame_a=3, frame_b=0, similarity=0.8),)


class _RecordingEvidence:
    def __init__(self, *, reject_first: bool = False):
        self.reject_first = reject_first
        self.window_types = []

    def estimate(self, window_a, window_b, candidate):
        del candidate
        self.window_types.append((type(window_a), type(window_b)))
        if self.reject_first and len(self.window_types) == 1:
            raise ValueError("synthetic Stage 1 rejection")
        return _identity_sim3(), _identity_sim3()


class _PassThroughOptimizer:
    def optimize(self, edges, constraints):
        assert constraints
        return edges


def _identity_residual(recorder):
    def residual_align(**values):
        recorder.append(values)
        return ResidualAlignmentResult(
            local_points=values["current_points"],
            camera_poses=values["current_poses"],
            sim3=_identity_sim3(),
            correspondence_count=1,
            abs_log_scale=0.0,
            rotation_rad=0.0,
            translation_norm=0.0,
        )

    return residual_align


def _build_stage_processors(optimizer_config):
    return (
        TraditionalLoopProcessor(
            optimizer_config,
            optimizer=_PassThroughOptimizer(),
        ),
        SecondGlobalLoopProcessor(
            optimizer_config,
            optimizer=_PassThroughOptimizer(),
        ),
    )


def test_new_mode_preserves_stage1_and_rebuilds_from_corrected_geometry():
    baseline_context, optimizer_config = _context(
        _predictions(),
        ReconstructionMode.TRADITIONAL,
        SequencedAnchor(scales=(3.0, 3.0)),
    )
    baseline = TraditionalReconstructionMode(
        detector=_OneCandidateDetector(),
        evidence=_RecordingEvidence(),
        optimizer_config=optimizer_config,
        optimizer=_PassThroughOptimizer(),
        register_adjacent=lambda *arguments: _identity_sim3(),
        build_graphs=lambda results, threshold: (tuple(results), threshold),
    ).run(baseline_context)

    staged_context, optimizer_config = _context(
        _predictions(),
        ReconstructionMode.TRADITIONAL_SECOND_GLOBAL,
        SequencedAnchor(scales=(3.0, 3.0)),
    )
    detector = _OneCandidateDetector()
    evidence = _RecordingEvidence()
    residual_inputs = []
    stage1_processor, stage2_processor = _build_stage_processors(
        optimizer_config
    )
    staged = TraditionalSecondGlobalReconstructionMode(
        detector=detector,
        evidence=evidence,
        optimizer_config=optimizer_config,
        stage1_processor=stage1_processor,
        stage2_processor=stage2_processor,
        register_adjacent=lambda *arguments: _identity_sim3(),
        build_graphs=lambda results, threshold: (tuple(results), threshold),
        residual_align=_identity_residual(residual_inputs),
    ).run(staged_context)

    assert isinstance(staged, StagedReconstructionArtifacts)
    for name in ("local_points", "global_points", "camera_poses", "confidence"):
        assert torch.equal(
            getattr(staged.stage1, name),
            getattr(baseline, name),
        )
    assert staged.stage1.diagnostics == baseline.diagnostics
    assert detector.call_count == 1
    assert evidence.window_types == [
        (TraditionalWindowState, TraditionalWindowState),
        (SecondGlobalWindowState, SecondGlobalWindowState),
    ]
    assert residual_inputs[0]["current_points"][..., 2].tolist() == [
        [[3.0]],
        [[3.0]],
    ]
    expected_global = torch.einsum(
        "nij,nhwj->nhwi",
        staged.stage2.camera_poses,
        homogenize_points(staged.stage2.local_points),
    )[..., :3]
    assert torch.equal(staged.stage2.global_points, expected_global)


def test_stage2_retries_detector_candidate_rejected_by_stage1_geometry():
    context, optimizer_config = _context(
        _predictions(),
        ReconstructionMode.TRADITIONAL_SECOND_GLOBAL,
        SequencedAnchor(scales=(3.0, 3.0)),
    )
    evidence = _RecordingEvidence(reject_first=True)
    stage1_processor, stage2_processor = _build_stage_processors(
        optimizer_config
    )

    staged = TraditionalSecondGlobalReconstructionMode(
        detector=_OneCandidateDetector(),
        evidence=evidence,
        optimizer_config=optimizer_config,
        stage1_processor=stage1_processor,
        stage2_processor=stage2_processor,
        register_adjacent=lambda *arguments: _identity_sim3(),
        build_graphs=lambda results, threshold: (tuple(results), threshold),
        residual_align=_identity_residual([]),
    ).run(context)

    assert len(evidence.window_types) == 2
    assert staged.stage1.diagnostics.constraint_count == 0
    assert staged.stage2.diagnostics.constraint_count == 1
