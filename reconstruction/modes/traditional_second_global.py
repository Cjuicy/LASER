from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import time

import torch

from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.segmentation import build_temporal_graphs
from inference_engine.utils.geometry import (
    accumulate_sim3,
    apply_sim3_to_pose,
    closed_form_inverse_sim3,
    homogenize_points,
)
from loop_closure.methods.second_global import SecondGlobalLoopProcessor
from loop_closure.methods.traditional import identity_sim3
from loop_closure.detection import LoopDetector
from loop_closure.evidence import LoopEvidenceProvider
from loop_closure.methods.traditional import TraditionalLoopProcessor
from loop_closure.types import (
    LoopCandidate,
    LoopConstraint,
    LoopSolution,
    Sim3,
    validate_sim3,
)
from pipeline.artifacts import (
    ReconstructionArtifact,
    ReconstructionDiagnostics,
    StagedReconstructionArtifacts,
)
from pipeline.config import OptimizerConfig, ReconstructionMode
from reconstruction.modes.base import ReconstructionContext
from reconstruction.modes.traditional import (
    TraditionalReconstructionMode,
    TraditionalWindowState,
)
from reconstruction.residual_alignment import (
    ResidualAlignmentResult,
    align_post_anchor_window,
    summarize_residual_alignments,
)
from reconstruction.shared import segment_and_refine_window


@dataclass(frozen=True)
class MaterializedTraditionalWindow:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    stage1_absolute_sim3: Sim3


@dataclass(frozen=True)
class SecondGlobalWindowState:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    sim3_abs: Sim3
    sim3_edge: Sim3 | None


def materialize_traditional_stage1_windows(
    states: Sequence[TraditionalWindowState],
    solution: LoopSolution,
    *,
    apply_pose_sim3: Callable = apply_sim3_to_pose,
) -> tuple[MaterializedTraditionalWindow, ...]:
    if not states:
        raise ValueError("traditional Stage 1 materialization requires states")
    if len(states) != len(solution.optimized_transforms):
        raise ValueError(
            "traditional Stage 1 materialization state/solution count mismatch"
        )

    reference = identity_sim3("cpu")
    windows = []
    for expected_index, (state, relative) in enumerate(
        zip(states, solution.optimized_transforms, strict=True)
    ):
        if state.window_index != expected_index:
            raise ValueError(
                "traditional Stage 1 materialization requires ordered windows"
            )
        validate_sim3(relative, context="traditional Stage 1 relative Sim(3)")
        absolute = accumulate_sim3(reference, relative)
        validate_sim3(absolute, context="traditional Stage 1 absolute Sim(3)")
        scale, rotation, translation = absolute
        local_points = state.local_points.clone()
        if state.anchor_scale_mask is None:
            local_points = (
                torch.as_tensor(
                    scale,
                    device=local_points.device,
                    dtype=local_points.dtype,
                )
                * local_points
            )
        else:
            local_points = (
                torch.as_tensor(
                    reference[0],
                    device=local_points.device,
                    dtype=local_points.dtype,
                )
                * state.anchor_scale_mask.to(local_points)
                * local_points
            )
        camera_poses = apply_pose_sim3(
            state.camera_poses.clone(),
            scale,
            rotation.to(state.camera_poses),
            translation.to(state.camera_poses),
        )
        if local_points.shape != state.local_points.shape:
            raise ValueError(
                "traditional Stage 1 materialization changed point shape"
            )
        if camera_poses.shape != state.camera_poses.shape:
            raise ValueError(
                "traditional Stage 1 materialization changed pose shape"
            )
        if not torch.isfinite(local_points).all() or not torch.isfinite(
            camera_poses
        ).all():
            raise ValueError(
                "traditional Stage 1 materialization outputs must be finite"
            )
        windows.append(
            MaterializedTraditionalWindow(
                window_index=state.window_index,
                frame_start=state.frame_start,
                frame_end=state.frame_end,
                local_points=local_points,
                camera_poses=camera_poses,
                confidence=state.confidence.clone(),
                stage1_absolute_sim3=absolute,
            )
        )
        reference = absolute
    return tuple(windows)


def build_second_global_states(
    windows: Sequence[MaterializedTraditionalWindow],
    *,
    overlap: int,
    confidence_keep_ratio: float,
    residual_align: Callable = align_post_anchor_window,
    register_adjacent: Callable = register_adjacent_windows,
    apply_pose_sim3: Callable = apply_sim3_to_pose,
) -> tuple[
    tuple[SecondGlobalWindowState, ...],
    tuple[ResidualAlignmentResult, ...],
]:
    if not windows:
        raise ValueError("second global state construction requires windows")
    states: list[SecondGlobalWindowState] = []
    residual_results: list[ResidualAlignmentResult] = []
    for expected_index, window in enumerate(windows):
        if window.window_index != expected_index:
            raise ValueError("second global windows must be in index order")
        if not states:
            sim3_abs = identity_sim3(window.local_points.device)
            local_points = window.local_points.clone()
            camera_poses = window.camera_poses.clone()
            sim3_edge = None
        else:
            previous = states[-1]
            result = residual_align(
                previous_points=previous.local_points,
                previous_poses=previous.camera_poses,
                previous_confidence=previous.confidence,
                current_points=window.local_points,
                current_poses=window.camera_poses,
                current_confidence=window.confidence,
                overlap=overlap,
                confidence_keep_ratio=confidence_keep_ratio,
                register_adjacent=register_adjacent,
                apply_pose_sim3=apply_pose_sim3,
            )
            if not isinstance(result, ResidualAlignmentResult):
                raise ValueError(
                    "second global residual alignment returned invalid result"
                )
            if result.local_points.shape != window.local_points.shape:
                raise ValueError("second global residual changed point shape")
            if result.camera_poses.shape != window.camera_poses.shape:
                raise ValueError("second global residual changed pose shape")
            sim3_abs = result.sim3
            validate_sim3(
                sim3_abs,
                context="second global residual absolute Sim(3)",
            )
            sim3_edge = accumulate_sim3(
                closed_form_inverse_sim3(*previous.sim3_abs),
                sim3_abs,
            )
            validate_sim3(
                sim3_edge,
                context="second global residual sequential Sim(3) edge",
            )
            local_points = result.local_points
            camera_poses = result.camera_poses
            residual_results.append(result)
        if not torch.isfinite(local_points).all() or not torch.isfinite(
            camera_poses
        ).all():
            raise ValueError("second global residual outputs must be finite")
        states.append(
            SecondGlobalWindowState(
                window_index=window.window_index,
                frame_start=window.frame_start,
                frame_end=window.frame_end,
                local_points=local_points,
                camera_poses=camera_poses,
                confidence=window.confidence.clone(),
                sim3_abs=sim3_abs,
                sim3_edge=sim3_edge,
            )
        )
    return tuple(states), tuple(residual_results)


class TraditionalSecondGlobalReconstructionMode:
    def __init__(
        self,
        *,
        detector: LoopDetector,
        evidence: LoopEvidenceProvider,
        optimizer_config: OptimizerConfig,
        stage1_processor: TraditionalLoopProcessor | None = None,
        stage2_processor: SecondGlobalLoopProcessor | None = None,
        register_adjacent: Callable = register_adjacent_windows,
        apply_pose_sim3: Callable = apply_sim3_to_pose,
        build_graphs: Callable = build_temporal_graphs,
        segment_window: Callable = segment_and_refine_window,
        residual_align: Callable = align_post_anchor_window,
    ) -> None:
        self.detector = detector
        self.evidence = evidence
        self.optimizer_config = optimizer_config
        self.stage1_processor = stage1_processor or TraditionalLoopProcessor(
            optimizer_config,
            apply_pose_sim3=apply_pose_sim3,
        )
        self.stage2_processor = stage2_processor or SecondGlobalLoopProcessor(
            optimizer_config,
            apply_pose_sim3=apply_pose_sim3,
        )
        self._register_adjacent = register_adjacent
        self._apply_pose_sim3 = apply_pose_sim3
        self._build_graphs = build_graphs
        self._segment_window = segment_window
        self._residual_align = residual_align
        self.stage1_trace: tuple[TraditionalWindowState, ...] = ()
        self.stage2_trace: tuple[SecondGlobalWindowState, ...] = ()

    def run(
        self,
        context: ReconstructionContext,
    ) -> StagedReconstructionArtifacts:
        if (
            context.reconstruction_mode
            is not ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
        ):
            raise ValueError(
                "TraditionalSecondGlobalReconstructionMode requires "
                "traditional_second_global context"
            )

        recording_detector = _RecordingDetector(self.detector)
        recording_processor = _RecordingTraditionalProcessor(
            self.stage1_processor
        )
        stage1_mode = TraditionalReconstructionMode(
            detector=recording_detector,
            evidence=self.evidence,
            optimizer_config=self.optimizer_config,
            processor=recording_processor,
            register_adjacent=self._register_adjacent,
            apply_pose_sim3=self._apply_pose_sim3,
            build_graphs=self._build_graphs,
            segment_window=self._segment_window,
        )

        started = time.perf_counter()
        stage1 = stage1_mode.run(
            replace(
                context,
                reconstruction_mode=ReconstructionMode.TRADITIONAL,
            )
        )
        stage1_ms = (time.perf_counter() - started) * 1000
        if recording_detector.candidates is None:
            raise RuntimeError("second global Stage 1 did not record candidates")
        if recording_processor.constraints is None:
            raise RuntimeError("second global Stage 1 did not record constraints")
        if recording_processor.solution is None:
            raise RuntimeError("second global Stage 1 did not record solution")
        if not stage1_mode.trace:
            raise RuntimeError("second global Stage 1 did not record windows")
        self.stage1_trace = stage1_mode.trace

        started = time.perf_counter()
        materialized = materialize_traditional_stage1_windows(
            self.stage1_trace,
            recording_processor.solution,
            apply_pose_sim3=self._apply_pose_sim3,
        )
        materialization_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        second_states, residual_results = build_second_global_states(
            materialized,
            overlap=context.window_config.overlap,
            confidence_keep_ratio=(
                context.registration_config.confidence_keep_ratio
            ),
            residual_align=self._residual_align,
            register_adjacent=self._register_adjacent,
            apply_pose_sim3=self._apply_pose_sim3,
        )
        residual_ms = (time.perf_counter() - started) * 1000
        if len(residual_results) != len(second_states) - 1:
            raise RuntimeError(
                "second global residual count does not match window count"
            )
        self.stage2_trace = second_states

        started = time.perf_counter()
        constraints = self.stage2_processor.build_constraints(
            second_states,
            recording_detector.candidates,
            self.evidence,
        )
        solution = self.stage2_processor.optimize(second_states, constraints)
        optimize_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        aggregate = self.stage2_processor.aggregate(second_states, solution)
        aggregate_ms = (time.perf_counter() - started) * 1000
        local_points = aggregate.local_points.detach().cpu()
        camera_poses = aggregate.camera_poses.detach().cpu()
        confidence = aggregate.confidence.detach().cpu()
        if local_points.shape[0] != len(context.frame_ids):
            raise ValueError(
                "second global aggregation did not cover every frame once"
            )
        global_points = torch.einsum(
            "nij,nhwj->nhwi",
            camera_poses,
            homogenize_points(local_points),
        )[..., :3]

        residual_scalars = {
            f"stage2_{name}": value
            for name, value in summarize_residual_alignments(
                residual_results,
                len(second_states),
            ).items()
        }
        stage1_scalars = stage1.diagnostics.mode_scalars
        stage2 = ReconstructionArtifact(
            schema_version=stage1.schema_version,
            frame_ids=stage1.frame_ids,
            local_points=local_points,
            global_points=global_points.detach().cpu(),
            camera_poses=camera_poses,
            confidence=confidence,
            segmentation_method=stage1.segmentation_method,
            reconstruction_mode=(
                ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
            ),
            prediction_key=stage1.prediction_key,
            diagnostics=ReconstructionDiagnostics(
                stage_timings_ms={
                    "stage1_reconstruction": stage1_ms,
                    "stage1_materialization": materialization_ms,
                    "stage2_adjacent_residual": residual_ms,
                    "stage2_constraints_optimization": optimize_ms,
                    "stage2_aggregation": aggregate_ms,
                },
                segmentation_summaries=(
                    stage1.diagnostics.segmentation_summaries
                ),
                candidate_count=len(recording_detector.candidates),
                constraint_count=len(constraints),
                mode_scalars={
                    "window_count": len(second_states),
                    "stage1_candidate_count": (
                        stage1.diagnostics.candidate_count
                    ),
                    "stage1_constraint_count": (
                        stage1.diagnostics.constraint_count
                    ),
                    "stage1_used_no_loop_path": int(
                        stage1_scalars.get("used_no_loop_path", 0)
                    ),
                    "stage2_candidate_count": len(
                        recording_detector.candidates
                    ),
                    "stage2_constraint_count": len(constraints),
                    "stage2_used_no_loop_path": int(
                        solution.used_no_loop_path
                    ),
                    **residual_scalars,
                    "stage2_max_abs_log_scale_delta": (
                        aggregate.mode_scalars["max_abs_log_scale_delta"]
                    ),
                    "stage2_mean_abs_log_scale_delta": (
                        aggregate.mode_scalars["mean_abs_log_scale_delta"]
                    ),
                },
            ),
        )
        return StagedReconstructionArtifacts(stage1=stage1, stage2=stage2)


class _RecordingDetector:
    def __init__(self, delegate: LoopDetector) -> None:
        self.delegate = delegate
        self.candidates: tuple[LoopCandidate, ...] | None = None

    def detect(self, manifest, images) -> tuple[LoopCandidate, ...]:
        candidates = self.delegate.detect(manifest, images)
        if not isinstance(candidates, tuple):
            raise ValueError("loop detector candidates must be a tuple")
        self.candidates = candidates
        return candidates


class _RecordingTraditionalProcessor:
    def __init__(self, delegate: TraditionalLoopProcessor) -> None:
        self.delegate = delegate
        self.constraints: tuple[LoopConstraint, ...] | None = None
        self.solution: LoopSolution | None = None

    def build_constraints(self, states, candidates, evidence):
        constraints = self.delegate.build_constraints(
            states,
            candidates,
            evidence,
        )
        self.constraints = tuple(constraints)
        return constraints

    def optimize(self, states, constraints) -> LoopSolution:
        self.solution = self.delegate.optimize(states, constraints)
        return self.solution

    def aggregate(self, states, solution):
        return self.delegate.aggregate(states, solution)


__all__ = [
    "MaterializedTraditionalWindow",
    "SecondGlobalWindowState",
    "TraditionalSecondGlobalReconstructionMode",
    "build_second_global_states",
    "materialize_traditional_stage1_windows",
]
