from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.utils.geometry import (
    accumulate_sim3,
    apply_sim3_to_pose,
    closed_form_inverse_sim3,
)
from loop_closure.methods.traditional import identity_sim3
from loop_closure.types import LoopSolution, Sim3, validate_sim3
from reconstruction.modes.traditional import TraditionalWindowState
from reconstruction.residual_alignment import (
    ResidualAlignmentResult,
    align_post_anchor_window,
)


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


__all__ = [
    "MaterializedTraditionalWindow",
    "SecondGlobalWindowState",
    "build_second_global_states",
    "materialize_traditional_stage1_windows",
]
