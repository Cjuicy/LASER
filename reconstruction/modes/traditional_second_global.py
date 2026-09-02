from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from inference_engine.utils.geometry import accumulate_sim3, apply_sim3_to_pose
from loop_closure.methods.traditional import identity_sim3
from loop_closure.types import LoopSolution, Sim3, validate_sim3
from reconstruction.modes.traditional import TraditionalWindowState


@dataclass(frozen=True)
class MaterializedTraditionalWindow:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    stage1_absolute_sim3: Sim3


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


__all__ = [
    "MaterializedTraditionalWindow",
    "materialize_traditional_stage1_windows",
]
