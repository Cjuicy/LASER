from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.utils.geometry import apply_sim3_to_pose
from loop_closure.types import Sim3, validate_sim3
from reconstruction.shared import mutual_confidence_mask


@dataclass(frozen=True)
class ResidualAlignmentResult:
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    sim3: Sim3
    correspondence_count: int
    abs_log_scale: float
    rotation_rad: float
    translation_norm: float


def _validated_result(
    local_points: torch.Tensor,
    camera_poses: torch.Tensor,
    sim3: Sim3,
    correspondence_count: int,
) -> ResidualAlignmentResult:
    scale, rotation, translation = sim3
    scale_value = float(torch.as_tensor(scale).detach().cpu().item())
    rotation_tensor = torch.as_tensor(rotation)
    translation_tensor = torch.as_tensor(translation)
    rotation_check = rotation_tensor.detach().to(dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64, device=rotation_check.device)
    if not torch.allclose(
        rotation_check.T @ rotation_check,
        identity,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("post-anchor residual rotation must be orthonormal")
    determinant = float(torch.linalg.det(rotation_check).cpu().item())
    if not math.isclose(determinant, 1.0, abs_tol=1e-4, rel_tol=1e-4):
        raise ValueError("post-anchor residual rotation determinant must be +1")
    cosine = torch.clamp((torch.trace(rotation_check) - 1.0) / 2.0, -1.0, 1.0)
    translation_check = translation_tensor.detach().to(dtype=torch.float64)
    metrics = (
        abs(math.log(scale_value)),
        float(torch.acos(cosine).detach().cpu().item()),
        float(torch.linalg.vector_norm(translation_check).cpu().item()),
    )
    if not all(math.isfinite(value) for value in metrics):
        raise ValueError("post-anchor residual metrics must be finite")
    if not torch.isfinite(local_points).all() or not torch.isfinite(camera_poses).all():
        raise ValueError("post-anchor residual outputs must be finite")
    return ResidualAlignmentResult(
        local_points=local_points,
        camera_poses=camera_poses,
        sim3=sim3,
        correspondence_count=correspondence_count,
        abs_log_scale=metrics[0],
        rotation_rad=metrics[1],
        translation_norm=metrics[2],
    )


def align_post_anchor_window(
    *,
    previous_points: torch.Tensor,
    previous_poses: torch.Tensor,
    previous_confidence: torch.Tensor,
    current_points: torch.Tensor,
    current_poses: torch.Tensor,
    current_confidence: torch.Tensor,
    overlap: int,
    confidence_keep_ratio: float,
    register_adjacent: Callable = register_adjacent_windows,
    apply_pose_sim3: Callable = apply_sim3_to_pose,
) -> ResidualAlignmentResult:
    mask = mutual_confidence_mask(
        previous_confidence,
        current_confidence,
        overlap,
        confidence_keep_ratio,
        context="post-anchor residual registration",
    )
    correspondence_count = int(torch.count_nonzero(mask).item())
    if correspondence_count == 0:
        raise ValueError("post-anchor residual registration has no correspondences")

    sim3 = register_adjacent(
        previous_points[-overlap:],
        current_points[:overlap],
        previous_poses[-overlap:],
        current_poses[:overlap],
        mask,
    )
    validate_sim3(sim3, context="post-anchor residual Sim(3)")
    scale, rotation, translation = sim3
    scale_tensor = torch.as_tensor(
        scale,
        device=current_points.device,
        dtype=current_points.dtype,
    )
    local_points = scale_tensor * current_points
    camera_poses = apply_pose_sim3(
        current_poses,
        scale,
        rotation.to(current_poses),
        translation.to(current_poses),
    )
    if local_points.shape != current_points.shape:
        raise ValueError("post-anchor residual changed point shape")
    if camera_poses.shape != current_poses.shape:
        raise ValueError("post-anchor residual changed pose shape")
    return _validated_result(
        local_points,
        camera_poses,
        sim3,
        correspondence_count,
    )


def summarize_residual_alignments(
    results: Sequence[ResidualAlignmentResult],
    window_count: int,
) -> dict[str, int | float]:
    if window_count < len(results):
        raise ValueError("residual result count exceeds window count")
    scales = [item.abs_log_scale for item in results]
    rotations = [item.rotation_rad for item in results]
    translations = [item.translation_norm for item in results]
    return {
        "residual_applied_window_count": len(results),
        "residual_skipped_window_count": window_count - len(results),
        "max_abs_log_residual_scale": max(scales, default=0.0),
        "mean_abs_log_residual_scale": (
            sum(scales) / len(scales) if scales else 0.0
        ),
        "max_residual_rotation_rad": max(rotations, default=0.0),
        "mean_residual_rotation_rad": (
            sum(rotations) / len(rotations) if rotations else 0.0
        ),
        "max_residual_translation_norm": max(translations, default=0.0),
        "mean_residual_translation_norm": (
            sum(translations) / len(translations) if translations else 0.0
        ),
    }
