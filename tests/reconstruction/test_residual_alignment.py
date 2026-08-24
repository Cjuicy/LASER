from __future__ import annotations

import importlib
import math

import pytest
import torch

from reconstruction.residual_alignment import (
    ResidualAlignmentResult,
    align_post_anchor_window,
    summarize_residual_alignments,
)


def _points(value: float) -> torch.Tensor:
    return torch.full((1, 1, 1, 3), value, dtype=torch.float32)


def _poses(x: float) -> torch.Tensor:
    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    poses[:, 0, 3] = x
    return poses


def test_post_anchor_residual_scales_points_and_transforms_poses():
    residual_alignment = importlib.import_module(
        "reconstruction.residual_alignment"
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([3.0, 4.0, 0.0])

    result = residual_alignment.align_post_anchor_window(
        previous_points=_points(1.0),
        previous_poses=_poses(0.0),
        previous_confidence=torch.ones((1, 1, 1)),
        current_points=_points(5.0),
        current_poses=_poses(2.0),
        current_confidence=torch.ones((1, 1, 1)),
        overlap=1,
        confidence_keep_ratio=1.0,
        register_adjacent=lambda *args: (2.0, rotation, translation),
    )

    assert torch.equal(result.local_points, _points(10.0))
    assert torch.allclose(
        result.camera_poses[:, :3, 3],
        torch.tensor([[3.0, 8.0, 0.0]]),
    )
    assert result.correspondence_count == 1
    assert result.abs_log_scale == pytest.approx(math.log(2.0))
    assert result.rotation_rad == pytest.approx(math.pi / 2)
    assert result.translation_norm == pytest.approx(5.0)


@pytest.mark.parametrize(
    "transform",
    [
        (0.0, torch.eye(3), torch.zeros(3)),
        (1.0, torch.diag(torch.tensor([-1.0, 1.0, 1.0])), torch.zeros(3)),
        (1.0, 2.0 * torch.eye(3), torch.zeros(3)),
        (1.0, torch.eye(3), torch.tensor([float("nan"), 0.0, 0.0])),
        (1.0, torch.eye(2), torch.zeros(3)),
    ],
    ids=[
        "zero-scale",
        "reflected-rotation",
        "non-orthonormal-rotation",
        "nan-component",
        "shape-mismatch",
    ],
)
def test_post_anchor_residual_rejects_invalid_sim3(transform):
    with pytest.raises(ValueError, match="post-anchor residual"):
        align_post_anchor_window(
            previous_points=_points(1.0),
            previous_poses=_poses(0.0),
            previous_confidence=torch.ones((1, 1, 1)),
            current_points=_points(2.0),
            current_poses=_poses(1.0),
            current_confidence=torch.ones((1, 1, 1)),
            overlap=1,
            confidence_keep_ratio=1.0,
            register_adjacent=lambda *args: transform,
        )


def test_post_anchor_residual_rejects_changed_output_shape():
    with pytest.raises(ValueError, match="post-anchor residual changed pose shape"):
        align_post_anchor_window(
            previous_points=_points(1.0),
            previous_poses=_poses(0.0),
            previous_confidence=torch.ones((1, 1, 1)),
            current_points=_points(2.0),
            current_poses=_poses(1.0),
            current_confidence=torch.ones((1, 1, 1)),
            overlap=1,
            confidence_keep_ratio=1.0,
            register_adjacent=lambda *args: (1.0, torch.eye(3), torch.zeros(3)),
            apply_pose_sim3=lambda *args: torch.empty((0, 4, 4)),
        )


def test_post_anchor_residual_rejects_zero_correspondences():
    with pytest.raises(ValueError, match="post-anchor residual.*no shared"):
        align_post_anchor_window(
            previous_points=torch.ones((1, 1, 2, 3)),
            previous_poses=_poses(0.0),
            previous_confidence=torch.tensor([[[1.0, 0.0]]]),
            current_points=torch.ones((1, 1, 2, 3)),
            current_poses=_poses(1.0),
            current_confidence=torch.tensor([[[0.0, 1.0]]]),
            overlap=1,
            confidence_keep_ratio=0.5,
            register_adjacent=lambda *args: (1.0, torch.eye(3), torch.zeros(3)),
        )


def test_residual_summary_reports_applied_skipped_and_literal_metrics():
    results = (
        ResidualAlignmentResult(
            local_points=_points(1.0),
            camera_poses=_poses(0.0),
            sim3=(2.0, torch.eye(3), torch.zeros(3)),
            correspondence_count=1,
            abs_log_scale=1.0,
            rotation_rad=2.0,
            translation_norm=3.0,
        ),
        ResidualAlignmentResult(
            local_points=_points(1.0),
            camera_poses=_poses(0.0),
            sim3=(3.0, torch.eye(3), torch.zeros(3)),
            correspondence_count=1,
            abs_log_scale=3.0,
            rotation_rad=4.0,
            translation_norm=5.0,
        ),
    )

    assert summarize_residual_alignments(results, window_count=3) == {
        "residual_applied_window_count": 2,
        "residual_skipped_window_count": 1,
        "max_abs_log_residual_scale": 3.0,
        "mean_abs_log_residual_scale": 2.0,
        "max_residual_rotation_rad": 4.0,
        "mean_residual_rotation_rad": 3.0,
        "max_residual_translation_norm": 5.0,
        "mean_residual_translation_norm": 4.0,
    }


def test_residual_summary_rejects_more_results_than_windows():
    result = ResidualAlignmentResult(
        local_points=_points(1.0),
        camera_poses=_poses(0.0),
        sim3=(1.0, torch.eye(3), torch.zeros(3)),
        correspondence_count=1,
        abs_log_scale=0.0,
        rotation_rad=0.0,
        translation_norm=0.0,
    )

    with pytest.raises(ValueError, match="exceeds window count"):
        summarize_residual_alignments((result,), window_count=0)
