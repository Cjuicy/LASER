import numpy as np
import pytest
import torch

from inference_engine.segmentation.window_selection import WindowReferenceSelector
from pipeline.config import load_pipeline_config


def _segmentation_config(*overrides):
    return load_pipeline_config(
        "configs/pipeline/default.yaml",
        (
            "segmentation.window_reference.enabled=true",
            *overrides,
        ),
    ).config.segmentation


def _identity_window(frame_count=3):
    intrinsic = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
    )
    local = torch.tensor(
        [
            [[-0.5, -0.5, 1.0], [0.0, -0.5, 1.0], [0.5, -0.5, 1.0]],
            [[-0.5, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0]],
            [[-0.5, 0.5, 1.0], [0.0, 0.5, 1.0], [0.5, 0.5, 1.0]],
        ]
    )
    return (
        local.unsqueeze(0).repeat(frame_count, 1, 1, 1),
        torch.eye(4).repeat(frame_count, 1, 1),
        torch.zeros((frame_count, 3, 3)),
        intrinsic,
    )


def _selection_fallback_fixture(reason):
    points, poses, confidence, intrinsic = _identity_window()
    if reason == "single_frame":
        points = points[:1]
        poses = poses[:1]
        confidence = confidence[:1]
    elif reason == "missing_intrinsic":
        intrinsic = None
    elif reason == "invalid_intrinsic":
        intrinsic = intrinsic.clone()
        intrinsic[2, 2] = 2.0
    elif reason == "intrinsic_incompatible":
        points = points.clone()
        points[..., 0] = 1.0
    elif reason == "invalid_geometry":
        poses = poses.clone()
        poses[:, 3, 3] = 0.0
    elif reason == "no_reference":
        confidence = torch.full_like(confidence, float("nan"))
    else:  # pragma: no cover - parametrization controls the fixture contract.
        raise AssertionError(f"unknown fallback fixture: {reason}")
    return points, poses, confidence, intrinsic


def test_selector_returns_keyframes_before_labels_exist():
    points, poses, confidence, intrinsic = _identity_window(frame_count=3)
    selector = WindowReferenceSelector(_segmentation_config())

    selected = selector.select(
        point_maps=points,
        camera_poses=poses,
        confidence=confidence,
        reference_intrinsic=intrinsic,
    )

    assert selected.fallback_reason is None
    assert selected.indices == (1,)
    assert selected.rejected_indices == ()
    assert selected.coverage_ratio == pytest.approx(1.0)
    assert selected.diagnostics["selected_count"] == 1


@pytest.mark.parametrize(
    "reason",
    (
        "single_frame",
        "missing_intrinsic",
        "invalid_intrinsic",
        "intrinsic_incompatible",
        "invalid_geometry",
        "no_reference",
    ),
)
def test_selector_returns_explicit_runtime_fallback(reason):
    points, poses, confidence, intrinsic = _selection_fallback_fixture(reason)

    selected = WindowReferenceSelector(_segmentation_config()).select(
        point_maps=points,
        camera_poses=poses,
        confidence=confidence,
        reference_intrinsic=intrinsic,
    )

    assert selected.indices == ()
    assert selected.fallback_reason == reason
    assert selected.coverage_ratio == 0.0
    assert selected.best_scores.shape == (points.shape[0],)
    assert np.all(selected.best_scores == 0.0)
