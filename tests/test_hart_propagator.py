import numpy as np
import pytest
import torch

from inference_engine.anchor_propagation.hart import HartAnchorPropagator
from inference_engine.anchor_propagation.segmentation import _frame_from_layers
from inference_engine.anchor_propagation.types import (
    AnchorPropagationState,
    RegistrationState,
    SegmentationWindow,
)


def _points(depth, frames=2, height=2, width=3):
    points = np.zeros((frames, height, width, 3), dtype=np.float32)
    points[..., -1] = depth
    return torch.from_numpy(points)


def _segments(frames=2):
    labels = np.zeros((2, 3), dtype=np.intp)
    return SegmentationWindow(
        tuple(_frame_from_layers(labels, labels) for _ in range(frames)),
        "depth",
    )


def test_first_window_initializes_identity_residual_and_no_pose_support():
    propagator = HartAnchorPropagator(anchor_min_pixels=1)
    result = propagator.refine(
        previous_registration_state=None,
        previous_anchor_state=None,
        current_base_points=_points(3.0),
        current_confidence=np.ones((2, 2, 3)),
        current_segments=_segments(),
        overlap=1,
    )

    assert result.window_scale == 1.0
    assert result.local_residual_mask.shape == (2, 2, 3, 1)
    np.testing.assert_array_equal(result.local_residual_mask, 1.0)
    np.testing.assert_array_equal(result.pose_support_mask, False)
    assert result.next_state.local_residual_tail.shape == (1, 2, 3)
    assert not result.diagnostics["pose_consensus_accepted"]


def test_uniform_regional_scale_becomes_window_scale_and_unit_residual():
    propagator = HartAnchorPropagator(anchor_min_pixels=1)
    previous_segments = _segments(frames=1)
    previous_registration = RegistrationState(
        final_base_points_tail=_points(6.0, frames=1),
        final_base_poses_tail=torch.eye(4)[None],
        pose_support_mask_tail=np.ones((1, 2, 3), dtype=bool),
    )
    previous_anchor = AnchorPropagationState(
        local_residual_tail=np.ones((1, 2, 3), dtype=np.float32),
        confidence_tail=np.ones((1, 2, 3)),
        segments_tail=previous_segments.frames,
    )

    result = propagator.refine(
        previous_registration_state=previous_registration,
        previous_anchor_state=previous_anchor,
        current_base_points=_points(3.0),
        current_confidence=np.ones((2, 2, 3)),
        current_segments=_segments(),
        overlap=1,
    )

    assert result.window_scale == pytest.approx(2.0)
    np.testing.assert_allclose(result.local_residual_mask, 1.0, rtol=1e-5)
    np.testing.assert_array_equal(result.pose_support_mask, True)
    np.testing.assert_allclose(result.next_state.local_residual_tail, 1.0)
    assert result.diagnostics["direct_anchor_count"] == 1
    assert result.diagnostics["pose_consensus_valid_pixels"] == 6
    assert result.diagnostics["pose_consensus_support_pixels"] == 6
    assert result.diagnostics["pose_consensus_support_ratio"] == 1.0
    assert result.diagnostics["pose_consensus_accepted"]


def test_previous_registration_and_anchor_states_must_be_atomic():
    propagator = HartAnchorPropagator(anchor_min_pixels=1)
    registration = RegistrationState(
        final_base_points_tail=_points(3.0, frames=1),
        final_base_poses_tail=torch.eye(4)[None],
        pose_support_mask_tail=np.zeros((1, 2, 3), dtype=bool),
    )
    with pytest.raises(ValueError, match="provided together"):
        propagator.refine(
            previous_registration_state=registration,
            previous_anchor_state=None,
            current_base_points=_points(3.0),
            current_confidence=np.ones((2, 2, 3)),
            current_segments=_segments(),
            overlap=1,
        )
