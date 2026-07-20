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


def test_first_window_initializes_identity_local_scale_state():
    propagator = HartAnchorPropagator(anchor_min_pixels=1)
    result = propagator.refine(
        previous_registration_state=None,
        previous_anchor_state=None,
        current_base_points=_points(3.0),
        current_confidence=np.ones((2, 2, 3)),
        current_segments=_segments(),
        overlap=1,
    )

    assert result.local_scale_mask.shape == (2, 2, 3, 1)
    np.testing.assert_array_equal(result.local_scale_mask, 1.0)
    assert result.next_state.local_scale_tail.shape == (1, 2, 3)


def test_next_window_uses_previous_base_times_previous_local_scale():
    propagator = HartAnchorPropagator(anchor_min_pixels=1)
    previous_segments = _segments(frames=1)
    previous_registration = RegistrationState(
        base_points_tail=_points(3.0, frames=1),
        base_poses_tail=torch.eye(4)[None],
    )
    previous_anchor = AnchorPropagationState(
        local_scale_tail=np.full((1, 2, 3), 2.0, dtype=np.float32),
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

    np.testing.assert_allclose(result.local_scale_mask, 2.0, rtol=1e-5)
    assert result.diagnostics["direct_anchor_count"] == 1


def test_previous_registration_and_anchor_states_must_be_atomic():
    propagator = HartAnchorPropagator(anchor_min_pixels=1)
    registration = RegistrationState(
        base_points_tail=_points(3.0, frames=1),
        base_poses_tail=torch.eye(4)[None],
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
