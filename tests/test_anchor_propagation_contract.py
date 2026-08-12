import numpy as np
import torch

from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.utils.depth import match_segmentation_seq


def identity_anchor_fixture():
    points = np.zeros((2, 2, 3, 3), dtype=np.float32)
    points[..., 2] = 1.0
    labels = [
        np.zeros((2, 3), dtype=np.intp),
        np.zeros((2, 3), dtype=np.intp),
    ]
    return (
        points.copy(),
        points.copy(),
        match_segmentation_seq(labels, iou_thresh=0.3),
        match_segmentation_seq(labels, iou_thresh=0.3),
    )


def test_anchor_propagation_keeps_existing_identity_scale_result():
    (
        source_points,
        target_points,
        source_graphs,
        target_graphs,
    ) = identity_anchor_fixture()
    propagator = AnchorPropagator(correspondence_iou_threshold=0.4)
    scale = propagator.propagate(
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap=1,
    )
    assert tuple(scale.shape) == (*target_points.shape[:-1], 1)
    torch.testing.assert_close(scale, torch.ones_like(scale))
