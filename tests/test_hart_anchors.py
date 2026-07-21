import numpy as np

from inference_engine.anchor_propagation.anchors import (
    estimate_direct_anchors,
    high_confidence_mask,
)
from inference_engine.anchor_propagation.correspondence import build_track_window
from inference_engine.anchor_propagation.segmentation import _frame_from_layers


def test_high_confidence_mask_uses_quantile_only_as_a_gate():
    confidence = np.asarray([[1.0, 2.0], [3.0, np.nan]])
    np.testing.assert_array_equal(
        high_confidence_mask(confidence, 0.5),
        [[False, True], [True, False]],
    )


def test_direct_anchor_uses_intersection_core_and_existing_irls_direction():
    labels = np.zeros((2, 3), dtype=np.intp)
    frame = _frame_from_layers(labels, labels)
    tracks = build_track_window((labels,), threshold=0.3)
    previous_depth = np.full((1, 2, 3), 6.0)
    current_depth = np.full((1, 2, 3), 3.0)
    confidence = np.ones((1, 2, 3))
    calls = []

    def fake_irls(src, tgt, mask):
        calls.append((src.copy(), tgt.copy(), mask.copy()))
        return np.median(tgt[mask] / src[mask])

    anchors, diagnostics = estimate_direct_anchors(
        previous_depth,
        current_depth,
        confidence,
        confidence,
        (frame,),
        (frame,),
        tracks,
        tracks,
        confidence_quantile=0.5,
        corr_iou_thresh=0.3,
        anchor_min_pixels=4,
        irls=fake_irls,
    )

    assert len(anchors) == 1
    assert anchors[0].scale == 2.0
    assert anchors[0].pixel_count == 6
    assert diagnostics["direct_anchor_count"] == 1
    assert diagnostics["pose_consensus_valid_pixels"] == 6
    np.testing.assert_array_equal(calls[0][0], current_depth[0])
    np.testing.assert_array_equal(calls[0][1], previous_depth[0])


def test_direct_anchor_rejects_small_high_confidence_core():
    labels = np.zeros((2, 3), dtype=np.intp)
    frame = _frame_from_layers(labels, labels)
    tracks = build_track_window((labels,), threshold=0.3)
    confidence = np.arange(6, dtype=np.float64).reshape(1, 2, 3)

    anchors, diagnostics = estimate_direct_anchors(
        np.full((1, 2, 3), 2.0),
        np.ones((1, 2, 3)),
        confidence,
        confidence,
        (frame,),
        (frame,),
        tracks,
        tracks,
        confidence_quantile=0.9,
        corr_iou_thresh=0.3,
        anchor_min_pixels=3,
    )

    assert anchors == ()
    assert diagnostics["rejected_small_core_count"] == 1
    assert diagnostics["pose_consensus_valid_pixels"] == 2
