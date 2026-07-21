import numpy as np
import pytest

from inference_engine.anchor_propagation.consensus import (
    STATUS_CONFLICT,
    STATUS_DIRECT,
    STATUS_LEAF_FILL,
    STATUS_NO_ANCHOR,
    STATUS_PARENT_FILL,
)
from inference_engine.anchor_propagation.pose_consensus import (
    decompose_regional_scales,
    select_pose_consensus,
)
from inference_engine.anchor_propagation.types import DirectAnchor


def _anchor(segment, scale, pixels, *, previous=None):
    return DirectAnchor(
        current_frame=0,
        current_anchor_id=segment,
        current_segment_id=segment,
        previous_segment_id=(segment + 10 if previous is None else previous),
        scale=scale,
        pixel_count=pixels,
    )


def test_strict_majority_selects_unweighted_track_median():
    anchors = (
        _anchor(segment=1, scale=1.03, pixels=30),
        _anchor(segment=2, scale=1.05, pixels=31),
        _anchor(segment=3, scale=0.80, pixels=10),
    )

    result = select_pose_consensus(
        anchors,
        {1: 1.03, 2: 1.05, 3: 0.80},
        valid_pixel_count=100,
        threshold=0.05,
    )

    assert result.accepted
    assert result.window_scale == pytest.approx(1.04)
    assert result.segment_ids == frozenset({1, 2})
    assert result.group_count == 2
    assert result.support_pixels == 61
    assert result.valid_pixels == 100
    assert result.support_ratio == pytest.approx(0.61)


def test_track_pixel_count_does_not_weight_window_scale():
    result = select_pose_consensus(
        (
            _anchor(segment=1, scale=1.00, pixels=90),
            _anchor(segment=2, scale=1.04, pixels=1),
        ),
        {1: 1.00, 2: 1.04},
        valid_pixel_count=100,
        threshold=0.05,
    )

    assert result.accepted
    assert result.window_scale == pytest.approx(1.02)
    assert result.support_pixels == 91


def test_exactly_half_support_is_rejected():
    result = select_pose_consensus(
        (_anchor(1, 1.02, 50),),
        {1: 1.02},
        valid_pixel_count=100,
        threshold=0.05,
    )

    assert not result.accepted
    assert result.window_scale == 1.0
    assert result.segment_ids == frozenset()
    assert result.support_ratio == pytest.approx(0.5)


def test_conflicting_or_unresolved_evidence_stays_in_denominator():
    result = select_pose_consensus(
        (
            _anchor(segment=1, scale=1.02, pixels=40),
            _anchor(segment=9, scale=0.70, pixels=50, previous=19),
        ),
        {1: 1.02},
        valid_pixel_count=100,
        threshold=0.05,
    )

    assert not result.accepted
    assert result.group_count == 1
    assert result.support_pixels == 40
    assert result.valid_pixels == 100


def test_no_candidate_group_returns_identity_consensus():
    result = select_pose_consensus(
        (_anchor(segment=9, scale=0.7, pixels=80),),
        {},
        valid_pixel_count=100,
        threshold=0.05,
    )

    assert not result.accepted
    assert result.window_scale == 1.0
    assert result.segment_ids == frozenset()
    assert result.group_count == 0
    assert result.support_pixels == 0
    assert result.support_ratio == 0.0


def test_unresolved_pixels_do_not_cancel_window_scale():
    regional = np.asarray([[[1.04, 1.04, 0.91, 1.0, 1.0]]])
    statuses = np.asarray(
        [[[
            STATUS_DIRECT,
            STATUS_LEAF_FILL,
            STATUS_DIRECT,
            STATUS_NO_ANCHOR,
            STATUS_CONFLICT,
        ]]]
    )
    segment_maps = np.asarray([[[1, 1, 2, 3, 4]]])

    residual, support = decompose_regional_scales(
        regional,
        statuses,
        segment_maps,
        window_scale=1.04,
        pose_segment_ids=frozenset({1}),
    )

    np.testing.assert_allclose(
        residual,
        [[[1.0, 1.0, 0.91 / 1.04, 1.0, 1.0]]],
    )
    np.testing.assert_array_equal(
        support,
        [[[True, False, False, False, False]]],
    )
    assert residual.dtype == np.float32
    assert support.dtype == np.bool_


def test_parent_fill_is_resolved_but_never_pose_support():
    residual, support = decompose_regional_scales(
        np.asarray([[[2.0, 2.0]]]),
        np.asarray([[[STATUS_DIRECT, STATUS_PARENT_FILL]]]),
        np.asarray([[[7, 7]]]),
        window_scale=2.0,
        pose_segment_ids=frozenset({7}),
    )

    np.testing.assert_array_equal(residual, 1.0)
    np.testing.assert_array_equal(support, [[[True, False]]])


def test_pose_consensus_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="valid_pixel_count"):
        select_pose_consensus((), {}, valid_pixel_count=-1, threshold=0.05)
    with pytest.raises(ValueError, match="threshold"):
        select_pose_consensus((), {}, valid_pixel_count=0, threshold=-0.01)
    with pytest.raises(ValueError, match="window_scale"):
        decompose_regional_scales(
            np.ones((1, 1, 1)),
            np.ones((1, 1, 1)),
            np.ones((1, 1, 1)),
            window_scale=0.0,
            pose_segment_ids=frozenset(),
        )


def test_decomposition_requires_aligned_maps():
    with pytest.raises(ValueError, match="share shape"):
        decompose_regional_scales(
            np.ones((1, 1, 2)),
            np.ones((1, 1, 1)),
            np.ones((1, 1, 2)),
            window_scale=1.0,
            pose_segment_ids=frozenset(),
        )
