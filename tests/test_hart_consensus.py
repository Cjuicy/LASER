import numpy as np

from inference_engine.anchor_propagation.consensus import (
    STATUS_CONFLICT,
    STATUS_DIRECT,
    STATUS_LEAF_FILL,
    aggregate_segment_scales,
    build_scale_mask,
    complete_link_groups,
)
from inference_engine.anchor_propagation.correspondence import build_track_window
from inference_engine.anchor_propagation.segmentation import _frame_from_layers
from inference_engine.anchor_propagation.types import DirectAnchor


def _anchor(segment, previous, scale, frame=0, anchor=0):
    return DirectAnchor(frame, anchor, segment, previous, scale, 100)


def test_temporal_evidence_is_medianed_before_provenance_conflict_check():
    anchors = (
        _anchor(4, 8, 2.00, frame=0),
        _anchor(4, 8, 2.02, frame=1),
        _anchor(4, 9, 2.01, frame=0),
    )
    scales, conflicts = aggregate_segment_scales(anchors, threshold=0.05)
    assert scales[4] == 2.01
    assert conflicts == set()


def test_conflicting_provenances_are_never_averaged():
    scales, conflicts = aggregate_segment_scales(
        (_anchor(4, 8, 1.0), _anchor(4, 9, 2.0)), threshold=0.05
    )
    assert 4 not in scales
    assert conflicts == {4}


def test_complete_link_prevents_chained_scale_cluster():
    groups = complete_link_groups([1.0, np.exp(0.04), np.exp(0.08)], 0.05)
    assert len(groups) == 2


def test_leaf_consensus_fills_missing_anchor_cell_but_conflict_stays_one():
    initial = np.asarray([[0, 0, 1, 1]])
    leaf = np.zeros_like(initial)
    frame = _frame_from_layers(initial, leaf)
    tracks = build_track_window((frame.anchor_labels,), threshold=0.3)

    mask, statuses, diagnostics = build_scale_mask(
        (frame,), tracks, {0: 2.0}, set(), threshold=0.05
    )

    np.testing.assert_array_equal(mask, [[[2.0, 2.0, 2.0, 2.0]]])
    assert np.all(statuses[0, :, :2] == STATUS_DIRECT)
    assert np.all(statuses[0, :, 2:] == STATUS_LEAF_FILL)
    assert diagnostics["regional_scale_min"] == 2.0
    assert diagnostics["regional_scale_median"] == 2.0
    assert diagnostics["regional_scale_max"] == 2.0
    assert "scale_mask_min" not in diagnostics

    conflict_mask, conflict_status, _ = build_scale_mask(
        (frame,), tracks, {}, {0}, threshold=0.05
    )
    np.testing.assert_array_equal(
        conflict_mask, [[[1.0, 1.0, 1.0, 1.0]]]
    )
    assert np.all(conflict_status[0, :, :2] == STATUS_CONFLICT)
