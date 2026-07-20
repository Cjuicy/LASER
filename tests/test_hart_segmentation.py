from types import SimpleNamespace

import numpy as np
import pytest

from inference_engine.anchor_propagation import segmentation
from inference_engine.anchor_propagation.segmentation import (
    compact_intersection_labels,
    compact_labels,
    label_parent_lookup,
    validate_segmentation_frame,
)
from inference_engine.anchor_propagation.types import SegmentationFrame


def test_anchor_cells_split_an_initial_atom_at_leaf_boundaries():
    initial = np.zeros((2, 4), dtype=np.intp)
    leaves = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.intp)

    anchors = compact_intersection_labels(initial, leaves)

    assert np.unique(anchors).size == 2
    np.testing.assert_array_equal(label_parent_lookup(anchors, leaves), [0, 1])


def test_segmentation_frame_validation_rejects_crossing_hierarchy():
    leaf = np.asarray([[0, 0], [1, 1]], dtype=np.intp)
    frame = SegmentationFrame(
        leaf_labels=leaf,
        parent_labels=np.zeros_like(leaf),
        anchor_labels=np.asarray([[0, 1], [0, 1]], dtype=np.intp),
        leaf_to_parent=np.asarray([0, 0]),
        anchor_to_leaf=np.asarray([0, 0]),
    )

    with pytest.raises(ValueError, match="crosses"):
        validate_segmentation_frame(frame)


@pytest.mark.parametrize(
    ("mode", "expected_leaf", "expected_parent"),
    [
        ("depth", [[0, 0, 1, 1]], [[0, 0, 1, 1]]),
        ("geometry", [[0, 0, 1, 1]], [[0, 0, 1, 1]]),
        ("layer_atomic", [[0, 0, 1, 1]], [[0, 0, 1, 1]]),
        ("layer_atomic_split", [[0, 1, 2, 2]], [[0, 0, 1, 1]]),
    ],
)
def test_four_modes_build_the_same_strict_hierarchy(
    monkeypatch, mode, expected_leaf, expected_parent
):
    initial = np.asarray([[3, 3, 8, 8]], dtype=np.intp)
    merged = np.asarray([[7, 7, 9, 9]], dtype=np.intp)
    refined = np.asarray([[4, 5, 6, 6]], dtype=np.intp)
    point_maps = np.ones((1, 1, 4, 3), dtype=np.float32)

    monkeypatch.setattr(
        segmentation,
        "segment_depth_felzenszwalb_rag_stages",
        lambda *args, **kwargs: (initial, merged, 0.1),
    )
    geometry_stages = SimpleNamespace(
        initial_labels=initial, merged_labels=merged
    )
    monkeypatch.setattr(
        segmentation,
        "segment_geometry_felzenszwalb_rag_stages",
        lambda *args, **kwargs: geometry_stages,
    )
    monkeypatch.setattr(
        segmentation,
        "segment_geometry_felzenszwalb_rag_baseline_params_stages",
        lambda *args, **kwargs: geometry_stages,
    )
    atomic_stages = SimpleNamespace(
        initial_labels=initial,
        merged_labels=merged,
        refined_labels=refined if mode == "layer_atomic_split" else merged,
        split_diagnostics=(
            SimpleNamespace(as_dict=lambda: {"accepted": 1})
            if mode == "layer_atomic_split"
            else None
        ),
    )
    monkeypatch.setattr(
        segmentation,
        "segment_point_map_layer_atomic_stages",
        lambda *args, **kwargs: atomic_stages,
    )

    result = segmentation.build_segmentation_window(
        point_maps, segment_mode=mode, n_jobs=1
    )
    frame = result.frames[0]

    np.testing.assert_array_equal(frame.leaf_labels, expected_leaf)
    np.testing.assert_array_equal(frame.parent_labels, expected_parent)
    validate_segmentation_frame(frame)
    if mode == "layer_atomic_split":
        assert frame.split_diagnostics == {"accepted": 1}


def test_compact_labels_rejects_non_image_input():
    with pytest.raises(ValueError, match="shape"):
        compact_labels(np.zeros((2, 2, 2)))


def test_empty_segmentation_window_has_explicit_shape():
    result = segmentation.build_segmentation_window(
        np.empty((0, 2, 3, 3), dtype=np.float32), n_jobs=1
    )
    assert result.frames == ()
    assert result.shape == (0, 0, 0)
