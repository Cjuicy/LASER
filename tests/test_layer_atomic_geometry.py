from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
inference_engine_module = sys.modules.get("inference_engine")
if inference_engine_module is not None and not hasattr(inference_engine_module, "__path__"):
    del sys.modules["inference_engine"]

from inference_engine.utils.layer_atomic_geometry import (
    merge_layer_atoms,
    segment_point_map_layer_atomic,
)


def _two_atom_map(top_row, scale=1.0):
    top = np.asarray(top_row, dtype=np.float64) * scale
    bottom = top + np.asarray([0.0, 1.0, 0.0]) * scale
    return np.stack([top, bottom])


def _two_atom_labels(*, same_coarse=True):
    initial = np.tile(np.asarray([10, 10, 30, 30]), (2, 1))
    if same_coarse:
        coarse = np.zeros_like(initial)
    else:
        coarse = np.tile(np.asarray([7, 7, 8, 8]), (2, 1))
    return initial, coarse


def _merge(top_row, *, same_coarse=True, scale=1.0):
    initial, coarse = _two_atom_labels(same_coarse=same_coarse)
    point_map = _two_atom_map(top_row, scale=scale)
    return merge_layer_atoms(point_map, initial, coarse, depth_merge_thresh=0.1)


def test_continuous_turn_merges_atoms_despite_surface_direction_change():
    labels = _merge(
        [
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 2.0),
        ]
    )

    np.testing.assert_array_equal(labels, np.zeros((2, 4), dtype=np.int64))


def test_real_3d_gap_splits_atoms_inside_the_same_coarse_layer():
    labels = _merge([(x, 0.0, 0.0) for x in (0.0, 1.0, 4.0, 5.0)])

    np.testing.assert_array_equal(
        labels,
        np.tile(np.asarray([0, 0, 1, 1]), (2, 1)),
    )


def test_continuous_atoms_can_remerge_across_coarse_layers():
    labels = _merge(
        [(x, 0.0, 0.0) for x in (0.0, 1.0, 2.0, 3.0)],
        same_coarse=False,
    )

    np.testing.assert_array_equal(labels, np.zeros((2, 4), dtype=np.int64))


def test_weak_layer_prior_accepts_only_same_layer_small_excess_gap():
    top_row = [(x, 0.0, 0.0) for x in (0.0, 1.0, 2.05, 3.05)]

    same_layer = _merge(top_row, same_coarse=True)
    cross_layer = _merge(top_row, same_coarse=False)

    assert np.unique(same_layer).size == 1
    assert np.unique(cross_layer).size == 2


@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 1e6])
def test_result_is_invariant_to_global_point_scale(global_scale):
    top_row = [(x, 0.0, 0.0) for x in (0.0, 1.0, 2.05, 3.05)]

    labels = _merge(top_row, same_coarse=True, scale=global_scale)

    np.testing.assert_array_equal(labels, np.zeros((2, 4), dtype=np.int64))


@pytest.mark.parametrize("global_scale", [1e-6, 1.0, 1e6])
def test_degenerate_atoms_do_not_merge_or_break_scale_invariance(global_scale):
    point_map = _two_atom_map(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ],
        scale=global_scale,
    )
    point_map[1, :2] = point_map[0, :2]
    initial, coarse = _two_atom_labels(same_coarse=True)

    labels = merge_layer_atoms(
        point_map,
        initial,
        coarse,
        depth_merge_thresh=0.1,
    )

    np.testing.assert_array_equal(
        labels,
        np.tile(np.asarray([0, 0, 1, 1]), (2, 1)),
    )


def test_result_is_compact_full_coverage_and_only_unions_atoms():
    top = np.asarray([(x, 0.0, 0.0) for x in (0, 1, 2, 3, 6, 7)], dtype=float)
    point_map = np.stack([top, top + (0.0, 1.0, 0.0)])
    initial = np.tile(np.asarray([10, 10, 30, 30, 90, 90]), (2, 1))
    coarse = np.zeros_like(initial)

    labels = merge_layer_atoms(point_map, initial, coarse, depth_merge_thresh=0.1)

    assert labels.shape == initial.shape
    assert np.all(labels >= 0)
    np.testing.assert_array_equal(np.unique(labels), np.arange(2))
    for atom in np.unique(initial):
        assert np.unique(labels[initial == atom]).size == 1
    np.testing.assert_array_equal(
        labels,
        np.tile(np.asarray([0, 0, 0, 0, 1, 1]), (2, 1)),
    )


def test_result_is_deterministic():
    top_row = [(x, 0.0, 0.0) for x in (0.0, 1.0, 2.0, 3.0)]

    results = [_merge(top_row, same_coarse=False) for _ in range(5)]

    for labels in results[1:]:
        np.testing.assert_array_equal(labels, results[0])


@pytest.mark.parametrize(
    "shape",
    [(2, 4), (2, 4, 2), (2, 4, 3, 1)],
)
def test_segment_point_map_layer_atomic_validates_shape(shape):
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        segment_point_map_layer_atomic(np.zeros(shape), depth_merge_thresh=0.1)
