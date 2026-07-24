import numpy as np
import pytest

from inference_engine.utils import post_merge_split
from pipeline.config import AtomicSplitMode


def split_fixture(height=24, width=32):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    labels = np.zeros((height, width), dtype=np.intp)
    atoms = np.zeros_like(labels)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    return points, labels, atoms, rgb


def _two_normal_fixture(monkeypatch):
    points, labels, atoms, rgb = split_fixture()
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        post_merge_split,
        "_normal_map",
        lambda point_map, method: (
            normals,
            np.ones(labels.shape, dtype=bool),
        ),
    )
    return points, labels, atoms, rgb


def _run(points, labels, atoms, rgb, mode, **kwargs):
    return post_merge_split.refine_auto_regions(
        points,
        rgb,
        labels,
        atoms,
        np.ones(int(atoms.max()) + 1, dtype=np.float64),
        seg_min_size=20,
        normal_method="cross",
        split_score_threshold=0.10,
        split_mode=mode,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("mode", "expected_regions"),
    (
        (AtomicSplitMode.NONE, 1),
        (AtomicSplitMode.CONSERVATIVE, 1),
        (AtomicSplitMode.NORMAL_ONLY, 2),
    ),
)
def test_atomic_split_modes_are_mutually_exclusive(
    monkeypatch,
    mode,
    expected_regions,
):
    points, labels, atoms, rgb = _two_normal_fixture(monkeypatch)
    refined, diagnostics = _run(
        points,
        labels,
        atoms,
        rgb,
        mode,
    )
    assert np.unique(refined).size == expected_regions
    assert diagnostics.split_mode == mode.value


def test_none_mode_skips_all_split_computation(monkeypatch):
    points, labels, atoms, rgb = split_fixture()
    labels[:, 16:] = 7
    monkeypatch.setattr(
        post_merge_split,
        "_normal_map",
        lambda *args: (_ for _ in ()).throw(AssertionError("called")),
    )
    refined, diagnostics = _run(
        points,
        labels,
        atoms,
        rgb,
        AtomicSplitMode.NONE,
    )
    np.testing.assert_array_equal(np.unique(refined), [0, 1])
    assert diagnostics.split_parent_count == 0


def test_rgb_can_confirm_conservative_split(monkeypatch):
    points, labels, atoms, rgb = _two_normal_fixture(monkeypatch)
    rgb[:, 16:] = 1.0
    refined, diagnostics = _run(
        points,
        labels,
        atoms,
        rgb,
        AtomicSplitMode.CONSERVATIVE,
    )
    assert np.unique(refined).size == 2
    assert diagnostics.split_accepted_count == 1


def test_normalized_gap_is_scale_invariant(monkeypatch):
    points, labels, atoms, _ = _two_normal_fixture(monkeypatch)
    atoms[:, 16:] = 1
    points[:, 16:, 0] += 8.0

    first, _ = post_merge_split.refine_auto_regions(
        points,
        None,
        labels,
        atoms,
        np.asarray([1.0, 1.0]),
        seg_min_size=20,
        normal_method="cross",
        split_score_threshold=0.10,
        split_mode=AtomicSplitMode.CONSERVATIVE,
    )
    second, _ = post_merge_split.refine_auto_regions(
        points * 7.0,
        None,
        labels,
        atoms,
        np.asarray([7.0, 7.0]),
        seg_min_size=20,
        normal_method="cross",
        split_score_threshold=0.10,
        split_mode=AtomicSplitMode.CONSERVATIVE,
    )
    assert np.unique(first).size == 2
    np.testing.assert_array_equal(first, second)


def test_normal_only_skips_rgb_and_gap_computation(monkeypatch):
    points, labels, atoms, rgb = _two_normal_fixture(monkeypatch)
    monkeypatch.setattr(
        post_merge_split,
        "_edge_fields",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("called")),
    )
    refined, _ = _run(
        points,
        labels,
        atoms,
        rgb,
        AtomicSplitMode.NORMAL_ONLY,
    )
    assert np.unique(refined).size == 2


def test_one_pass_never_exceeds_four_leaves(monkeypatch):
    points, labels, atoms, rgb = split_fixture(height=30, width=40)
    normals = np.zeros_like(points)
    stripe_normals = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1]]
    for stripe, normal in enumerate(stripe_normals):
        start = stripe * 8
        normals[:, start : start + 8] = normal
    monkeypatch.setattr(
        post_merge_split,
        "_normal_map",
        lambda point_map, method: (
            normals,
            np.ones(labels.shape, dtype=bool),
        ),
    )
    refined, _ = _run(
        points,
        labels,
        atoms,
        rgb,
        AtomicSplitMode.NORMAL_ONLY,
    )
    assert 2 <= np.unique(refined).size <= 4


def test_small_child_rejects_entire_parent_split(monkeypatch):
    points, labels, atoms, rgb = split_fixture(height=9)
    normals = np.zeros_like(points)
    normals[:, :-2, 2] = 1.0
    normals[:, -2:, 0] = 1.0
    monkeypatch.setattr(
        post_merge_split,
        "_normal_map",
        lambda point_map, method: (
            normals,
            np.ones(labels.shape, dtype=bool),
        ),
    )
    refined, diagnostics = _run(
        points,
        labels,
        atoms,
        rgb,
        AtomicSplitMode.NORMAL_ONLY,
    )
    assert np.unique(refined).size == 1
    assert (
        diagnostics.split_reject_no_markers
        + diagnostics.split_reject_small_child
        >= 1
    )


def test_invalid_points_keep_full_compact_deterministic_coverage(monkeypatch):
    points, labels, atoms, rgb = split_fixture()
    points[4:8, 15:17] = np.nan
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    valid = np.isfinite(points).all(axis=-1)
    monkeypatch.setattr(
        post_merge_split,
        "_normal_map",
        lambda point_map, method: (normals, valid),
    )

    first, _ = _run(
        points,
        labels,
        atoms,
        rgb,
        AtomicSplitMode.NORMAL_ONLY,
    )
    second, _ = _run(
        points,
        labels,
        atoms,
        rgb,
        AtomicSplitMode.NORMAL_ONLY,
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == labels.shape
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(
        np.unique(first),
        np.arange(np.unique(first).size),
    )
