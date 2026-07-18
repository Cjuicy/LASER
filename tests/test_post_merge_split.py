from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine.utils import post_merge_split as pms


def _fixture(height=24, width=32):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    labels = np.zeros((height, width), dtype=np.intp)
    atoms = np.zeros_like(labels)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    return points, labels, atoms, rgb


def _run(points, labels, atoms, rgb, **kwargs):
    return pms.refine_auto_regions(
        points,
        rgb,
        labels,
        atoms,
        np.ones(int(atoms.max()) + 1, dtype=np.float64),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
        **kwargs,
    )


def test_normal_edges_ignore_normal_sign_flips():
    normals = np.zeros((6, 8, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    normals[:, 4:, 2] = -1.0
    valid = np.ones((6, 8), dtype=bool)

    edge = pms._normal_edge_map(normals, valid)

    assert np.max(edge) == 0.0


def test_normal_dispersion_is_equal_weight_and_sign_invariant():
    normals = np.asarray(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]],
        dtype=np.float64,
    )
    mask = np.ones((1, 2), dtype=bool)

    assert pms._normal_dispersion(normals, mask) == pytest.approx(0.0)


def test_texture_only_plane_is_not_split(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    rgb[:, 16:] = 1.0
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    refined, stats = _run(points, labels, atoms, rgb)

    assert np.unique(refined).size == 1
    assert stats.split_accepted_count == 0


def test_gradual_twenty_degree_turn_is_not_split(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    angles = np.deg2rad(np.linspace(0.0, 20.0, labels.shape[1]))
    normals = np.zeros_like(points)
    normals[..., 0] = np.sin(angles)[None, :]
    normals[..., 2] = np.cos(angles)[None, :]
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    refined, _ = _run(
        points,
        labels,
        atoms,
        rgb,
        split_aux_confirmation=False,
    )

    assert np.unique(refined).size == 1


def test_one_pass_can_create_two_children(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    rgb[:, 16:] = 1.0
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    refined, stats = _run(points, labels, atoms, rgb)

    assert np.unique(refined).size == 2
    assert stats.split_accepted_count == 1
    assert stats.split_added_regions == 1


def test_trace_records_accepted_parent_evidence(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    rgb[:, 16:] = 1.0
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    trace = pms.refine_auto_regions_with_trace(
        points,
        rgb,
        labels,
        atoms,
        np.ones(1),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )

    assert trace.labels.shape == labels.shape
    assert trace.changed_mask.dtype == np.bool_
    assert trace.parent_map.shape == labels.shape
    assert trace.child_map.shape == labels.shape
    assert trace.score_map.shape == labels.shape
    assert trace.decision_map.shape == labels.shape
    assert trace.diagnostics.split_accepted_count == 1
    assert np.all(trace.decision_map == 1)
    assert np.all(trace.parent_map == 0)
    assert set(np.unique(trace.child_map)) == {1, 2}
    assert np.all(trace.score_map >= 0.10)
    assert np.any(trace.changed_mask)
    evidence = trace.diagnostics.as_dict()
    assert evidence["split_score_mean"] is not None
    assert evidence["split_score_max"] is not None
    assert evidence["split_score_quantiles"] == {
        "p10": pytest.approx(evidence["split_score_mean"]),
        "p25": pytest.approx(evidence["split_score_mean"]),
        "p50": pytest.approx(evidence["split_score_mean"]),
        "p75": pytest.approx(evidence["split_score_mean"]),
        "p90": pytest.approx(evidence["split_score_mean"]),
    }
    assert evidence["split_score_invalid_reason"] is None
    assert evidence["split_pre_segment_count"] == 1
    assert evidence["split_post_segment_count"] == 2
    assert evidence["split_segment_count_delta"] == 1
    assert evidence["split_changed_pixel_count"] > 0
    assert evidence["split_changed_pixel_ratio"] > 0
    assert evidence["split_child_count_mean"] == 2
    assert evidence["split_min_child_fraction_min"] == pytest.approx(0.5)
    assert evidence["split_parent_normal_dispersion_mean"] is not None
    assert evidence["split_child_normal_dispersion_mean"] is not None
    assert evidence["split_normal_dispersion_gain_mean"] > 0
    assert evidence["split_largest_segment_ratio_delta"] < 0
    assert evidence["split_area_entropy_delta"] > 0
    assert evidence["split_effective_segment_count_delta"] > 0
    assert evidence["split_boundary_ratio_delta"] > 0
    assert evidence["split_fragmentation_signal"] in (True, False)


def test_trace_records_rejected_parent_evidence(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    trace = pms.refine_auto_regions_with_trace(
        points,
        rgb,
        labels,
        atoms,
        np.ones(1),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )

    assert trace.diagnostics.split_reject_low_score == 1
    assert np.all(trace.decision_map == 4)
    assert not np.any(trace.changed_mask)
    assert np.all(trace.score_map == 0.0)
    assert set(np.unique(trace.child_map)) == {1, 2}
    evidence = trace.diagnostics.as_dict()
    assert evidence["split_score_mean"] == 0.0
    assert evidence["split_score_invalid_reason"] is None
    assert evidence["split_pre_segment_count"] == evidence["split_post_segment_count"] == 1
    assert evidence["split_segment_count_delta"] == 0
    assert evidence["split_changed_pixel_count"] == 0
    assert evidence["split_changed_pixel_ratio"] == 0.0
    assert evidence["split_child_count_mean"] == 2
    assert evidence["split_min_child_fraction_min"] == pytest.approx(0.5)


def test_trace_records_no_marker_rejection(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    trace = pms.refine_auto_regions_with_trace(
        points,
        rgb,
        labels,
        atoms,
        np.ones(1),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )

    assert trace.diagnostics.split_reject_no_markers == 1
    assert np.all(trace.decision_map == 2)
    assert np.all(trace.child_map == -1)
    assert np.all(np.isnan(trace.score_map))
    evidence = trace.diagnostics.as_dict()
    assert evidence["split_score_mean"] is None
    assert evidence["split_score_max"] is None
    assert evidence["split_score_quantiles"] is None
    assert evidence["split_score_invalid_reason"] == "no_scored_candidates"
    assert evidence["split_child_count_mean"] is None
    assert evidence["split_child_summary_invalid_reason"] == "no_valid_child_proposals"


def test_non_finite_split_score_is_rejected_and_reported_as_missing(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )
    monkeypatch.setattr(pms, "_normal_gain", lambda *args, **kwargs: np.nan)

    trace = pms.refine_auto_regions_with_trace(
        points,
        rgb,
        labels,
        atoms,
        np.ones(1),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )

    evidence = trace.diagnostics.as_dict()
    assert trace.diagnostics.split_accepted_count == 0
    assert trace.diagnostics.split_reject_low_score == 1
    assert evidence["split_score_mean"] is None
    assert evidence["split_score_max"] is None
    assert evidence["split_score_quantiles"] is None
    assert evidence["split_score_invalid_reason"] == "non_finite_split_score"
    assert evidence["split_normal_dispersion_invalid_reason"] == "non_finite_normal_dispersion"
    assert not np.any(trace.changed_mask)


def test_trace_records_small_child_rejection(monkeypatch):
    points, labels, atoms, rgb = _fixture(height=10, width=10)
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    markers = np.zeros_like(labels, dtype=np.int32)
    markers[:, :5] = 1
    markers[:, 5:] = 2
    candidate = np.ones_like(labels, dtype=np.int32)
    candidate[:, -1] = 2
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )
    monkeypatch.setattr(pms, "_markers", lambda *args: (markers, 2))
    monkeypatch.setattr(pms, "watershed", lambda *args, **kwargs: candidate)

    trace = pms.refine_auto_regions_with_trace(
        points,
        rgb,
        labels,
        atoms,
        np.ones(1),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )

    assert trace.diagnostics.split_reject_small_child == 1
    assert np.all(trace.decision_map == 3)
    assert set(np.unique(trace.child_map)) == {1, 2}
    assert np.all(np.isnan(trace.score_map))


def test_trace_marks_ineligible_parent_evidence_as_missing(monkeypatch):
    points, labels, atoms, rgb = _fixture(height=4, width=9)
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    trace = pms.refine_auto_regions_with_trace(
        points,
        rgb,
        labels,
        atoms,
        np.ones(1),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )

    assert np.all(trace.decision_map == 0)
    assert np.all(trace.child_map == -1)
    assert np.all(np.isnan(trace.score_map))


def test_one_pass_never_exceeds_four_children(monkeypatch):
    points, labels, atoms, rgb = _fixture(height=30, width=40)
    normals = np.zeros_like(points)
    stripe_normals = np.eye(3, dtype=np.float32)[[0, 1, 2, 0, 1]]
    for stripe, normal in enumerate(stripe_normals):
        start = stripe * 8
        normals[:, start : start + 8] = normal
        rgb[:, start : start + 8] = stripe / 4.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    refined, _ = _run(points, labels, atoms, rgb)

    assert 2 <= np.unique(refined).size <= 4


def test_gap_can_confirm_without_rgb_and_is_scale_invariant(monkeypatch):
    points, labels, atoms, _ = _fixture()
    atoms[:, 16:] = 1
    points[:, 16:, 0] += 8.0
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    refined_a, _ = pms.refine_auto_regions(
        points,
        None,
        labels,
        atoms,
        np.asarray([1.0, 1.0]),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )
    refined_b, _ = pms.refine_auto_regions(
        points * 7.0,
        None,
        labels,
        atoms,
        np.asarray([7.0, 7.0]),
        seg_min_size=20,
        normal_method="cross",
        split_score_thresh=0.10,
    )

    assert np.unique(refined_a).size == 2
    np.testing.assert_array_equal(refined_a, refined_b)


def test_auxiliary_switch_changes_only_acceptance_score(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    with_aux, with_stats = _run(points, labels, atoms, rgb)
    without_aux, without_stats = _run(
        points,
        labels,
        atoms,
        rgb,
        split_aux_confirmation=False,
    )

    assert np.unique(with_aux).size == 1
    assert np.unique(without_aux).size == 2
    assert with_stats.split_aux_confirmation is True
    assert without_stats.split_aux_confirmation is False


def test_auxiliary_off_skips_rgb_and_gap_computation(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )
    monkeypatch.setattr(
        pms,
        "_edge_fields",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")),
    )

    refined, _ = _run(
        points,
        labels,
        atoms,
        rgb,
        split_aux_confirmation=False,
    )

    assert np.unique(refined).size == 2


def test_small_child_rejects_entire_parent_split(monkeypatch):
    points, labels, atoms, rgb = _fixture(height=9)
    rgb[:, -2:] = 1.0
    normals = np.zeros_like(points)
    normals[:, :-2, 2] = 1.0
    normals[:, -2:, 0] = 1.0
    monkeypatch.setattr(
        pms,
        "_normal_map",
        lambda point_map, method: (normals, np.ones(labels.shape, bool)),
    )

    refined, stats = _run(points, labels, atoms, rgb)

    assert np.unique(refined).size == 1
    assert stats.split_reject_no_markers == 1


def test_invalid_points_still_receive_compact_deterministic_labels(monkeypatch):
    points, labels, atoms, rgb = _fixture()
    points[4:8, 15:17] = np.nan
    rgb[:, 16:] = 1.0
    normals = np.zeros_like(points)
    normals[:, :16, 2] = 1.0
    normals[:, 16:, 0] = 1.0
    valid = np.isfinite(points).all(axis=-1)
    monkeypatch.setattr(pms, "_normal_map", lambda point_map, method: (normals, valid))

    first, _ = _run(points, labels, atoms, rgb)
    second, _ = _run(points, labels, atoms, rgb)

    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(np.unique(first), np.arange(np.unique(first).size))
