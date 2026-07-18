import copy

import numpy as np
import pytest

from inference_engine.diagnostics.selection import (
    REQUIRED_COVERAGE_REASONS,
    robust_zscore,
    select_intervals,
    select_intervals_with_coverage,
)


def _records():
    records = []
    for sequence in ("00", "02", "04", "05", "09", "10"):
        regrets = [
            (-4.0, -2.0),
            (0.0, 0.0),
            (5.0, 3.0),
            (5.2, 3.1),
            (0.1, 0.1),
            (0.2, 0.2),
            (1.0, 0.8),
            (0.3, 0.3),
        ]
        split_activity = [
            (0, 0.0),
            (0, 0.0),
            (1, 0.15),
            (0, 0.0),
            (2, 0.35),
            (0, 0.0),
            (10, 0.90),
            (0, 0.0),
        ]
        for window, ((depth_regret, geometry_regret), (accepted, changed)) in enumerate(
            zip(regrets, split_activity, strict=True)
        ):
            common = {
                "sequence_id": sequence,
                "window_id": window,
                "frame_start": window * 5,
                "frame_end": window * 5 + 9,
                "split_minus_depth_regret": depth_regret,
                "split_minus_geometry_regret": geometry_regret,
                "merge_anomaly": 0.1 * window,
                "atom_anomaly": 0.08 * window,
                "scale_dispersion": 0.06 * window,
                "temporal_churn": 0.04 * window,
                "gt_speed": window + 1,
                "gt_turn": window * 0.1,
                "confidence": 0.9 - window * 0.01,
            }
            for config in ("depth", "geometry_baseline", "layer_atomic_split"):
                # Split evidence from baseline rows must never influence selection.
                is_split = config == "layer_atomic_split"
                records.append({
                    **common,
                    "config_id": config,
                    "split_accepted_count": accepted if is_split else 100,
                    "split_changed_pixel_ratio": changed if is_split else 1.0,
                })
    return records


def test_robust_zscore_is_finite_for_constant_and_outlier_inputs():
    np.testing.assert_array_equal(robust_zscore([1, 1, 1]), np.zeros(3))
    values = robust_zscore([0, 0, 10])
    assert np.isfinite(values).all()
    assert values[-1] > values[0]


def test_selector_rejects_zero_limit_and_matches_controls_with_no_accepted_splits():
    with pytest.raises(ValueError, match="positive"):
        select_intervals(_records(), limit=0)

    records = _records()
    selected = select_intervals(records, limit=48)
    for item in selected:
        if "matched_control" not in item.reasons:
            continue
        split_rows = [
            row for row in records
            if row["config_id"] == "layer_atomic_split"
            and row["sequence_id"] == item.sequence_id
            and row["frame_start"] == item.start_frame
            and row["frame_end"] == item.end_frame
        ]
        assert split_rows
        assert split_rows[0]["split_accepted_count"] == 0


def test_selector_emits_all_split_trajectory_reasons_deterministically():
    records = _records()
    first = select_intervals(records, limit=48, context_windows=2)
    second = select_intervals(copy.deepcopy(records), limit=48, context_windows=2)

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) <= 48
    assert {item.sequence_id for item in first} >= {"00", "02", "04", "05", "09", "10"}
    reasons = {reason for item in first for reason in item.reasons}
    assert {
        "trajectory_degradation",
        "trajectory_improvement",
        "trajectory_change",
        "split_anomaly",
        "split_no_trajectory_effect",
        "matched_control",
    } <= reasons
    exact = select_intervals(records, limit=48, context_windows=0)
    expected_starts = {
        "trajectory_improvement": 0,
        "trajectory_change": 20,
        "trajectory_degradation": 15,
        "split_no_trajectory_effect": 20,
        "split_anomaly": 30,
        "matched_control": 25,
    }
    for reason, start in expected_starts.items():
        assert any(
            item.sequence_id == "02"
            and item.start_frame == start
            and reason in item.reasons
            for item in exact
        )
    assert all(item.reasons for item in first)
    assert all(
        item.end_frame - item.start_frame + 1 <= 30
        for item in first if "matched_control" not in item.reasons
    )
    _, coverage = select_intervals_with_coverage(
        records, limit=48, context_windows=2
    )
    assert set(coverage) == set(REQUIRED_COVERAGE_REASONS)
    assert all(item["available"] is True for item in coverage.values())
    assert all(item["selected"] is True for item in coverage.values())
    assert all(item["reason"] is None for item in coverage.values())


def test_selector_preserves_nontrajectory_family_diversity_and_context_merging():
    selected = select_intervals(_records(), limit=48)
    reasons = {reason for item in selected for reason in item.reasons}
    assert {"merge", "immutable_atom", "scale", "temporal"} <= reasons
    assert any(len(item.reasons) > 1 for item in selected)
    assert len(selected) <= 48


def test_selector_does_not_treat_missing_split_count_as_a_matched_control():
    records = []
    for window, (accepted, changed, speed) in enumerate([
        (2, 0.5, 1.0),
        (None, None, 1.01),
        (0, 0.0, 3.0),
    ]):
        records.append({
            "config_id": "layer_atomic_split",
            "sequence_id": "01",
            "frame_start": window * 10,
            "frame_end": window * 10 + 9,
            "split_minus_depth_regret": 2.0 if window == 0 else 0.0,
            "split_minus_geometry_regret": 1.0 if window == 0 else 0.0,
            "split_accepted_count": accepted,
            "split_changed_pixel_ratio": changed,
            "merge_anomaly": 0.0,
            "atom_anomaly": 0.0,
            "scale_dispersion": 0.0,
            "temporal_churn": 0.0,
            "gt_speed": speed,
            "gt_turn": 0.0,
            "confidence": 0.8,
        })

    selected = select_intervals(records, context_windows=0)

    controls = [item for item in selected if "matched_control" in item.reasons]
    assert len(controls) == 1
    assert controls[0].start_frame == 20


def test_selector_does_not_infer_trajectory_effects_from_all_missing_regrets():
    records = []
    for window, accepted in enumerate((2, 0, 0)):
        records.append({
            "config_id": "layer_atomic_split",
            "sequence_id": "01",
            "frame_start": window * 10,
            "frame_end": window * 10 + 9,
            "split_minus_depth_regret": None,
            "split_minus_geometry_regret": None,
            "split_accepted_count": accepted,
            "split_changed_pixel_ratio": 0.5 if accepted else 0.0,
            "merge_anomaly": 0.0,
            "atom_anomaly": 0.0,
            "scale_dispersion": 0.0,
            "temporal_churn": 0.0,
            "gt_speed": float(window),
            "gt_turn": 0.0,
            "confidence": 0.8,
        })

    selected = select_intervals(records, context_windows=0)
    reasons = {reason for item in selected for reason in item.reasons}

    assert "split_anomaly" in reasons
    assert "matched_control" in reasons
    assert not {
        "trajectory_degradation",
        "trajectory_improvement",
        "trajectory_change",
        "split_no_trajectory_effect",
    } & reasons


def test_selector_change_uses_only_adjacent_comparable_finite_regrets():
    records = []
    for window, (depth_regret, geometry_regret) in enumerate([
        (0.0, None),
        (1.0, None),
        (None, 100.0),
    ]):
        records.append({
            "config_id": "layer_atomic_split",
            "sequence_id": "01",
            "frame_start": window * 10,
            "frame_end": window * 10 + 9,
            "split_minus_depth_regret": depth_regret,
            "split_minus_geometry_regret": geometry_regret,
            "split_accepted_count": 0,
            "split_changed_pixel_ratio": 0.0,
            "merge_anomaly": 0.0,
            "atom_anomaly": 0.0,
            "scale_dispersion": 0.0,
            "temporal_churn": 0.0,
            "gt_speed": float(window),
            "gt_turn": 0.0,
            "confidence": 0.8,
        })

    selected = select_intervals(records, context_windows=0)
    changes = [item for item in selected if "trajectory_change" in item.reasons]

    assert len(changes) == 1
    assert changes[0].start_frame == 10


def test_selector_does_not_create_matched_control_without_split_treatment():
    records = []
    for window in range(3):
        records.append({
            "config_id": "layer_atomic_split",
            "sequence_id": "01",
            "frame_start": window * 10,
            "frame_end": window * 10 + 9,
            "split_minus_depth_regret": float(window),
            "split_minus_geometry_regret": float(window),
            "split_accepted_count": 0,
            "split_changed_pixel_ratio": 0.0,
            "merge_anomaly": 0.0,
            "atom_anomaly": 0.0,
            "scale_dispersion": 0.0,
            "temporal_churn": 0.0,
            "gt_speed": float(window),
            "gt_turn": 0.0,
            "confidence": 0.8,
        })

    selected = select_intervals(records, context_windows=0)

    assert all("matched_control" not in item.reasons for item in selected)


def test_zero_delta_and_zero_zscore_do_not_claim_change_or_split_anomaly():
    records = []
    for window in range(3):
        records.append({
            "config_id": "layer_atomic_split",
            "sequence_id": "01",
            "frame_start": window * 10,
            "frame_end": window * 10 + 9,
            "split_minus_depth_regret": 1.0,
            "split_minus_geometry_regret": 1.0,
            "split_accepted_count": 1,
            "split_changed_pixel_ratio": 0.1,
            "merge_anomaly": 0.0,
            "atom_anomaly": 0.0,
            "scale_dispersion": 0.0,
            "temporal_churn": 0.0,
            "gt_speed": 1.0,
            "gt_turn": 0.0,
            "confidence": 0.8,
        })

    selected, coverage = select_intervals_with_coverage(
        records, context_windows=0
    )
    reasons = {reason for item in selected for reason in item.reasons}

    assert "trajectory_change" not in reasons
    assert "split_anomaly" not in reasons
    assert coverage["trajectory_change"] == {
        "available": False,
        "selected": False,
        "reason": "no_qualifying_window",
        "qualifying_window_count": 0,
    }
    assert coverage["split_anomaly"] == {
        "available": False,
        "selected": False,
        "reason": "no_qualifying_window",
        "qualifying_window_count": 0,
    }


def test_empty_evidence_has_explicit_unavailable_coverage_for_all_six_reasons():
    selected, coverage = select_intervals_with_coverage([], limit=48)

    assert selected == []
    assert list(coverage) == list(REQUIRED_COVERAGE_REASONS)
    assert all(item == {
        "available": False,
        "selected": False,
        "reason": "no_qualifying_window",
        "qualifying_window_count": 0,
    } for item in coverage.values())
