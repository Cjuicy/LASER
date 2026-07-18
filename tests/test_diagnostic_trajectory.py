import numpy as np
import pytest

from inference_engine.diagnostics.metrics import (
    build_sequence_summary,
    compute_regret_series,
    evaluate_stability_guard,
    recovery_score,
    summarize_regret,
)
from inference_engine.diagnostics.trajectory import evaluate_trajectory


def _poses(count=8):
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, 0, 3] = np.arange(count)
    poses[:, 1, 3] = np.sin(np.arange(count) * .3)
    for index in range(count):
        angle = index * .02
        poses[index, :3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    return poses


def _sim3(poses):
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    result = poses.copy()
    result[:, :3, 3] = 2.5 * (rotation @ poses[:, :3, 3].T).T + np.array([5, -2, 1])
    result[:, :3, :3] = rotation @ poses[:, :3, :3]
    return result


def test_one_global_sim3_alignment_recovers_known_trajectory():
    gt = _poses()
    result = evaluate_trajectory(_sim3(gt), gt)
    assert result["valid"] is True
    assert result["ate_rmse"] < 1e-10
    assert result["rpe_translation_rmse"] < 1e-10
    assert result["rpe_rotation_rmse_deg"] < 1e-8
    assert len(result["per_frame_translation_error"]) == len(gt)
    assert result["evaluation_signature"]["rpe_delta"] == 1
    assert result["evaluation_signature"]["all_pairs"] is True


def test_invalid_trajectory_returns_missing_metrics_not_zero():
    result = evaluate_trajectory(np.eye(4)[None], np.eye(4)[None])
    assert result["valid"] is False
    assert result["ate_rmse"] is None
    assert result["invalid_reason"] == "at_least_two_poses_required"


def test_stability_guard_uses_depth_as_its_baseline_and_recovery_is_depth_to_geometry():
    depth_ates = {seq: value for seq, value in zip([f"{i:02d}" for i in range(11)], range(10, 21))}
    split_ates = {seq: value for seq, value in depth_ates.items()}
    guard = evaluate_stability_guard(
        split_ates, depth_ates, expected_sequences=("00", "05", "09")
    )
    assert guard["passed"] is True
    assert guard["baseline_config"] == "depth"
    split_ates["00"] *= 1.11
    assert evaluate_stability_guard(split_ates, depth_ates)["passed"] is False
    missing = dict(depth_ates); missing.pop("08")
    expected_guard = evaluate_stability_guard(
        missing, missing, expected_sequences=tuple(depth_ates)
    )
    assert expected_guard["passed"] is False
    assert "sequence_08_missing" in expected_guard["failure_reasons"]

    assert recovery_score(87.0, 78.0, 69.0)["score"] == pytest.approx(0.5)
    assert recovery_score(70.0, 60.0, 75.0)["valid"] is False


@pytest.mark.parametrize(
    "depth,split,geometry",
    [
        (np.nan, 1.0, 0.5),
        (1.0, np.inf, 0.5),
        (1.0, 0.5, -np.inf),
    ],
)
def test_recovery_rejects_each_non_finite_ate_input(depth, split, geometry):
    result = recovery_score(depth, split, geometry)

    assert result == {
        "valid": False,
        "score": None,
        "invalid_reason": "non_finite_ate_input",
    }


def test_sequence_summary_is_strict_three_method_contract_with_recovery_scores():
    values = {
        "depth": {
            sequence: {"ate_rmse": 87.0, "valid": True}
            for sequence in ("02", "04", "10")
        },
        "geometry_baseline": {
            sequence: {"ate_rmse": 69.0, "valid": True}
            for sequence in ("02", "04", "10")
        },
        "layer_atomic_split": {
            sequence: {"ate_rmse": 78.0, "valid": True}
            for sequence in ("02", "04", "10")
        },
    }
    summary = build_sequence_summary(values)
    assert set(summary["official_aggregate"]) == {
        "depth", "geometry_baseline", "layer_atomic_split"
    }
    assert "legacy_reference" not in summary
    assert "recovery_gap" not in summary
    assert summary["recovery"]["02"]["score"] == pytest.approx(0.5)


def test_regret_summary_exports_positive_area_duration_persistence_and_change_points():
    summary = summarize_regret(np.asarray([-1.0, 2.0, 3.0, -2.0, 1.0]))

    assert summary == {
        "valid": True,
        "invalid_reason": None,
        "mean": pytest.approx(0.6),
        "max": 3.0,
        "positive_area": 6.0,
        "positive_duration": 3,
        "positive_persistence": 2,
        "change_points": [
            {"frame_index": 3, "delta": -5.0, "magnitude": 5.0},
            {"frame_index": 1, "delta": 3.0, "magnitude": 3.0},
            {"frame_index": 4, "delta": 3.0, "magnitude": 3.0},
        ],
    }


def test_regret_series_preserves_partial_and_missing_comparisons():
    partial = compute_regret_series([1.0, np.nan, 4.0], [0.5, 2.0])
    assert partial["valid"] is True
    assert partial["invalid_reason"] == "partial_missing_comparison"
    assert partial["values"] == [0.5, None, None]
    assert partial["summary"]["mean"] == 0.5
    assert partial["summary"]["positive_duration"] == 1

    missing = compute_regret_series([], [1.0, 2.0])
    assert missing["valid"] is False
    assert missing["invalid_reason"] == "missing_comparison"
    assert missing["values"] == [None, None]
    assert missing["summary"]["mean"] is None
    assert missing["summary"]["positive_duration"] is None
