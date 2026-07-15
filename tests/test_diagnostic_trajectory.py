import numpy as np

from inference_engine.diagnostics.metrics import (
    build_sequence_summary,
    evaluate_stability_guard,
    recovery_score,
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


def test_stability_guard_and_recovery_metrics():
    baseline = {seq: value for seq, value in zip([f"{i:02d}" for i in range(11)], range(10, 21))}
    candidate = {seq: value for seq, value in baseline.items()}
    guard = evaluate_stability_guard(candidate, baseline)
    assert guard["passed"] is True
    candidate["00"] *= 1.11
    assert evaluate_stability_guard(candidate, baseline)["passed"] is False
    missing = dict(baseline); missing.pop("08")
    expected_guard = evaluate_stability_guard(
        missing, missing, expected_sequences=tuple(baseline)
    )
    assert expected_guard["passed"] is False
    assert "sequence_08_missing" in expected_guard["failure_reasons"]

    assert recovery_score(87.0, 78.0, 69.0)["score"] == 0.5
    assert recovery_score(70.0, 60.0, 75.0)["valid"] is False


def test_sequence_summary_keeps_legacy_out_of_official_ranking():
    values = {
        "depth": {"00": {"ate_rmse": 3.0, "valid": True}},
        "geometry_baseline": {"00": {"ate_rmse": 2.0, "valid": True}},
        "layer_atomic": {"00": {"ate_rmse": 1.0, "valid": True}},
        "geometry_legacy_reference": {"00": {"ate_rmse": .5, "valid": True}},
    }
    summary = build_sequence_summary(values)
    assert summary["official_ranking"]["00"][0]["config_id"] == "layer_atomic"
    assert all(row["config_id"] != "geometry_legacy_reference" for row in summary["official_ranking"]["00"])
    assert summary["legacy_reference"]["00"] == .5
