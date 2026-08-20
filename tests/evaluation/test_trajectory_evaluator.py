from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.artifacts import TrajectoryEstimate
from evaluation.trajectory.config import TrajectoryEvaluationConfig
from evaluation.trajectory.evaluator import (
    GroundTruthTrajectory,
    evaluate_trajectory,
)
from evo.core.trajectory import PoseTrajectory3D


def _poses(translations):
    poses = torch.eye(4).repeat(len(translations), 1, 1)
    poses[:, :3, 3] = torch.as_tensor(translations, dtype=torch.float32)
    return poses


def test_sim3_equivalent_trajectory_has_zero_ate_and_rpe():
    estimate = TrajectoryEstimate(
        (0, 1, 2),
        _poses([(10, 0, 0), (12, 0, 0), (14, 0, 0)]),
    )
    ground_truth = GroundTruthTrajectory(
        frame_ids=(0, 1, 2),
        camera_poses=_poses([(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    )

    result = evaluate_trajectory(
        estimate,
        ground_truth,
        TrajectoryEvaluationConfig(),
    )

    assert result.ate_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.rpe_translation_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.rpe_rotation_rmse_deg == pytest.approx(0.0, abs=1e-9)


def test_trajectory_evaluator_canonicalizes_numerically_drifted_rotations():
    truth_poses = _poses([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    estimate_poses = truth_poses.clone()
    estimate_poses[1, :3, :3] *= 1.00001
    frame_ids = (0, 1, 2)

    result = evaluate_trajectory(
        TrajectoryEstimate(frame_ids, estimate_poses),
        GroundTruthTrajectory(frame_ids, truth_poses),
        TrajectoryEvaluationConfig(),
    )

    assert result.ate_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.rpe_translation_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.rpe_rotation_rmse_deg == pytest.approx(0.0, abs=1e-9)


def test_trajectory_evaluator_associates_shared_frame_ids():
    estimate = TrajectoryEstimate(
        (1, 2, 3),
        _poses([(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    )
    ground_truth = GroundTruthTrajectory(
        frame_ids=(0, 1, 2),
        camera_poses=_poses([(-1, 0, 0), (0, 0, 0), (1, 0, 0)]),
    )

    result = evaluate_trajectory(
        estimate,
        ground_truth,
        TrajectoryEvaluationConfig(),
    )

    assert result.matched_frame_count == 2


def test_trajectory_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "ate.yaml"
    path.write_text(
        "version: 1\nalignment: sim3\nrpe_delta_frames: 1\nmodel: pi3\n",
        encoding="utf-8",
    )

    from evaluation.trajectory.config import load_trajectory_evaluation_config

    with pytest.raises(ValueError, match="unknown trajectory evaluation field"):
        load_trajectory_evaluation_config(path)


def test_new_evaluator_matches_existing_vo_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    from eval.vo_eval import eval_metrics

    truth_poses = _poses(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    )
    estimate_poses = truth_poses.clone()
    estimate_poses[:, :3, 3] = 2.0 * estimate_poses[:, :3, 3] + torch.tensor(
        [10.0, -3.0, 4.0]
    )
    frame_ids = (0, 1, 2, 3)
    result = evaluate_trajectory(
        TrajectoryEstimate(frame_ids, estimate_poses),
        GroundTruthTrajectory(frame_ids, truth_poses),
        TrajectoryEvaluationConfig(),
    )

    def evo_tuple(poses):
        trajectory = PoseTrajectory3D(
            poses_se3=poses.to(torch.float64).numpy(),
            timestamps=np.asarray(frame_ids, dtype=np.float64),
        )
        values = np.column_stack(
            (trajectory.positions_xyz, trajectory.orientations_quat_wxyz)
        )
        return values, trajectory.timestamps

    old_ate, old_rpe_translation, old_rpe_rotation = eval_metrics(
        evo_tuple(estimate_poses),
        evo_tuple(truth_poses),
        filename=str(tmp_path / "legacy.txt"),
    )

    assert result.ate_rmse_m == pytest.approx(old_ate, abs=1e-9)
    assert result.rpe_translation_rmse_m == pytest.approx(
        old_rpe_translation,
        abs=1e-9,
    )
    assert result.rpe_rotation_rmse_deg == pytest.approx(
        old_rpe_rotation,
        abs=1e-9,
    )
