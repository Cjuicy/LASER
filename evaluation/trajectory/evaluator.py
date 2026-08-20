from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import numpy as np
import torch
from evo.core.metrics import PoseRelation, Unit
from evo.core.geometry import GeometryException
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation

from pipeline.artifacts import TrajectoryEstimate

from .config import TrajectoryEvaluationConfig
from .results import TrajectoryMetrics


def _validate_poses(name: str, value: torch.Tensor, frame_count: int) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (frame_count, 4, 4)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must have finite shape (N,4,4)")


@dataclass(frozen=True)
class GroundTruthTrajectory:
    frame_ids: tuple[int, ...]
    camera_poses: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_ids, tuple)
            or len(self.frame_ids) < 2
            or len(set(self.frame_ids)) != len(self.frame_ids)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in self.frame_ids
            )
        ):
            raise ValueError("ground-truth frame_ids must be unique integers")
        _validate_poses("ground-truth camera_poses", self.camera_poses, len(self.frame_ids))


def _matched_poses(
    estimate: TrajectoryEstimate,
    ground_truth: GroundTruthTrajectory,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    estimate_lookup = {frame_id: index for index, frame_id in enumerate(estimate.frame_ids)}
    truth_lookup = {frame_id: index for index, frame_id in enumerate(ground_truth.frame_ids)}
    shared = tuple(sorted(set(estimate_lookup) & set(truth_lookup)))
    if len(shared) < 2:
        raise ValueError("trajectory evaluation requires at least two matched frames")
    estimated = torch.stack(
        [estimate.camera_poses[estimate_lookup[frame_id]] for frame_id in shared]
    )
    truth = torch.stack(
        [ground_truth.camera_poses[truth_lookup[frame_id]] for frame_id in shared]
    )
    return estimated, truth, shared


def _evo_trajectory(poses: torch.Tensor, frame_ids: tuple[int, ...]):
    matrices = poses.detach().cpu().to(torch.float64).numpy().copy()
    # Preserve the legacy matrix -> quaternion semantics before evo's strict
    # SO(3) check, removing only accumulated floating-point rotation drift.
    matrices[:, :3, :3] = Rotation.from_matrix(
        matrices[:, :3, :3]
    ).as_matrix()
    return PoseTrajectory3D(
        poses_se3=matrices,
        timestamps=np.asarray(frame_ids, dtype=np.float64),
    )


def _fallback_sim3_align(
    estimated: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    source = estimated.detach().cpu().to(torch.float64).numpy()
    target = truth.detach().cpu().to(torch.float64).numpy()
    source_xyz = source[:, :3, 3]
    target_xyz = target[:, :3, 3]
    source_center = source_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)
    source_zero = source_xyz - source_center
    target_zero = target_xyz - target_center
    covariance = target_zero.T @ source_zero / len(source_xyz)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.sum(source_zero * source_zero) / len(source_xyz))
    scale = (
        float(np.sum(singular * sign) / variance)
        if variance > np.finfo(np.float64).eps
        else 1.0
    )
    translation = target_center - scale * (rotation @ source_center)
    aligned = source.copy()
    aligned[:, :3, :3] = np.einsum(
        "ij,njk->nik",
        rotation,
        source[:, :3, :3],
    )
    aligned[:, :3, 3] = (
        scale * np.einsum("ij,nj->ni", rotation, source_xyz)
        + translation
    )
    return torch.as_tensor(aligned, dtype=torch.float64)


def evaluate_trajectory(
    estimate: TrajectoryEstimate,
    ground_truth: GroundTruthTrajectory,
    config: TrajectoryEvaluationConfig,
) -> TrajectoryMetrics:
    if not isinstance(estimate, TrajectoryEstimate):
        raise ValueError("estimate must be TrajectoryEstimate")
    if not isinstance(ground_truth, GroundTruthTrajectory):
        raise ValueError("ground_truth must be GroundTruthTrajectory")
    if not isinstance(config, TrajectoryEvaluationConfig):
        raise ValueError("config must be TrajectoryEvaluationConfig")
    estimated, truth, shared = _matched_poses(estimate, ground_truth)
    if len(shared) <= config.rpe_delta_frames:
        raise ValueError("matched trajectory is too short for configured RPE delta")
    traj_est = _evo_trajectory(estimated, shared)
    traj_ref = _evo_trajectory(truth, shared)
    common = {
        "traj_ref": traj_ref,
        "traj_est": traj_est,
        "est_name": "artifact",
        "align": True,
        "correct_scale": True,
    }
    try:
        ate = main_ape.ape(
            **common,
            pose_relation=PoseRelation.translation_part,
        )
    except GeometryException:
        traj_est = _evo_trajectory(
            _fallback_sim3_align(estimated, truth),
            shared,
        )
        common = {
            **common,
            "traj_est": traj_est,
            "align": False,
            "correct_scale": False,
        }
        ate = main_ape.ape(
            **common,
            pose_relation=PoseRelation.translation_part,
        )
    rpe_translation = main_rpe.rpe(
        **common,
        pose_relation=PoseRelation.translation_part,
        delta=config.rpe_delta_frames,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    rpe_rotation = main_rpe.rpe(
        **common,
        pose_relation=PoseRelation.rotation_angle_deg,
        delta=config.rpe_delta_frames,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    return TrajectoryMetrics(
        ate_rmse_m=float(ate.stats["rmse"]),
        rpe_translation_rmse_m=float(rpe_translation.stats["rmse"]),
        rpe_rotation_rmse_deg=float(rpe_rotation.stats["rmse"]),
        matched_frame_count=len(shared),
    )


def _poses_from_evo(trajectory) -> GroundTruthTrajectory:
    poses = torch.as_tensor(np.asarray(trajectory.poses_se3), dtype=torch.float64)
    return GroundTruthTrajectory(
        frame_ids=tuple(range(len(poses))),
        camera_poses=poses,
    )


def load_ground_truth_trajectory(
    path: str | Path,
    format_name: str,
) -> GroundTruthTrajectory:
    path = Path(path)
    if format_name in {"tum", "tartanair"}:
        return _poses_from_evo(file_interface.read_tum_trajectory_file(path))
    if format_name == "replica":
        raw = np.loadtxt(path)
        raw = np.atleast_2d(raw)
        if raw.shape[1] not in (12, 16):
            raise ValueError("replica trajectory rows must contain 12 or 16 values")
        poses = []
        for row in raw:
            if row.shape[0] == 16:
                pose = row.reshape(4, 4)
            else:
                pose = np.eye(4)
                pose[:3] = row.reshape(3, 4)
            poses.append(pose)
        return GroundTruthTrajectory(
            tuple(range(len(poses))),
            torch.as_tensor(np.stack(poses), dtype=torch.float64),
        )
    if format_name == "sintel":
        camera_files = sorted(path / name for name in os.listdir(path) if name.endswith(".cam"))
        poses = []
        for camera_file in camera_files:
            with camera_file.open("rb") as source:
                tag = np.fromfile(source, dtype=np.float32, count=1)
                if tag.shape != (1,) or tag[0] != np.float32(202021.25):
                    raise ValueError(f"invalid Sintel camera file: {camera_file}")
                np.fromfile(source, dtype=np.float64, count=9)
                world_to_camera = np.fromfile(source, dtype=np.float64, count=12)
            if world_to_camera.shape != (12,):
                raise ValueError(f"truncated Sintel camera file: {camera_file}")
            matrix = np.eye(4)
            matrix[:3] = world_to_camera.reshape(3, 4)
            poses.append(np.linalg.inv(matrix))
        if len(poses) < 2:
            raise ValueError("Sintel trajectory requires at least two camera files")
        return GroundTruthTrajectory(
            tuple(range(len(poses))),
            torch.as_tensor(np.stack(poses), dtype=torch.float64),
        )
    raise ValueError(f"unsupported ground-truth format: {format_name!r}")
