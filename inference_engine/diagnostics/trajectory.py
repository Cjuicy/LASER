"""Trajectory metrics with one global Sim(3) alignment per sequence."""

from __future__ import annotations

import numpy as np


SIGNATURE = {
    "ape_relation": "translation_part",
    "align": True,
    "correct_scale": True,
    "rpe_translation_relation": "translation_part",
    "rpe_rotation_relation": "rotation_angle_deg",
    "rpe_delta": 1,
    "delta_unit": "frames",
    "all_pairs": True,
}


def _invalid(reason: str) -> dict:
    return {
        "valid": False,
        "invalid_reason": reason,
        "ate_rmse": None,
        "rpe_translation_rmse": None,
        "rpe_rotation_rmse_deg": None,
        "per_frame_translation_error": None,
        "aligned_poses": None,
        "alignment": None,
        "evaluation_signature": dict(SIGNATURE),
    }


def _sim3_align(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    variance = float(np.mean(np.sum(x * x, axis=1)))
    if not np.isfinite(variance) or variance <= 1e-15:
        raise ValueError("degenerate_predicted_positions")
    covariance = y.T @ x / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular * signs) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    if cosine > 1.0 - 1e-14:
        return 0.0
    return float(np.rad2deg(np.arccos(cosine)))


def evaluate_trajectory(pred, gt) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[1:] != (4, 4) or gt.shape[1:] != (4, 4):
        return _invalid("poses_must_have_shape_n_4_4")
    count = min(len(pred), len(gt))
    if count < 2:
        return _invalid("at_least_two_poses_required")
    pred = pred[:count]
    gt = gt[:count]
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        return _invalid("non_finite_poses")
    try:
        scale, rotation, translation = _sim3_align(pred[:, :3, 3], gt[:, :3, 3])
    except ValueError as exc:
        return _invalid(str(exc))
    aligned = pred.copy()
    aligned[:, :3, 3] = scale * (rotation @ pred[:, :3, 3].T).T + translation
    aligned[:, :3, :3] = rotation @ pred[:, :3, :3]
    frame_error = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    trans_error: list[float] = []
    rot_error: list[float] = []
    for index in range(count - 1):
        reference_delta = np.linalg.inv(gt[index]) @ gt[index + 1]
        estimate_delta = np.linalg.inv(aligned[index]) @ aligned[index + 1]
        error = np.linalg.inv(reference_delta) @ estimate_delta
        trans_error.append(float(np.linalg.norm(error[:3, 3])))
        rot_error.append(_rotation_angle_degrees(error[:3, :3]))
    return {
        "valid": True,
        "invalid_reason": None,
        "ate_rmse": float(np.sqrt(np.mean(frame_error ** 2))),
        "rpe_translation_rmse": float(np.sqrt(np.mean(np.square(trans_error)))),
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rot_error)))),
        "per_frame_translation_error": frame_error.tolist(),
        "aligned_poses": aligned,
        "alignment": {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()},
        "associated_pose_count": count,
        "evaluation_signature": dict(SIGNATURE),
    }
