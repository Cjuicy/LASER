from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .config import PointCloudEvaluationConfig


@dataclass(frozen=True)
class PrimaryMetrics:
    accuracy_mean_m: float
    accuracy_median_m: float
    completion_mean_m: float
    completion_median_m: float
    normal_consistency_mean: float
    normal_consistency_median: float


@dataclass(frozen=True)
class DirectionalNormalMetrics:
    nc1_mean: float
    nc1_median: float
    nc2_mean: float
    nc2_median: float


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold_m: float
    precision: float
    recall: float
    fscore: float


@dataclass(frozen=True)
class DirectionalMetricResult:
    primary: PrimaryMetrics
    directional_normals: DirectionalNormalMetrics
    thresholds: tuple[ThresholdMetrics, ...]
    chamfer_l1_m: float


@dataclass(frozen=True)
class BackendResult:
    predicted_points: np.ndarray
    ground_truth_points: np.ndarray
    predicted_normals: np.ndarray
    ground_truth_normals: np.ndarray
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float


class GeometryBackend(Protocol):
    def refine_and_estimate_normals(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        threshold_m: float,
    ) -> BackendResult: ...


@dataclass(frozen=True)
class GeometryDiagnostics:
    umeyama_scale: float
    icp_transformation: tuple[tuple[float, ...], ...]
    icp_fitness: float
    icp_inlier_rmse: float
    predicted_point_count: int
    ground_truth_point_count: int
    directional_normals: DirectionalNormalMetrics
    chamfer_l1_m: float
    thresholds: tuple[ThresholdMetrics, ...]


@dataclass(frozen=True)
class GeometryEvaluation:
    primary: PrimaryMetrics
    diagnostics: GeometryDiagnostics


def _finite_array(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _points(value: object, label: str) -> np.ndarray:
    array = _finite_array(value, label)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3) with N >= 1")
    return np.asarray(array, dtype=np.float64)


def combine_normal_consistency(
    nc1: np.ndarray,
    nc2: np.ndarray,
) -> tuple[float, float]:
    first = _finite_array(nc1, "NC1").reshape(-1)
    second = _finite_array(nc2, "NC2").reshape(-1)
    if first.size == 0 or second.size == 0:
        raise ValueError("normal consistency directions must be non-empty")
    return (
        (float(np.mean(first)) + float(np.mean(second))) / 2.0,
        (float(np.median(first)) + float(np.median(second))) / 2.0,
    )


def _threshold_metrics(
    accuracy: np.ndarray,
    completion: np.ndarray,
    thresholds: Sequence[float],
) -> tuple[ThresholdMetrics, ...]:
    results = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("F-score thresholds must be finite and positive")
        precision = float(np.mean(accuracy < threshold))
        recall = float(np.mean(completion < threshold))
        denominator = precision + recall
        results.append(
            ThresholdMetrics(
                threshold,
                precision,
                recall,
                0.0
                if denominator == 0.0
                else float(2.0 * precision * recall / denominator),
            )
        )
    return tuple(results)


def compute_directional_metrics(
    predicted_points: np.ndarray,
    ground_truth_points: np.ndarray,
    predicted_normals: np.ndarray,
    ground_truth_normals: np.ndarray,
    fscore_thresholds_m: Sequence[float],
) -> DirectionalMetricResult:
    predicted = _points(predicted_points, "predicted points")
    ground_truth = _points(ground_truth_points, "ground-truth points")
    pred_normals = _points(predicted_normals, "predicted normals")
    gt_normals = _points(ground_truth_normals, "ground-truth normals")
    if pred_normals.shape != predicted.shape:
        raise ValueError("predicted normal shape must match predicted points")
    if gt_normals.shape != ground_truth.shape:
        raise ValueError("ground-truth normal shape must match ground-truth points")
    accuracy, gt_indices = cKDTree(ground_truth).query(predicted, workers=-1)
    completion, pred_indices = cKDTree(predicted).query(ground_truth, workers=-1)
    accuracy = _finite_array(accuracy, "accuracy distances").reshape(-1)
    completion = _finite_array(completion, "completion distances").reshape(-1)
    nc1 = np.abs(np.sum(gt_normals[gt_indices] * pred_normals, axis=-1))
    nc2 = np.abs(np.sum(gt_normals * pred_normals[pred_indices], axis=-1))
    nc_mean, nc_median = combine_normal_consistency(nc1, nc2)
    primary = PrimaryMetrics(
        float(np.mean(accuracy)),
        float(np.median(accuracy)),
        float(np.mean(completion)),
        float(np.median(completion)),
        nc_mean,
        nc_median,
    )
    return DirectionalMetricResult(
        primary=primary,
        directional_normals=DirectionalNormalMetrics(
            float(np.mean(nc1)),
            float(np.median(nc1)),
            float(np.mean(nc2)),
            float(np.median(nc2)),
        ),
        thresholds=_threshold_metrics(accuracy, completion, fscore_thresholds_m),
        chamfer_l1_m=(primary.accuracy_mean_m + primary.completion_mean_m) / 2.0,
    )


def _crop(height: int, width: int, size: int) -> tuple[slice, slice]:
    if height < size or width < size:
        raise ValueError(
            f"crop size {size} exceeds input spatial shape ({height}, {width})"
        )
    top = height // 2 - size // 2
    left = width // 2 - size // 2
    return slice(top, top + size), slice(left, left + size)


def _umeyama(source: np.ndarray, target: np.ndarray):
    x = source.T
    y = target.T
    mean_x = x.mean(axis=1, keepdims=True)
    mean_y = y.mean(axis=1, keepdims=True)
    variance = np.square(x - mean_x).sum(axis=0).mean()
    covariance = ((y - mean_y) @ (x - mean_x).T) / x.shape[1]
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.eye(x.shape[0])
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1
    scale = np.trace(np.diag(singular) @ sign) / variance
    rotation = u @ sign @ vh
    translation = mean_y - scale * rotation @ mean_x
    return float(scale), rotation, translation


def evaluate_aligned_point_maps(
    predicted_points: np.ndarray,
    ground_truth_points: np.ndarray,
    valid_mask: np.ndarray,
    config: PointCloudEvaluationConfig,
    backend: GeometryBackend,
) -> GeometryEvaluation:
    predicted = np.asarray(predicted_points)
    ground_truth = np.asarray(ground_truth_points)
    mask = np.asarray(valid_mask)
    if predicted.ndim != 4 or predicted.shape[-1] != 3:
        raise ValueError("predicted point maps must have shape (N,H,W,3)")
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground-truth point maps must match")
    if mask.shape != predicted.shape[:-1]:
        raise ValueError("valid mask shape must match point maps")
    crop_y, crop_x = _crop(
        predicted.shape[1], predicted.shape[2], config.center_crop_size
    )
    predicted = predicted[:, crop_y, crop_x]
    ground_truth = ground_truth[:, crop_y, crop_x]
    mask = mask[:, crop_y, crop_x].astype(bool, copy=False)
    source = predicted[mask]
    target = ground_truth[mask]
    if len(source) < 3:
        raise ValueError("Umeyama alignment requires at least three valid points")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("selected Umeyama points contain non-finite values")
    centered = source - np.mean(source, axis=0, keepdims=True)
    variance = float(np.mean(np.sum(np.square(centered), axis=1)))
    if not math.isfinite(variance) or variance <= np.finfo(np.float64).eps:
        raise ValueError("degenerate Umeyama prediction variance")
    scale, rotation, translation = _umeyama(source, target)
    if (
        not math.isfinite(scale)
        or scale <= 0
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
    ):
        raise ValueError("Umeyama produced a non-finite or invalid transform")
    aligned = (
        scale * np.einsum("nhwj,ij->nhwi", predicted, rotation)
        + translation.T
    )[mask]
    backend_result = backend.refine_and_estimate_normals(
        np.asarray(aligned, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        config.icp_threshold_m,
    )
    pred = _points(backend_result.predicted_points, "ICP predicted points")
    gt = _points(backend_result.ground_truth_points, "ICP ground-truth points")
    pred_normals = _points(backend_result.predicted_normals, "ICP predicted normals")
    gt_normals = _points(backend_result.ground_truth_normals, "ICP ground-truth normals")
    transform = _finite_array(backend_result.transformation, "ICP transformation")
    if transform.shape != (4, 4):
        raise ValueError("ICP transformation must have shape (4, 4)")
    directional = compute_directional_metrics(
        pred, gt, pred_normals, gt_normals, config.fscore_thresholds_m
    )
    return GeometryEvaluation(
        directional.primary,
        GeometryDiagnostics(
            scale,
            tuple(tuple(float(value) for value in row) for row in transform),
            float(backend_result.fitness),
            float(backend_result.inlier_rmse),
            len(pred),
            len(gt),
            directional.directional_normals,
            directional.chamfer_l1_m,
            directional.thresholds,
        ),
    )


class Open3DGeometryBackend:
    def refine_and_estimate_normals(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        threshold_m: float,
    ) -> BackendResult:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("point-cloud evaluation requires Open3D") from exc
        predicted_cloud = o3d.geometry.PointCloud()
        predicted_cloud.points = o3d.utility.Vector3dVector(predicted)
        ground_truth_cloud = o3d.geometry.PointCloud()
        ground_truth_cloud.points = o3d.utility.Vector3dVector(ground_truth)
        registration = o3d.pipelines.registration.registration_icp(
            predicted_cloud,
            ground_truth_cloud,
            float(threshold_m),
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        predicted_cloud.transform(registration.transformation)
        predicted_cloud.estimate_normals()
        ground_truth_cloud.estimate_normals()
        return BackendResult(
            np.asarray(predicted_cloud.points),
            np.asarray(ground_truth_cloud.points),
            np.asarray(predicted_cloud.normals),
            np.asarray(ground_truth_cloud.normals),
            np.asarray(registration.transformation),
            float(registration.fitness),
            float(registration.inlier_rmse),
        )
