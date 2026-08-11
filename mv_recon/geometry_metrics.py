from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from scipy.spatial import cKDTree

from mv_recon.eval_utils import umeyama
from mv_recon.protocol import GeometryProtocol


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


def _finite_array(
    value: object,
    label: str,
    *,
    shape_tail: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if shape_tail is not None and (
        array.ndim < len(shape_tail)
        or tuple(array.shape[-len(shape_tail) :]) != shape_tail
    ):
        raise ValueError(f"{label} has invalid shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _validate_points(value: object, label: str) -> np.ndarray:
    array = _finite_array(value, label, shape_tail=(3,))
    if array.ndim != 2 or array.shape[0] < 1:
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
    mean = (float(np.mean(first)) + float(np.mean(second))) / 2.0
    median = (float(np.median(first)) + float(np.median(second))) / 2.0
    return mean, median


def _threshold_metrics(
    accuracy_distances: np.ndarray,
    completion_distances: np.ndarray,
    thresholds: Sequence[float],
) -> tuple[ThresholdMetrics, ...]:
    results = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("F-score thresholds must be finite and positive")
        precision = float(np.mean(accuracy_distances < threshold))
        recall = float(np.mean(completion_distances < threshold))
        denominator = precision + recall
        fscore = (
            0.0
            if denominator == 0.0
            else float(2.0 * precision * recall / denominator)
        )
        results.append(
            ThresholdMetrics(
                threshold_m=threshold,
                precision=precision,
                recall=recall,
                fscore=fscore,
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
    predicted = _validate_points(predicted_points, "predicted points")
    ground_truth = _validate_points(
        ground_truth_points,
        "ground-truth points",
    )
    pred_normals = _validate_points(predicted_normals, "predicted normals")
    gt_normals = _validate_points(
        ground_truth_normals,
        "ground-truth normals",
    )
    if pred_normals.shape != predicted.shape:
        raise ValueError("predicted normal shape must match predicted points")
    if gt_normals.shape != ground_truth.shape:
        raise ValueError("ground-truth normal shape must match ground-truth points")

    accuracy_distances, gt_indices = cKDTree(ground_truth).query(
        predicted,
        workers=-1,
    )
    completion_distances, pred_indices = cKDTree(predicted).query(
        ground_truth,
        workers=-1,
    )
    accuracy_distances = _finite_array(
        accuracy_distances,
        "accuracy distances",
    ).reshape(-1)
    completion_distances = _finite_array(
        completion_distances,
        "completion distances",
    ).reshape(-1)

    nc1_samples = np.abs(
        np.sum(gt_normals[gt_indices] * pred_normals, axis=-1)
    )
    nc2_samples = np.abs(
        np.sum(gt_normals * pred_normals[pred_indices], axis=-1)
    )
    nc_mean, nc_median = combine_normal_consistency(
        nc1_samples,
        nc2_samples,
    )
    primary = PrimaryMetrics(
        accuracy_mean_m=float(np.mean(accuracy_distances)),
        accuracy_median_m=float(np.median(accuracy_distances)),
        completion_mean_m=float(np.mean(completion_distances)),
        completion_median_m=float(np.median(completion_distances)),
        normal_consistency_mean=nc_mean,
        normal_consistency_median=nc_median,
    )
    directional = DirectionalNormalMetrics(
        nc1_mean=float(np.mean(nc1_samples)),
        nc1_median=float(np.median(nc1_samples)),
        nc2_mean=float(np.mean(nc2_samples)),
        nc2_median=float(np.median(nc2_samples)),
    )
    return DirectionalMetricResult(
        primary=primary,
        directional_normals=directional,
        thresholds=_threshold_metrics(
            accuracy_distances,
            completion_distances,
            fscore_thresholds_m,
        ),
        chamfer_l1_m=(
            primary.accuracy_mean_m + primary.completion_mean_m
        )
        / 2.0,
    )


def _center_crop_bounds(
    height: int,
    width: int,
    size: int,
) -> tuple[slice, slice]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("crop size must be a positive integer")
    if height < size or width < size:
        raise ValueError(
            f"crop size {size} exceeds input spatial shape ({height}, {width})"
        )
    top = height // 2 - size // 2
    left = width // 2 - size // 2
    return slice(top, top + size), slice(left, left + size)


def _validate_point_maps(
    predicted_points: object,
    ground_truth_points: object,
    valid_mask: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_points)
    ground_truth = np.asarray(ground_truth_points)
    mask = np.asarray(valid_mask)
    if predicted.ndim != 4 or predicted.shape[-1] != 3:
        raise ValueError(
            f"predicted point maps must have shape (N,H,W,3), got {predicted.shape}"
        )
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"predicted shape {predicted.shape} does not match "
            f"ground truth {ground_truth.shape}"
        )
    if mask.shape != ground_truth.shape[:-1]:
        raise ValueError(
            f"valid mask shape {mask.shape} does not match "
            f"point maps {ground_truth.shape[:-1]}"
        )
    return predicted, ground_truth, mask.astype(bool, copy=False)


def _validate_backend_result(result: BackendResult) -> BackendResult:
    predicted = _validate_points(
        result.predicted_points,
        "ICP predicted points",
    )
    ground_truth = _validate_points(
        result.ground_truth_points,
        "ICP ground-truth points",
    )
    pred_normals = _validate_points(
        result.predicted_normals,
        "ICP predicted normals",
    )
    gt_normals = _validate_points(
        result.ground_truth_normals,
        "ICP ground-truth normals",
    )
    if pred_normals.shape != predicted.shape:
        raise ValueError("ICP predicted normals must match predicted points")
    if gt_normals.shape != ground_truth.shape:
        raise ValueError("ICP ground-truth normals must match ground-truth points")
    transformation = _finite_array(
        result.transformation,
        "ICP transformation",
    )
    if transformation.shape != (4, 4):
        raise ValueError("ICP transformation must have shape (4, 4)")
    fitness = float(result.fitness)
    inlier_rmse = float(result.inlier_rmse)
    if not math.isfinite(fitness) or not math.isfinite(inlier_rmse):
        raise ValueError("ICP fitness and inlier RMSE must be finite")
    return BackendResult(
        predicted_points=predicted,
        ground_truth_points=ground_truth,
        predicted_normals=pred_normals,
        ground_truth_normals=gt_normals,
        transformation=np.asarray(transformation, dtype=np.float64),
        fitness=fitness,
        inlier_rmse=inlier_rmse,
    )


def evaluate_point_maps(
    predicted_points: np.ndarray,
    ground_truth_points: np.ndarray,
    valid_mask: np.ndarray,
    geometry: GeometryProtocol,
    *,
    backend: GeometryBackend | None = None,
) -> GeometryEvaluation:
    predicted, ground_truth, mask = _validate_point_maps(
        predicted_points,
        ground_truth_points,
        valid_mask,
    )
    crop_y, crop_x = _center_crop_bounds(
        ground_truth.shape[1],
        ground_truth.shape[2],
        geometry.center_crop_size,
    )
    predicted = predicted[:, crop_y, crop_x]
    ground_truth = ground_truth[:, crop_y, crop_x]
    mask = mask[:, crop_y, crop_x]

    predicted_correspondences = predicted[mask]
    gt_correspondences = ground_truth[mask]
    if predicted_correspondences.shape[0] < 3:
        raise ValueError("Umeyama alignment requires at least three valid points")
    if not np.all(np.isfinite(predicted_correspondences)) or not np.all(
        np.isfinite(gt_correspondences)
    ):
        raise ValueError("selected Umeyama points contain non-finite values")
    centered = predicted_correspondences - np.mean(
        predicted_correspondences,
        axis=0,
        keepdims=True,
    )
    variance = float(np.mean(np.sum(np.square(centered), axis=1)))
    if not math.isfinite(variance) or variance <= np.finfo(np.float64).eps:
        raise ValueError("degenerate Umeyama prediction variance")

    with np.errstate(all="ignore"):
        scale, rotation, translation = umeyama(
            predicted_correspondences.T,
            gt_correspondences.T,
        )
    scale = float(scale)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if (
        not math.isfinite(scale)
        or scale <= 0
        or rotation.shape != (3, 3)
        or translation.shape != (3, 1)
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
    ):
        raise ValueError("Umeyama produced a non-finite or invalid transform")
    aligned_maps = (
        scale * np.einsum("nhwj,ij->nhwi", predicted, rotation)
        + translation.T
    )
    aligned_predicted = np.asarray(aligned_maps[mask], dtype=np.float64)
    flattened_gt = np.asarray(ground_truth[mask], dtype=np.float64)
    if aligned_predicted.size == 0 or flattened_gt.size == 0:
        raise ValueError("aligned point clouds must be non-empty")

    selected_backend = backend or Open3DGeometryBackend()
    backend_result = _validate_backend_result(
        selected_backend.refine_and_estimate_normals(
            aligned_predicted,
            flattened_gt,
            geometry.icp_threshold_m,
        )
    )
    directional = compute_directional_metrics(
        backend_result.predicted_points,
        backend_result.ground_truth_points,
        backend_result.predicted_normals,
        backend_result.ground_truth_normals,
        geometry.fscore_thresholds_m,
    )
    return GeometryEvaluation(
        primary=directional.primary,
        diagnostics=GeometryDiagnostics(
            umeyama_scale=scale,
            icp_transformation=tuple(
                tuple(float(value) for value in row)
                for row in backend_result.transformation
            ),
            icp_fitness=backend_result.fitness,
            icp_inlier_rmse=backend_result.inlier_rmse,
            predicted_point_count=len(backend_result.predicted_points),
            ground_truth_point_count=len(backend_result.ground_truth_points),
            directional_normals=directional.directional_normals,
            chamfer_l1_m=directional.chamfer_l1_m,
            thresholds=directional.thresholds,
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
            raise RuntimeError(
                "strict LASER paper evaluation requires Open3D; install "
                "requirements with the repository Python 3.11 environment"
            ) from exc

        predicted_cloud = o3d.geometry.PointCloud()
        predicted_cloud.points = o3d.utility.Vector3dVector(predicted)
        gt_cloud = o3d.geometry.PointCloud()
        gt_cloud.points = o3d.utility.Vector3dVector(ground_truth)
        registration = o3d.pipelines.registration.registration_icp(
            predicted_cloud,
            gt_cloud,
            float(threshold_m),
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        predicted_cloud.transform(registration.transformation)
        predicted_cloud.estimate_normals()
        gt_cloud.estimate_normals()
        return BackendResult(
            predicted_points=np.asarray(predicted_cloud.points),
            ground_truth_points=np.asarray(gt_cloud.points),
            predicted_normals=np.asarray(predicted_cloud.normals),
            ground_truth_normals=np.asarray(gt_cloud.normals),
            transformation=np.asarray(registration.transformation),
            fitness=float(registration.fitness),
            inlier_rmse=float(registration.inlier_rmse),
        )
