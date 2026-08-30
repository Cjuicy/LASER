from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import torch

from pipeline.config import ConfidenceQuantileMethod, SegmentationConfig


def _centered_axis(length: int, stride: int) -> np.ndarray:
    start = ((length - 1) % stride) // 2
    return np.arange(start, length, stride, dtype=np.int64)


def _confidence_probability(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -20.0, 20.0)
    with np.errstate(over="ignore", under="ignore"):
        return 1.0 / (1.0 + np.exp(-clipped))


def _confidence_mask_and_quality(
    points: np.ndarray,
    logits: np.ndarray,
    *,
    keep_ratio: float,
    method: ConfidenceQuantileMethod | str,
) -> tuple[np.ndarray, float]:
    points = np.asarray(points)
    logits = np.asarray(logits)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (H, W, 3)")
    if logits.shape != points.shape[:2]:
        raise ValueError("confidence must have shape (H, W)")
    if not np.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError("confidence keep_ratio must be in (0, 1]")
    try:
        quantile_method = ConfidenceQuantileMethod(method)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "confidence quantile method must be higher or nearest"
        ) from exc

    valid = (
        np.isfinite(logits)
        & np.all(np.isfinite(points), axis=-1)
        & (points[..., 2] > 1e-6)
    )
    finite_values = logits[valid]
    if finite_values.size == 0:
        return np.zeros(logits.shape, dtype=bool), 0.0

    threshold = np.quantile(
        finite_values,
        1.0 - keep_ratio,
        method=quantile_method.value,
    )
    selected = valid & (logits >= threshold)
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        return selected, 0.0
    probability_mean = float(np.mean(_confidence_probability(logits[selected])))
    quality = (selected_count / float(logits.size)) * probability_mean
    return selected, float(np.clip(quality, 0.0, 1.0))


def _tensor_numpy(value: object, *, dtype: np.dtype) -> np.ndarray:
    try:
        if isinstance(value, torch.Tensor):
            requested = np.dtype(dtype)
            torch_dtype = {
                np.dtype(np.float16): torch.float16,
                np.dtype(np.float32): torch.float32,
                np.dtype(np.float64): torch.float64,
            }.get(requested)
            if torch_dtype is None:
                raise TypeError(f"unsupported tensor dtype: {requested}")
            return (
                value.detach()
                .to(device="cpu", dtype=torch_dtype)
                .numpy()
                .copy()
            )
        return np.asarray(value, dtype=dtype).copy()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("window reference tensors must be numeric") from exc


def _validate_window_shapes(
    point_maps: object,
    camera_poses: object,
    confidence: object,
) -> tuple[int, int, int]:
    try:
        point_shape = tuple(point_maps.shape)  # type: ignore[attr-defined]
        pose_shape = tuple(camera_poses.shape)  # type: ignore[attr-defined]
        confidence_shape = tuple(confidence.shape)  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ValueError(
            "point_maps, camera_poses, and confidence must expose shapes"
        ) from exc

    if len(point_shape) != 4 or point_shape[-1] != 3:
        raise ValueError("point_maps must have shape (N, H, W, 3)")
    if len(pose_shape) != 3 or pose_shape[1:] != (4, 4):
        raise ValueError("camera_poses must have shape (N, 4, 4)")
    if len(confidence_shape) != 3:
        raise ValueError("confidence must have shape (N, H, W)")

    frame_count, height, width, _ = point_shape
    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError("window reference tensors must be non-empty")
    if pose_shape[0] != frame_count:
        raise ValueError("camera_poses frame count must match point_maps")
    if confidence_shape != (frame_count, height, width):
        raise ValueError("confidence shape must match point_maps")
    return frame_count, height, width


def _validate_intrinsic(reference_intrinsic: object) -> np.ndarray | None:
    try:
        if isinstance(reference_intrinsic, torch.Tensor):
            intrinsic = (
                reference_intrinsic.detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
                .copy()
            )
        else:
            intrinsic = np.asarray(reference_intrinsic, dtype=np.float64).copy()
    except (TypeError, ValueError, RuntimeError):
        return None
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        return None
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        return None
    if not np.allclose(
        intrinsic[2],
        np.array([0.0, 0.0, 1.0]),
        rtol=1e-6,
        atol=1e-6,
    ):
        return None
    return intrinsic


def _validate_pose_geometry(camera_poses: np.ndarray) -> bool:
    if not np.all(np.isfinite(camera_poses)):
        return False
    for pose in camera_poses:
        try:
            inverse = np.linalg.inv(pose.astype(np.float64, copy=False))
        except np.linalg.LinAlgError:
            return False
        if not np.all(np.isfinite(inverse)):
            return False
    return True


def _intrinsic_compatible(
    point_maps: np.ndarray,
    intrinsic: np.ndarray,
    stride: int,
) -> bool:
    _, height, width, _ = point_maps.shape
    rows = _centered_axis(height, stride)
    columns = _centered_axis(width, stride)
    sampled = point_maps[:, rows[:, None], columns[None, :], :]
    valid = np.all(np.isfinite(sampled), axis=-1) & (sampled[..., 2] > 1e-6)
    if not np.any(valid):
        return False
    xyz = sampled[valid]
    homogeneous = xyz @ intrinsic.T
    denominator = homogeneous[:, 2]
    if not np.all(np.isfinite(homogeneous)) or not np.all(denominator > 1e-6):
        return False
    u = homogeneous[:, 0] / denominator
    v = homogeneous[:, 1] / denominator
    expected_u = np.broadcast_to(columns[None, :], sampled.shape[:3])[valid]
    expected_v = np.broadcast_to(rows[:, None], sampled.shape[:3])[valid]
    return bool(
        np.all(np.abs(u - expected_u) <= 0.5)
        and np.all(np.abs(v - expected_v) <= 0.5)
    )


@dataclass(frozen=True)
class PairProjection:
    source_flat_indices: np.ndarray
    target_flat_indices: np.ndarray
    correspondence_weights: np.ndarray
    coverage: float = 0.0
    geometry_ratio: float = 0.0
    mean_confidence: float = 0.0
    score: float = 0.0
    projected_samples: int = 0
    occluded_samples: int = 0
    depth_rejected_samples: int = 0
    target_invalid_samples: int = 0
    out_of_bounds_samples: int = 0
    nonpositive_depth_samples: int = 0
    invalid_source_samples: int = 0

    def __post_init__(self) -> None:
        arrays = (
            ("source_flat_indices", np.int64),
            ("target_flat_indices", np.int64),
            ("correspondence_weights", np.float64),
        )
        for name, dtype in arrays:
            value = np.asarray(getattr(self, name), dtype=dtype).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PairSummary:
    """Scalar pair evidence retained after adaptive selection scoring."""

    score: float = 0.0
    coverage: float = 0.0
    geometry_ratio: float = 0.0
    mean_confidence: float = 0.0
    projected_samples: int = 0
    occluded_samples: int = 0
    depth_rejected_samples: int = 0
    target_invalid_samples: int = 0
    out_of_bounds_samples: int = 0
    nonpositive_depth_samples: int = 0
    invalid_source_samples: int = 0


@dataclass(frozen=True)
class WindowReferenceSelection:
    indices: tuple[int, ...]
    best_scores: np.ndarray
    pair_projections: tuple[tuple[int, int, object], ...] = ()
    rejected_indices: tuple[int, ...] = ()
    coverage_ratio: float = 0.0
    diagnostics: Mapping[str, int | float] = field(default_factory=dict)
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.best_scores, dtype=np.float64).copy()
        scores.setflags(write=False)
        indices = tuple(int(index) for index in self.indices)
        rejected_indices = tuple(int(index) for index in self.rejected_indices)
        coverage_ratio = float(self.coverage_ratio)
        fallback_reason = self.fallback_reason
        if indices != tuple(sorted(set(indices))):
            raise ValueError("reference indices must be sorted and unique")
        if not np.isfinite(coverage_ratio) or not 0.0 <= coverage_ratio <= 1.0:
            raise ValueError("reference coverage ratio must be finite and in [0, 1]")
        if fallback_reason is not None:
            if indices or not str(fallback_reason):
                raise ValueError("fallback selection requires empty indices and a reason")
            fallback_reason = str(fallback_reason)
        elif not indices:
            raise ValueError("successful selection requires at least one reference")
        object.__setattr__(self, "best_scores", scores)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "rejected_indices", rejected_indices)
        object.__setattr__(self, "pair_projections", tuple(self.pair_projections))
        object.__setattr__(self, "coverage_ratio", coverage_ratio)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        object.__setattr__(self, "fallback_reason", fallback_reason)


def _projection_score(projection: object) -> float:
    score = getattr(projection, "score", projection)
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("pair evaluator must return a numeric score") from exc
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _pair_summary(projection: object) -> PairSummary:
    return PairSummary(
        score=_projection_score(projection),
        coverage=float(getattr(projection, "coverage", 0.0)),
        geometry_ratio=float(getattr(projection, "geometry_ratio", 0.0)),
        mean_confidence=float(getattr(projection, "mean_confidence", 0.0)),
        projected_samples=int(getattr(projection, "projected_samples", 0)),
        occluded_samples=int(getattr(projection, "occluded_samples", 0)),
        depth_rejected_samples=int(
            getattr(projection, "depth_rejected_samples", 0)
        ),
        target_invalid_samples=int(getattr(projection, "target_invalid_samples", 0)),
        out_of_bounds_samples=int(getattr(projection, "out_of_bounds_samples", 0)),
        nonpositive_depth_samples=int(
            getattr(projection, "nonpositive_depth_samples", 0)
        ),
        invalid_source_samples=int(getattr(projection, "invalid_source_samples", 0)),
    )


def _fallback_selection(frame_count: int, reason: str) -> WindowReferenceSelection:
    return WindowReferenceSelection(
        indices=(),
        best_scores=np.zeros(frame_count, dtype=np.float64),
        coverage_ratio=0.0,
        diagnostics={
            "evaluated_pairs": 0,
            "reliable_pairs": 0,
            "selected_count": 0,
            "rejected_count": 0,
            "coverage_ratio": 0.0,
        },
        fallback_reason=reason,
    )


def _select_references(
    *,
    qualities: np.ndarray,
    frame_count: int,
    evaluate: Callable[[int, int], object],
    config: object,
) -> WindowReferenceSelection:
    qualities = np.asarray(qualities, dtype=np.float64)
    if qualities.ndim != 1 or qualities.size != frame_count:
        raise ValueError("qualities must contain one value per frame")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")

    best_scores = np.zeros(frame_count, dtype=np.float64)
    positive = [
        index
        for index, quality in enumerate(qualities)
        if np.isfinite(quality) and quality > 0.0
    ]
    if not positive:
        return _fallback_selection(frame_count, "no_reference")

    center = (frame_count - 1) / 2.0
    first = min(
        positive,
        key=lambda index: (-qualities[index], abs(index - center), index),
    )
    selected_order = [first]
    rejected_indices: list[int] = []
    reliable_summaries: list[tuple[int, int, PairSummary]] = []
    evaluated_pairs = 0
    max_keyframes = min(int(config.max_keyframes), frame_count)

    def evaluate_source(source: int) -> None:
        nonlocal evaluated_pairs
        for target in range(frame_count):
            if target == source:
                continue
            projection = evaluate(source, target)
            evaluated_pairs += 1
            score = _projection_score(projection)
            best_scores[target] = max(best_scores[target], score)
            if score >= config.min_reference_score:
                reliable_summaries.append((source, target, _pair_summary(projection)))
            del projection

    def current_coverage() -> float:
        covered = best_scores >= config.min_reference_score
        covered[selected_order] = True
        return float(np.mean(covered))

    evaluate_source(first)
    coverage_ratio = current_coverage()
    while (
        len(selected_order) < max_keyframes
        and coverage_ratio < config.stop_coverage_ratio
    ):
        candidates = [
            index
            for index, quality in enumerate(qualities)
            if (
                np.isfinite(quality)
                and quality > 0.0
                and index not in selected_order
                and index not in rejected_indices
            )
        ]
        if not candidates:
            break
        candidate = min(
            candidates,
            key=lambda index: (
                -qualities[index] * (1.0 - best_scores[index]),
                index,
            ),
        )

        candidate_scores = np.zeros(frame_count, dtype=np.float64)
        candidate_summaries: list[tuple[int, int, PairSummary]] = []
        for target in range(frame_count):
            if target == candidate:
                continue
            projection = evaluate(candidate, target)
            evaluated_pairs += 1
            score = _projection_score(projection)
            candidate_scores[target] = score
            if score >= config.min_reference_score:
                candidate_summaries.append(
                    (candidate, target, _pair_summary(projection))
                )
            del projection

        selected_set = set(selected_order)
        eligible_targets = [
            target
            for target in range(frame_count)
            if target not in selected_set and target != candidate
        ]
        gain = (
            float(
                np.mean(
                    [
                        max(best_scores[target], candidate_scores[target])
                        - best_scores[target]
                        for target in eligible_targets
                    ]
                )
            )
            if eligible_targets
            else 0.0
        )
        if gain < config.min_coverage_gain:
            rejected_indices.append(candidate)
            continue

        selected_order.append(candidate)
        np.maximum(best_scores, candidate_scores, out=best_scores)
        reliable_summaries.extend(candidate_summaries)
        coverage_ratio = current_coverage()

    indices = tuple(sorted(selected_order))
    diagnostics = {
        "evaluated_pairs": evaluated_pairs,
        "reliable_pairs": len(reliable_summaries),
        "selected_count": len(indices),
        "rejected_count": len(rejected_indices),
        "coverage_ratio": coverage_ratio,
    }
    return WindowReferenceSelection(
        indices=indices,
        best_scores=best_scores,
        pair_projections=tuple(reliable_summaries),
        rejected_indices=tuple(rejected_indices),
        coverage_ratio=coverage_ratio,
        diagnostics=diagnostics,
    )


def _empty_pair_projection(
    *,
    projected_samples: int = 0,
    occluded_samples: int = 0,
    depth_rejected_samples: int = 0,
    target_invalid_samples: int = 0,
    out_of_bounds_samples: int = 0,
    nonpositive_depth_samples: int = 0,
    invalid_source_samples: int = 0,
) -> PairProjection:
    return PairProjection(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        projected_samples=projected_samples,
        occluded_samples=occluded_samples,
        depth_rejected_samples=depth_rejected_samples,
        target_invalid_samples=target_invalid_samples,
        out_of_bounds_samples=out_of_bounds_samples,
        nonpositive_depth_samples=nonpositive_depth_samples,
        invalid_source_samples=invalid_source_samples,
    )


def project_pair(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_probability: np.ndarray,
    target_probability: np.ndarray,
    intrinsic: np.ndarray,
    source_rows: np.ndarray,
    source_columns: np.ndarray,
    sampling_stride: int,
    relative_depth_tolerance: float,
) -> PairProjection:
    """Project one sparse source frame into one target frame."""

    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)
    source_pose = np.asarray(source_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    source_mask = np.asarray(source_mask)
    target_mask = np.asarray(target_mask)
    source_probability = np.asarray(source_probability)
    target_probability = np.asarray(target_probability)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    source_columns = np.asarray(source_columns, dtype=np.int64)

    if source_points.ndim != 3 or source_points.shape[-1] != 3:
        raise ValueError("source_points must have shape (H, W, 3)")
    if target_points.shape != source_points.shape:
        raise ValueError("target_points must match source_points shape")
    height, width, _ = source_points.shape
    masks_shape = (height, width)
    if source_mask.shape != masks_shape or target_mask.shape != masks_shape:
        raise ValueError("source and target masks must have shape (H, W)")
    if (
        source_probability.shape != masks_shape
        or target_probability.shape != masks_shape
    ):
        raise ValueError("source and target probabilities must have shape (H, W)")
    if source_pose.shape != (4, 4) or target_pose.shape != (4, 4):
        raise ValueError("source_pose and target_pose must have shape (4, 4)")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")
    if sampling_stride <= 0 or not np.isfinite(relative_depth_tolerance):
        raise ValueError("projection parameters must be valid")
    if relative_depth_tolerance <= 0.0:
        raise ValueError("relative_depth_tolerance must be positive")
    if source_rows.ndim != 1 or source_columns.ndim != 1:
        raise ValueError("source_rows and source_columns must be one-dimensional")
    if (
        np.any(source_rows < 0)
        or np.any(source_rows >= height)
        or np.any(source_columns < 0)
        or np.any(source_columns >= width)
    ):
        raise ValueError("source sample indices must be in bounds")

    grid_rows, grid_columns = np.meshgrid(
        source_rows,
        source_columns,
        indexing="ij",
    )
    source_flat_all = (grid_rows * width + grid_columns).reshape(-1)
    sampled_points = source_points[grid_rows, grid_columns].reshape(-1, 3)
    sampled_mask = source_mask[grid_rows, grid_columns].reshape(-1)
    sampled_probability = source_probability[grid_rows, grid_columns].reshape(-1)
    sampled_finite = (
        sampled_mask
        & np.all(np.isfinite(sampled_points), axis=1)
        & np.isfinite(sampled_probability)
    )
    source_finite = sampled_finite & (sampled_points[:, 2] > 1e-6)
    invalid_source_samples = int(source_finite.size - np.count_nonzero(source_finite))
    nonpositive_depth_samples = int(
        np.count_nonzero(sampled_finite & (sampled_points[:, 2] <= 1e-6))
    )
    if not np.any(source_finite):
        return _empty_pair_projection(
            nonpositive_depth_samples=nonpositive_depth_samples,
            invalid_source_samples=invalid_source_samples,
        )

    source_flat = source_flat_all[source_finite]
    sampled_points = sampled_points[source_finite]
    sampled_probability = np.clip(sampled_probability[source_finite], 0.0, 1.0)
    source_homogeneous = np.concatenate(
        [sampled_points, np.ones((sampled_points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    try:
        relative = np.linalg.inv(target_pose.astype(np.float64)) @ source_pose.astype(
            np.float64
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera poses must be invertible") from exc
    target_homogeneous = (relative @ source_homogeneous.T).T
    target_xyz = target_homogeneous[:, :3]
    positive_depth = np.all(np.isfinite(target_xyz), axis=1) & (
        target_xyz[:, 2] > 1e-6
    )
    nonpositive_depth_samples += int(
        positive_depth.size - np.count_nonzero(positive_depth)
    )
    if not np.any(positive_depth):
        return _empty_pair_projection(
            nonpositive_depth_samples=nonpositive_depth_samples,
            invalid_source_samples=invalid_source_samples,
        )

    source_flat = source_flat[positive_depth]
    sampled_probability = sampled_probability[positive_depth]
    target_xyz = target_xyz[positive_depth]
    projected = target_xyz @ intrinsic.T
    denominator = projected[:, 2]
    finite_projection = np.all(np.isfinite(projected), axis=1) & (
        denominator > 1e-6
    )
    source_flat = source_flat[finite_projection]
    sampled_probability = sampled_probability[finite_projection]
    target_xyz = target_xyz[finite_projection]
    projected = projected[finite_projection]
    denominator = denominator[finite_projection]
    if source_flat.size == 0:
        return _empty_pair_projection(
            nonpositive_depth_samples=nonpositive_depth_samples,
            invalid_source_samples=invalid_source_samples,
        )

    pixel_u = np.floor(projected[:, 0] / denominator + 0.5).astype(np.int64)
    pixel_v = np.floor(projected[:, 1] / denominator + 0.5).astype(np.int64)
    projected_depth = target_xyz[:, 2]
    in_bounds = (
        (pixel_u >= 0)
        & (pixel_u < width)
        & (pixel_v >= 0)
        & (pixel_v < height)
    )
    out_of_bounds_samples = int(in_bounds.size - np.count_nonzero(in_bounds))
    source_flat = source_flat[in_bounds]
    sampled_probability = sampled_probability[in_bounds]
    projected_depth = projected_depth[in_bounds]
    pixel_u = pixel_u[in_bounds]
    pixel_v = pixel_v[in_bounds]
    projected_samples = int(source_flat.size)
    if projected_samples == 0:
        return _empty_pair_projection(
            projected_samples=0,
            out_of_bounds_samples=out_of_bounds_samples,
            nonpositive_depth_samples=nonpositive_depth_samples,
            invalid_source_samples=invalid_source_samples,
        )

    target_flat = pixel_v * width + pixel_u
    order = np.lexsort((source_flat, projected_depth, target_flat))
    source_flat = source_flat[order]
    sampled_probability = sampled_probability[order]
    projected_depth = projected_depth[order]
    target_flat = target_flat[order]
    first = np.r_[True, target_flat[1:] != target_flat[:-1]]
    winners_source = source_flat[first]
    winners_target = target_flat[first]
    winners_probability = sampled_probability[first]
    winners_depth = projected_depth[first]
    winner_samples = int(winners_source.size)
    occluded_samples = projected_samples - winner_samples

    target_xyz_flat = target_points.reshape(-1, 3)
    target_mask_flat = target_mask.reshape(-1)
    target_probability_flat = target_probability.reshape(-1)
    valid_target_flat = np.flatnonzero(target_mask_flat)
    valid_target_points = target_xyz_flat[valid_target_flat]
    valid_target_probability = target_probability_flat[valid_target_flat]
    valid_target = (
        np.all(np.isfinite(valid_target_points), axis=1)
        & (valid_target_points[:, 2] > 1e-6)
        & np.isfinite(valid_target_probability)
    )
    valid_target_flat = valid_target_flat[valid_target]
    if valid_target_flat.size:
        valid_target_v = valid_target_flat // width
        valid_target_u = valid_target_flat % width
        target_cells = np.unique(
            (valid_target_v // sampling_stride)
            * ((width + sampling_stride - 1) // sampling_stride)
            + (valid_target_u // sampling_stride)
        )
    else:
        target_cells = np.empty(0, dtype=np.int64)

    winner_target_points = target_xyz_flat[winners_target]
    winner_target_probability = target_probability_flat[winners_target]
    winner_target_valid = (
        target_mask_flat[winners_target]
        & np.all(np.isfinite(winner_target_points), axis=1)
        & (winner_target_points[:, 2] > 1e-6)
        & np.isfinite(winner_target_probability)
    )
    target_invalid_samples = int(winner_samples - np.count_nonzero(winner_target_valid))
    winner_target_depth = winner_target_points[:, 2]
    denominator_depth = np.maximum(np.abs(winner_target_depth), 1e-6)
    relative_error = np.full(winner_samples, np.inf, dtype=np.float64)
    relative_error[winner_target_valid] = np.abs(
        winners_depth[winner_target_valid] - winner_target_depth[winner_target_valid]
    ) / denominator_depth[winner_target_valid]
    consistent = winner_target_valid & (relative_error < relative_depth_tolerance)
    depth_rejected_samples = int(
        np.count_nonzero(winner_target_valid & ~consistent)
    )
    if np.any(consistent):
        consistent_target_v = winners_target[consistent] // width
        consistent_target_u = winners_target[consistent] % width
        target_cells_for_winners = np.unique(
            (consistent_target_v // sampling_stride)
            * ((width + sampling_stride - 1) // sampling_stride)
            + (consistent_target_u // sampling_stride)
        )
        confidence_weight = np.sqrt(
            winners_probability[consistent]
            * np.clip(winner_target_probability[consistent], 0.0, 1.0)
        )
        depth_weight = np.maximum(
            0.0,
            1.0 - relative_error[consistent] / relative_depth_tolerance,
        )
        correspondence_weights = confidence_weight * depth_weight
        source_indices = winners_source[consistent]
        target_indices = winners_target[consistent]
    else:
        target_cells_for_winners = np.empty(0, dtype=np.int64)
        confidence_weight = np.empty(0, dtype=np.float64)
        correspondence_weights = np.empty(0, dtype=np.float64)
        source_indices = np.empty(0, dtype=np.int64)
        target_indices = np.empty(0, dtype=np.int64)

    coverage = (
        float(target_cells_for_winners.size / target_cells.size)
        if target_cells.size
        else 0.0
    )
    geometry_denominator = int(np.count_nonzero(winner_target_valid))
    geometry_ratio = (
        float(np.count_nonzero(consistent) / geometry_denominator)
        if geometry_denominator
        else 0.0
    )
    mean_confidence = (
        float(np.mean(confidence_weight)) if confidence_weight.size else 0.0
    )
    score = coverage * float(np.sqrt(geometry_ratio * mean_confidence))
    components = (coverage, geometry_ratio, mean_confidence, score)
    coverage, geometry_ratio, mean_confidence, score = tuple(
        float(np.clip(value, 0.0, 1.0)) for value in components
    )
    return PairProjection(
        source_indices,
        target_indices,
        correspondence_weights,
        coverage=coverage,
        geometry_ratio=geometry_ratio,
        mean_confidence=mean_confidence,
        score=score,
        projected_samples=projected_samples,
        occluded_samples=occluded_samples,
        depth_rejected_samples=depth_rejected_samples,
        target_invalid_samples=target_invalid_samples,
        out_of_bounds_samples=out_of_bounds_samples,
        nonpositive_depth_samples=nonpositive_depth_samples,
        invalid_source_samples=invalid_source_samples,
    )


def select_window_references(
    *,
    config: SegmentationConfig,
    point_maps: torch.Tensor,
    camera_poses: torch.Tensor,
    confidence: torch.Tensor,
    reference_intrinsic: torch.Tensor | None,
) -> WindowReferenceSelection:
    frame_count, _, _ = _validate_window_shapes(
        point_maps,
        camera_poses,
        confidence,
    )
    if frame_count == 1:
        return _fallback_selection(frame_count, "single_frame")
    if reference_intrinsic is None:
        return _fallback_selection(frame_count, "missing_intrinsic")

    intrinsic = _validate_intrinsic(reference_intrinsic)
    if intrinsic is None:
        return _fallback_selection(frame_count, "invalid_intrinsic")

    points = _tensor_numpy(point_maps, dtype=np.float32)
    poses = _tensor_numpy(camera_poses, dtype=np.float64)
    logits = _tensor_numpy(confidence, dtype=np.float32)
    valid_points = np.all(np.isfinite(points), axis=-1) & (points[..., 2] > 1e-6)
    if not np.any(valid_points) or not _validate_pose_geometry(poses):
        return _fallback_selection(frame_count, "invalid_geometry")
    if not _intrinsic_compatible(
        points,
        intrinsic,
        config.window_reference.sampling_stride,
    ):
        return _fallback_selection(frame_count, "intrinsic_incompatible")

    selected_masks: list[np.ndarray] = []
    qualities: list[float] = []
    probabilities: list[np.ndarray] = []
    for frame_index in range(frame_count):
        mask, quality = _confidence_mask_and_quality(
            points[frame_index],
            logits[frame_index],
            keep_ratio=config.confidence_keep_ratio,
            method=config.confidence_quantile_method,
        )
        selected_masks.append(mask)
        qualities.append(quality)
        probabilities.append(_confidence_probability(logits[frame_index]))
    if not any(quality > 0.0 for quality in qualities):
        return _fallback_selection(frame_count, "no_reference")

    height, width = points.shape[1:3]
    source_rows = _centered_axis(height, config.window_reference.sampling_stride)
    source_columns = _centered_axis(width, config.window_reference.sampling_stride)

    def evaluate(source: int, target: int) -> PairProjection:
        return project_pair(
            source_points=points[source],
            target_points=points[target],
            source_pose=poses[source],
            target_pose=poses[target],
            source_mask=selected_masks[source],
            target_mask=selected_masks[target],
            source_probability=probabilities[source],
            target_probability=probabilities[target],
            intrinsic=intrinsic,
            source_rows=source_rows,
            source_columns=source_columns,
            sampling_stride=config.window_reference.sampling_stride,
            relative_depth_tolerance=(
                config.window_reference.relative_depth_tolerance
            ),
        )

    selection = _select_references(
        qualities=np.asarray(qualities, dtype=np.float64),
        frame_count=frame_count,
        evaluate=evaluate,
        config=config.window_reference,
    )
    if not selection.indices:
        return _fallback_selection(frame_count, "no_reference")
    return selection


class WindowReferenceSelector:
    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self.window_reference = config.window_reference

    def select(
        self,
        *,
        point_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        confidence: torch.Tensor,
        reference_intrinsic: torch.Tensor | None,
    ) -> WindowReferenceSelection:
        return select_window_references(
            config=self.config,
            point_maps=point_maps,
            camera_poses=camera_poses,
            confidence=confidence,
            reference_intrinsic=reference_intrinsic,
        )
