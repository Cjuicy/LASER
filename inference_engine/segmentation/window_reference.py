from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from inference_engine.segmentation.base import SegmentationResult
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


def _fallback_results(
    results: list[SegmentationResult],
    reason: str,
) -> list[SegmentationResult]:
    fallback_results = []
    for result in results:
        region_count = int(np.unique(result.labels).size)
        fallback_results.append(
            SegmentationResult(
                labels=result.labels.copy(),
                diagnostics={
                    **dict(result.diagnostics),
                    "region_count": region_count,
                    "window_reference_applied": False,
                    "window_reference_keyframes": "",
                    "window_reference_keyframe_count": 0,
                    "window_reference_is_keyframe": False,
                    "window_reference_coverage_ratio": 0.0,
                    "window_reference_regions_before": region_count,
                    "window_reference_regions_after": region_count,
                    "window_reference_candidate_edges": 0,
                    "window_reference_accepted_edges": 0,
                    "window_reference_conflict_edges": 0,
                    "window_reference_projected_samples": 0,
                    "window_reference_occluded_samples": 0,
                    "window_reference_depth_rejected_samples": 0,
                    "window_reference_fallback": reason,
                },
            )
        )
    return fallback_results


def _tensor_numpy(value: object, *, dtype: np.dtype) -> np.ndarray:
    try:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        return np.asarray(value, dtype=dtype).copy()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("window reference tensors must be numeric") from exc


def _validate_refine_shapes(
    results: list[SegmentationResult],
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
    if len(results) != frame_count:
        raise ValueError("results frame count must match point_maps")
    if pose_shape[0] != frame_count:
        raise ValueError("camera_poses frame count must match point_maps")
    if confidence_shape != (frame_count, height, width):
        raise ValueError("confidence shape must match point_maps")
    for result in results:
        if tuple(result.labels.shape) != (height, width):
            raise ValueError("segmentation labels shape must match point_maps")
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
class _PairProjection:
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
    winner_samples: int = 0

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

    @property
    def weights(self) -> np.ndarray:
        return self.correspondence_weights

    @property
    def source_indices(self) -> np.ndarray:
        return self.source_flat_indices

    @property
    def target_indices(self) -> np.ndarray:
        return self.target_flat_indices

    @property
    def coverage_ratio(self) -> float:
        return self.coverage

    @property
    def confidence(self) -> float:
        return self.mean_confidence

    @property
    def confidence_score(self) -> float:
        return self.mean_confidence

    @property
    def pair_score(self) -> float:
        return self.score

    @property
    def projected_count(self) -> int:
        return self.projected_samples

    @property
    def projected_sample_count(self) -> int:
        return self.projected_samples

    @property
    def occluded_count(self) -> int:
        return self.occluded_samples

    @property
    def occluded_sample_count(self) -> int:
        return self.occluded_samples

    @property
    def depth_rejected_count(self) -> int:
        return self.depth_rejected_samples

    @property
    def depth_rejected_sample_count(self) -> int:
        return self.depth_rejected_samples

    @property
    def target_invalid_count(self) -> int:
        return self.target_invalid_samples

    @property
    def invalid_target_samples(self) -> int:
        return self.target_invalid_samples

    @property
    def target_rejected_samples(self) -> int:
        return self.target_invalid_samples

    @property
    def out_of_bounds_count(self) -> int:
        return self.out_of_bounds_samples

    @property
    def negative_depth_samples(self) -> int:
        return self.nonpositive_depth_samples


def _empty_pair_projection(
    *,
    projected_samples: int = 0,
    occluded_samples: int = 0,
    depth_rejected_samples: int = 0,
    target_invalid_samples: int = 0,
    out_of_bounds_samples: int = 0,
    nonpositive_depth_samples: int = 0,
    invalid_source_samples: int = 0,
    winner_samples: int = 0,
) -> _PairProjection:
    return _PairProjection(
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
        winner_samples=winner_samples,
    )


def _project_pair(*args: object, **kwargs: object) -> _PairProjection:
    """Project one sparse source frame into one target frame deterministically.

    The canonical positional order is ``source_points, target_points,
    source_pose, target_pose, source_mask, target_mask, source_probability,
    target_probability, intrinsic, source_rows, source_columns``.  Keyword
    aliases also accept prepared frame stacks via ``points``, ``poses``,
    ``masks``, ``probabilities``, ``source_index``, and ``target_index``.
    """

    names = (
        "source_points",
        "target_points",
        "source_pose",
        "target_pose",
        "source_mask",
        "target_mask",
        "source_probability",
        "target_probability",
        "intrinsic",
        "source_rows",
        "source_columns",
    )
    values: dict[str, object] = {}
    if args:
        first_array = np.asarray(args[0])
        indexed_stack = (
            len(args) in (9, 10, 11)
            and np.isscalar(args[0])
            and np.isscalar(args[1])
            and np.asarray(args[2]).ndim == 4
        )
        stacked_frames = len(args) in (9, 10, 11) and first_array.ndim == 4
        if indexed_stack:
            # Indexed stack style: source index, target index, points,
            # poses, masks/probabilities, probabilities/masks, K, rows,
            # columns, [stride, tol].
            values.update(
                {
                    "source_index": args[0],
                    "target_index": args[1],
                    "points": args[2],
                    "poses": args[3],
                    "intrinsic": args[6],
                    "source_rows": args[7],
                    "source_columns": args[8],
                }
            )
            first_confidence, second_confidence = args[4], args[5]
            if np.asarray(first_confidence).dtype.kind == "b":
                values["masks"] = first_confidence
                values["probabilities"] = second_confidence
            else:
                values["probabilities"] = first_confidence
                values["masks"] = second_confidence
            if len(args) >= 10:
                kwargs.setdefault("sampling_stride", args[9])
            if len(args) >= 11:
                kwargs.setdefault("relative_depth_tolerance", args[10])
        elif stacked_frames:
            # Stack style: points, poses, masks/probabilities,
            # probabilities/masks, K, source index, target index, rows,
            # columns, [stride, tol].
            values.update(
                {
                    "points": args[0],
                    "poses": args[1],
                    "intrinsic": args[4],
                    "source_index": args[5],
                    "target_index": args[6],
                    "source_rows": args[7],
                    "source_columns": args[8],
                }
            )
            first_confidence, second_confidence = args[2], args[3]
            if np.asarray(first_confidence).dtype.kind == "b":
                values["masks"] = first_confidence
                values["probabilities"] = second_confidence
            else:
                values["probabilities"] = first_confidence
                values["masks"] = second_confidence
            if len(args) >= 10:
                kwargs.setdefault("sampling_stride", args[9])
            if len(args) >= 11:
                kwargs.setdefault("relative_depth_tolerance", args[10])
        elif len(args) in (11, 12, 13):
            values.update(dict(zip(names, args[:11], strict=False)))
            if len(args) >= 12:
                kwargs.setdefault("sampling_stride", args[11])
            if len(args) >= 13:
                kwargs.setdefault("relative_depth_tolerance", args[12])
        else:
            raise TypeError("unsupported _project_pair positional arguments")
    values.update(kwargs)

    points = values.pop("points", None)
    poses = values.pop("poses", values.pop("camera_poses", None))
    masks = values.pop("masks", values.pop("selected_masks", None))
    probabilities = values.pop(
        "probabilities",
        values.pop("confidence_probabilities", None),
    )
    source_index = values.pop(
        "source_index",
        values.pop("source_frame", values.pop("source_frame_index", None)),
    )
    target_index = values.pop(
        "target_index",
        values.pop("target_frame", values.pop("target_frame_index", None)),
    )

    if values.get("intrinsic") is None:
        values["intrinsic"] = values.pop(
            "K",
            values.pop("reference_intrinsic", None),
        )
    if values.get("source_rows") is None:
        values["source_rows"] = values.pop(
            "rows",
            values.pop("sample_rows", values.pop("sampled_rows", None)),
        )
    if values.get("source_columns") is None:
        values["source_columns"] = values.pop(
            "columns",
            values.pop(
                "sample_columns",
                values.pop("sampled_columns", None),
            ),
        )
    if values.get("sampling_stride") is None:
        values["sampling_stride"] = values.pop("stride", None)
    if values.get("relative_depth_tolerance") is None:
        values["relative_depth_tolerance"] = values.pop("depth_tolerance", None)
    if values.get("source_probability") is None:
        values["source_probability"] = values.pop(
            "source_probabilities",
            values.pop(
                "source_confidence_probability",
                values.pop(
                    "source_confidence_prob",
                    values.pop(
                        "source_probs",
                        values.pop("source_confidence", None),
                    ),
                ),
            ),
        )
    if values.get("source_mask") is None:
        values["source_mask"] = values.pop(
            "source_valid_mask",
            values.pop("source_high_confidence_mask", None),
        )
    if values.get("target_mask") is None:
        values["target_mask"] = values.pop(
            "target_valid_mask",
            values.pop("target_high_confidence_mask", None),
        )
    if values.get("target_probability") is None:
        values["target_probability"] = values.pop(
            "target_probabilities",
            values.pop(
                "target_confidence_probability",
                values.pop(
                    "target_confidence_prob",
                    values.pop(
                        "target_probs",
                        values.pop("target_confidence", None),
                    ),
                ),
            ),
        )

    sample_indices = values.pop(
        "sample_indices",
        values.pop("source_sample_indices", None),
    )
    if (
        values.get("source_rows") is None
        and values.get("source_columns") is None
        and sample_indices is not None
    ):
        try:
            values["source_rows"], values["source_columns"] = sample_indices
        except (TypeError, ValueError) as exc:
            raise TypeError("sample_indices must contain rows and columns") from exc

    if points is not None:
        if source_index is None or target_index is None:
            raise TypeError("stack projection requires source and target indices")
        points_array = np.asarray(points)
        values["source_points"] = points_array[int(source_index)]
        values["target_points"] = points_array[int(target_index)]
    if poses is not None:
        if source_index is None or target_index is None:
            raise TypeError("stack projection requires source and target indices")
        poses_array = np.asarray(poses)
        values["source_pose"] = poses_array[int(source_index)]
        values["target_pose"] = poses_array[int(target_index)]
    if masks is not None:
        masks_array = np.asarray(masks)
        if source_index is not None and target_index is not None:
            values["source_mask"] = masks_array[int(source_index)]
            values["target_mask"] = masks_array[int(target_index)]
        elif masks_array.ndim == 3 and masks_array.shape[0] == 2:
            values["source_mask"], values["target_mask"] = masks_array
    if probabilities is not None:
        probabilities_array = np.asarray(probabilities)
        if source_index is not None and target_index is not None:
            values["source_probability"] = probabilities_array[int(source_index)]
            values["target_probability"] = probabilities_array[int(target_index)]
        elif probabilities_array.ndim == 3 and probabilities_array.shape[0] == 2:
            values["source_probability"], values["target_probability"] = (
                probabilities_array
            )

    required = (
        "source_points",
        "target_points",
        "source_pose",
        "target_pose",
        "source_mask",
        "target_mask",
        "source_probability",
        "target_probability",
        "intrinsic",
        "source_rows",
        "source_columns",
        "sampling_stride",
        "relative_depth_tolerance",
    )
    missing = [name for name in required if values.get(name) is None]
    if missing:
        raise TypeError(f"missing _project_pair arguments: {', '.join(missing)}")

    source_points = np.asarray(values["source_points"], dtype=np.float64)
    target_points = np.asarray(values["target_points"], dtype=np.float64)
    source_pose = np.asarray(values["source_pose"], dtype=np.float64)
    target_pose = np.asarray(values["target_pose"], dtype=np.float64)
    raw_source_mask = np.asarray(values["source_mask"])
    raw_target_mask = np.asarray(values["target_mask"])
    raw_source_probability = np.asarray(values["source_probability"])
    raw_target_probability = np.asarray(values["target_probability"])
    if (
        raw_source_mask.dtype.kind != "b"
        and raw_source_probability.dtype.kind == "b"
    ):
        source_mask = raw_source_probability.astype(bool)
        target_mask = raw_target_probability.astype(bool)
        source_probability = raw_source_mask.astype(np.float64)
        target_probability = raw_target_mask.astype(np.float64)
    else:
        source_mask = raw_source_mask.astype(bool)
        target_mask = raw_target_mask.astype(bool)
        source_probability = raw_source_probability.astype(np.float64)
        target_probability = raw_target_probability.astype(np.float64)
    intrinsic = np.asarray(values["intrinsic"], dtype=np.float64)
    source_rows = np.asarray(values["source_rows"], dtype=np.int64)
    source_columns = np.asarray(values["source_columns"], dtype=np.int64)
    sampling_stride = int(values["sampling_stride"])
    relative_depth_tolerance = float(values["relative_depth_tolerance"])

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
    source_finite = (
        sampled_finite
        & (sampled_points[:, 2] > 1e-6)
    )
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
    pixel_u = pixel_u[order]
    pixel_v = pixel_v[order]
    target_flat = target_flat[order]
    first = np.r_[True, target_flat[1:] != target_flat[:-1]]
    winners_source = source_flat[first]
    winners_target = target_flat[first]
    winners_probability = sampled_probability[first]
    winners_depth = projected_depth[first]
    winners_u = pixel_u[first]
    winners_v = pixel_v[first]
    winner_samples = int(winners_source.size)
    occluded_samples = projected_samples - winner_samples

    target_flat_all = np.arange(height * width, dtype=np.int64)
    target_xyz_flat = target_points.reshape(-1, 3)
    target_probability_flat = target_probability.reshape(-1)
    target_geometry = (
        target_mask.reshape(-1)
        & np.all(np.isfinite(target_xyz_flat), axis=1)
        & (target_xyz_flat[:, 2] > 1e-6)
        & np.isfinite(target_probability_flat)
    )
    target_geometry = target_geometry.reshape(height, width)
    valid_target_flat = target_flat_all[target_geometry.reshape(-1)]
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

    winner_target_valid = target_geometry[winners_v, winners_u]
    target_invalid_samples = int(winner_samples - np.count_nonzero(winner_target_valid))
    winner_target_depth = target_points[winners_v, winners_u, 2]
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
        target_cells_for_winners = np.unique(
            (winners_v[consistent] // sampling_stride)
            * ((width + sampling_stride - 1) // sampling_stride)
            + (winners_u[consistent] // sampling_stride)
        )
        confidence_weight = np.sqrt(
            winners_probability[consistent]
            * np.clip(target_probability[winners_v[consistent], winners_u[consistent]], 0.0, 1.0)
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
    return _PairProjection(
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
        winner_samples=winner_samples,
    )


class WindowReferenceRefinement(Protocol):
    enabled: bool

    def refine(
        self,
        results: list[SegmentationResult],
        *,
        point_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        confidence: torch.Tensor,
        reference_intrinsic: torch.Tensor | None,
    ) -> list[SegmentationResult]:
        ...


class DisabledWindowReferenceRefiner:
    enabled = False

    def refine(
        self,
        results: list[SegmentationResult],
        *,
        point_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        confidence: torch.Tensor,
        reference_intrinsic: torch.Tensor | None,
    ) -> list[SegmentationResult]:
        del point_maps, camera_poses, confidence, reference_intrinsic
        return results


class WindowReferenceRefiner:
    enabled = True

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self.window_reference = config.window_reference

    def refine(
        self,
        results: list[SegmentationResult],
        *,
        point_maps: torch.Tensor,
        camera_poses: torch.Tensor,
        confidence: torch.Tensor,
        reference_intrinsic: torch.Tensor | None,
    ) -> list[SegmentationResult]:
        frame_count, _, _ = _validate_refine_shapes(
            results,
            point_maps,
            camera_poses,
            confidence,
        )
        if frame_count == 1:
            return _fallback_results(results, "single_frame")
        if reference_intrinsic is None:
            return _fallback_results(results, "missing_intrinsic")

        intrinsic = _validate_intrinsic(reference_intrinsic)
        if intrinsic is None:
            return _fallback_results(results, "invalid_intrinsic")

        points = _tensor_numpy(point_maps, dtype=np.float64)
        poses = _tensor_numpy(camera_poses, dtype=np.float64)
        logits = _tensor_numpy(confidence, dtype=np.float64)
        valid_points = np.all(np.isfinite(points), axis=-1) & (
            points[..., 2] > 1e-6
        )
        if not np.any(valid_points):
            return _fallback_results(results, "invalid_geometry")
        if not _validate_pose_geometry(poses):
            return _fallback_results(results, "invalid_geometry")
        if not _intrinsic_compatible(
            points,
            intrinsic,
            self.window_reference.sampling_stride,
        ):
            return _fallback_results(results, "intrinsic_incompatible")

        selected_masks = []
        qualities = []
        for frame_index in range(frame_count):
            mask, quality = _confidence_mask_and_quality(
                points[frame_index],
                logits[frame_index],
                keep_ratio=self.config.confidence_keep_ratio,
                method=self.config.confidence_quantile_method,
            )
            selected_masks.append(mask)
            qualities.append(quality)
        del selected_masks, intrinsic, poses
        if not any(quality > 0.0 for quality in qualities):
            return _fallback_results(results, "no_reference")

        # Task 3 establishes the preparation and fallback boundary. Sparse
        # projection, adaptive reference selection, and region merging are
        # added by the following tasks.
        return _fallback_results(results, "no_reference")


def build_window_reference_refiner(
    config: SegmentationConfig,
) -> WindowReferenceRefinement:
    if config.window_reference.enabled:
        return WindowReferenceRefiner(config)
    return DisabledWindowReferenceRefiner()
