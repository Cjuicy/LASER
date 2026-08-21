from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from inference_engine.segmentation.base import SegmentationResult, compact_labels
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
    return [
        SegmentationResult(
            labels=result.labels.copy(),
            diagnostics={
                **dict(result.diagnostics),
                "window_reference_applied": False,
                "window_reference_fallback": reason,
                "window_reference_regions_before": int(
                    np.unique(result.labels).size
                ),
                "window_reference_regions_after": int(
                    np.unique(result.labels).size
                ),
            },
        )
        for result in results
    ]


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
    u = intrinsic[0, 0] * xyz[:, 0] / xyz[:, 2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * xyz[:, 1] / xyz[:, 2] + intrinsic[1, 2]
    expected_u = np.broadcast_to(columns[None, :], sampled.shape[:3])[valid]
    expected_v = np.broadcast_to(rows[:, None], sampled.shape[:3])[valid]
    return bool(
        np.all(np.abs(u - expected_u) <= 0.5)
        and np.all(np.abs(v - expected_v) <= 0.5)
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
