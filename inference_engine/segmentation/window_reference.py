from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol

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
class _PairSummary:
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


def _pair_summary(projection: object) -> _PairSummary:
    return _PairSummary(
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


@dataclass(frozen=True)
class _RegionMapping:
    source_label: int
    unique_hits: int
    coverage: float
    purity: float
    support: float


def _aggregate_region_evidence(
    labels: np.ndarray,
    evidence: Iterable[tuple[int, int, float]],
) -> tuple[dict[int, dict[int, float]], dict[int, set[int]]]:
    """Group sparse evidence once by target region and source label."""

    pair_weights_by_region: dict[int, dict[int, float]] = {}
    target_hits: dict[int, set[int]] = {}
    flat_size = int(labels.size)
    for item in evidence:
        try:
            target_pixel, source_label, weight = item
            target_pixel = int(target_pixel)
            source_label = int(source_label)
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "region evidence must contain pixel, label, weight"
            ) from exc
        if target_pixel < 0 or target_pixel >= flat_size:
            continue
        if not np.isfinite(weight) or weight < 0.0:
            continue
        target_region = int(labels.flat[target_pixel])
        target_hits.setdefault(target_region, set()).add(target_pixel)
        source_weights = pair_weights_by_region.setdefault(target_region, {})
        source_weights[source_label] = source_weights.get(source_label, 0.0) + weight
    return pair_weights_by_region, target_hits


def _dominant_region_mappings(
    target_labels: np.ndarray,
    evidence: Iterable[tuple[int, int, float]],
    *,
    sampling_stride: int,
    min_region_correspondences: int,
    min_region_coverage: float,
    min_region_purity: float,
) -> dict[int, _RegionMapping]:
    """Return reliable dominant source labels for sparse target evidence."""

    labels = np.asarray(target_labels)
    if labels.ndim != 2:
        raise ValueError("target_labels must be two-dimensional")
    if sampling_stride <= 0 or min_region_correspondences <= 0:
        raise ValueError("region mapping integer thresholds must be positive")

    region_area = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    pair_weights_by_region, target_hits = _aggregate_region_evidence(
        labels,
        evidence,
    )

    mappings: dict[int, _RegionMapping] = {}
    for target_region, area in region_area.items():
        hits = len(target_hits.get(target_region, ()))
        if hits < min_region_correspondences:
            continue
        source_weights = pair_weights_by_region.get(target_region, {})
        total_weight = float(sum(source_weights.values()))
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            continue
        dominant_weight = max(source_weights.values())
        dominant_label = min(
            source_label
            for source_label, weight in source_weights.items()
            if weight == dominant_weight
        )
        coverage = float(
            min(1.0, hits * float(sampling_stride**2) / float(area))
        )
        purity = float(dominant_weight / total_weight)
        support = float(min(coverage, purity))
        if coverage < min_region_coverage or purity < min_region_purity:
            continue
        mappings[target_region] = _RegionMapping(
            source_label=int(dominant_label),
            unique_hits=hits,
            coverage=coverage,
            purity=purity,
            support=support,
        )
    return mappings


@dataclass(frozen=True)
class _MergeEdge:
    label_pair: tuple[int, int]
    merge_evidence: float
    separate_evidence: float
    ratio: float
    separate_references: tuple[int, ...] = ()


def _adjacent_region_edges(labels: np.ndarray) -> tuple[tuple[int, int], ...]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be two-dimensional")
    pairs: list[np.ndarray] = []
    for left, right in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        different = left != right
        if np.any(different):
            lower = np.minimum(left[different], right[different]).astype(np.int64)
            higher = np.maximum(left[different], right[different]).astype(np.int64)
            pairs.append(np.column_stack((lower, higher)))
    if not pairs:
        return ()
    unique_pairs = np.unique(np.concatenate(pairs, axis=0), axis=0)
    return tuple((int(pair[0]), int(pair[1])) for pair in unique_pairs)


def _merge_vote_edges(
    target_labels: np.ndarray,
    reference_mappings: Iterable[
        tuple[float, Mapping[int, _RegionMapping]]
        | tuple[int, float, Mapping[int, _RegionMapping]]
    ],
    *,
    merge_vote_threshold: float,
) -> tuple[_MergeEdge, ...]:
    """Build sorted adjacent edges from reliable per-reference mappings."""

    merge_totals: dict[tuple[int, int], float] = {}
    separate_totals: dict[tuple[int, int], float] = {}
    separate_references: dict[tuple[int, int], set[int]] = {}
    adjacent_edges = _adjacent_region_edges(target_labels)
    for fallback_index, item in enumerate(reference_mappings):
        if len(item) == 3:
            reference_index, pair_score, mappings = item
        else:
            reference_index = fallback_index
            pair_score, mappings = item
        reference_index = int(reference_index)
        try:
            score = float(pair_score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score) or score <= 0.0:
            continue
        for label_pair in adjacent_edges:
            mapping_a = mappings.get(label_pair[0])
            mapping_b = mappings.get(label_pair[1])
            if mapping_a is None or mapping_b is None:
                continue
            support = min(float(mapping_a.support), float(mapping_b.support))
            if not np.isfinite(support) or support <= 0.0:
                continue
            evidence = score * support
            totals = (
                merge_totals if mapping_a.source_label == mapping_b.source_label
                else separate_totals
            )
            totals[label_pair] = totals.get(label_pair, 0.0) + evidence
            if mapping_a.source_label != mapping_b.source_label:
                separate_references.setdefault(label_pair, set()).add(reference_index)

    eligible: list[_MergeEdge] = []
    for label_pair in adjacent_edges:
        merge_evidence = float(merge_totals.get(label_pair, 0.0))
        separate_evidence = float(separate_totals.get(label_pair, 0.0))
        if merge_evidence <= 0.0:
            continue
        denominator = merge_evidence + separate_evidence
        ratio = float(merge_evidence / denominator)
        if ratio < merge_vote_threshold:
            continue
        eligible.append(
            _MergeEdge(
                label_pair=label_pair,
                merge_evidence=merge_evidence,
                separate_evidence=separate_evidence,
                ratio=ratio,
                separate_references=tuple(
                    sorted(separate_references.get(label_pair, ()))
                ),
            )
        )
    eligible.sort(
        key=lambda edge: (
            -edge.ratio,
            -edge.merge_evidence,
            edge.label_pair[0],
            edge.label_pair[1],
        )
    )
    return tuple(eligible)


def _merge_region_components(
    region_count: int,
    edges: Iterable[_MergeEdge],
    signatures: Mapping[int, Mapping[int, int]],
) -> tuple[np.ndarray, int, int]:
    """Union compatible region signatures and return roots and edge counts."""

    if region_count < 0:
        raise ValueError("region_count must be non-negative")
    parent = list(range(region_count))
    component_signatures: dict[int, dict[int, set[int]]] = {}
    for region in range(region_count):
        normalized: dict[int, set[int]] = {}
        for reference, source_label in signatures.get(region, {}).items():
            if isinstance(source_label, (set, frozenset, tuple, list)):
                labels = {int(label) for label in source_label}
            else:
                labels = {int(source_label)}
            if labels:
                normalized[int(reference)] = labels
        component_signatures[region] = normalized

    def find(region: int) -> int:
        root = region
        while parent[root] != root:
            root = parent[root]
        while parent[region] != region:
            next_region = parent[region]
            parent[region] = root
            region = next_region
        return root

    ordered_edges = sorted(
        edges,
        key=lambda edge: (
            -float(edge.ratio),
            -float(edge.merge_evidence),
            edge.label_pair[0],
            edge.label_pair[1],
        ),
    )
    accepted_edges = 0
    conflict_edges = 0
    for edge in ordered_edges:
        left, right = edge.label_pair
        if not (0 <= left < region_count and 0 <= right < region_count):
            raise ValueError("merge edge region labels must be in bounds")
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        left_signature = component_signatures[left_root]
        right_signature = component_signatures[right_root]
        incompatible = False
        for reference in left_signature.keys() & right_signature.keys():
            left_labels = left_signature[reference]
            right_labels = right_signature[reference]
            if len(left_labels) > 1 or len(right_labels) > 1:
                incompatible = True
                break
            if left_labels != right_labels and reference not in edge.separate_references:
                incompatible = True
                break
        if incompatible:
            conflict_edges += 1
            continue
        root = min(left_root, right_root)
        child = max(left_root, right_root)
        parent[child] = root
        merged_signature = {
            reference: set(labels)
            for reference, labels in left_signature.items()
        }
        for reference, labels in right_signature.items():
            merged_signature.setdefault(reference, set()).update(labels)
        component_signatures[root] = merged_signature
        component_signatures[child] = {}
        accepted_edges += 1

    roots = np.fromiter((find(region) for region in range(region_count)), dtype=np.intp)
    return roots, accepted_edges, conflict_edges


def _projection_region_mappings(
    *,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    projection: object,
    sampling_stride: int,
    min_region_correspondences: int,
    min_region_coverage: float,
    min_region_purity: float,
) -> dict[int, _RegionMapping]:
    source_flat = np.asarray(
        getattr(projection, "source_flat_indices", ()), dtype=np.int64
    ).reshape(-1)
    target_flat = np.asarray(
        getattr(projection, "target_flat_indices", ()), dtype=np.int64
    ).reshape(-1)
    weights = np.asarray(
        getattr(projection, "correspondence_weights", ()), dtype=np.float64
    ).reshape(-1)
    if not (source_flat.size == target_flat.size == weights.size):
        raise ValueError("pair projection arrays must have equal length")
    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)
    evidence = (
        (int(target_pixel), int(source_labels.flat[source_pixel]), float(weight))
        for source_pixel, target_pixel, weight in zip(
            source_flat,
            target_flat,
            weights,
            strict=True,
        )
        if 0 <= int(source_pixel) < source_labels.size
        and 0 <= int(target_pixel) < target_labels.size
    )
    return _dominant_region_mappings(
        target_labels,
        evidence,
        sampling_stride=sampling_stride,
        min_region_correspondences=min_region_correspondences,
        min_region_coverage=min_region_coverage,
        min_region_purity=min_region_purity,
    )


@dataclass(frozen=True)
class _ReferenceSelection:
    indices: tuple[int, ...]
    best_scores: np.ndarray
    pair_projections: tuple[tuple[int, int, object], ...] = ()
    rejected_indices: tuple[int, ...] = ()
    coverage_ratio: float = 0.0
    diagnostics: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scores = np.asarray(self.best_scores, dtype=np.float64).copy()
        scores.setflags(write=False)
        object.__setattr__(self, "best_scores", scores)
        object.__setattr__(self, "indices", tuple(int(index) for index in self.indices))
        object.__setattr__(
            self,
            "rejected_indices",
            tuple(int(index) for index in self.rejected_indices),
        )
        object.__setattr__(self, "pair_projections", tuple(self.pair_projections))
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )


def _projection_score(projection: object) -> float:
    score = getattr(projection, "score", projection)
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("pair evaluator must return a numeric score") from exc
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _select_references(
    *,
    qualities: np.ndarray,
    frame_count: int,
    evaluate: Callable[[int, int], object],
    config: object,
) -> _ReferenceSelection:
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
        return _ReferenceSelection(
            indices=(),
            best_scores=best_scores,
            coverage_ratio=0.0,
            diagnostics={
                "evaluated_pairs": 0,
                "reliable_pairs": 0,
                "selected_count": 0,
                "rejected_count": 0,
                "coverage_ratio": 0.0,
            },
        )

    center = (frame_count - 1) / 2.0
    first = min(
        positive,
        key=lambda index: (-qualities[index], abs(index - center), index),
    )
    selected_order = [first]
    rejected_indices: list[int] = []
    reliable_summaries: list[tuple[int, int, _PairSummary]] = []
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
        candidate_summaries: list[tuple[int, int, _PairSummary]] = []
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
        for source, target, summary in candidate_summaries:
            score = candidate_scores[target]
            best_scores[target] = max(best_scores[target], score)
            if score >= config.min_reference_score:
                reliable_summaries.append((source, target, summary))
        coverage_ratio = current_coverage()

    indices = tuple(sorted(selected_order))
    diagnostics = {
        "evaluated_pairs": evaluated_pairs,
        "reliable_pairs": len(reliable_summaries),
        "selected_count": len(indices),
        "rejected_count": len(rejected_indices),
        "coverage_ratio": coverage_ratio,
    }
    return _ReferenceSelection(
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
    )


def _project_pair(
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
) -> _PairProjection:
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

        points = _tensor_numpy(point_maps, dtype=np.float32)
        poses = _tensor_numpy(camera_poses, dtype=np.float64)
        logits = _tensor_numpy(confidence, dtype=np.float32)
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
        probabilities = []
        for frame_index in range(frame_count):
            mask, quality = _confidence_mask_and_quality(
                points[frame_index],
                logits[frame_index],
                keep_ratio=self.config.confidence_keep_ratio,
                method=self.config.confidence_quantile_method,
            )
            selected_masks.append(mask)
            qualities.append(quality)
            probabilities.append(_confidence_probability(logits[frame_index]))
        if not any(quality > 0.0 for quality in qualities):
            return _fallback_results(results, "no_reference")

        height, width = points.shape[1:3]
        source_rows = _centered_axis(
            height,
            self.window_reference.sampling_stride,
        )
        source_columns = _centered_axis(
            width,
            self.window_reference.sampling_stride,
        )

        def evaluate(source: int, target: int) -> _PairProjection:
            return _project_pair(
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
                sampling_stride=self.window_reference.sampling_stride,
                relative_depth_tolerance=(
                    self.window_reference.relative_depth_tolerance
                ),
            )

        selection = _select_references(
            qualities=np.asarray(qualities, dtype=np.float64),
            frame_count=frame_count,
            evaluate=evaluate,
            config=self.window_reference,
        )
        if not selection.indices:
            return _fallback_results(results, "no_reference")

        selected_indices = set(selection.indices)
        keyframes = ",".join(str(index) for index in selection.indices)
        # Tests and low-level callers may provide already-built projections.  The
        # normal selector only returns scalar summaries, so the enabled path below
        # reprojects selected pairs one at a time and releases each array after use.
        supplied_projections_by_target: dict[int, list[tuple[int, object]]] = {}
        for source, target, projection in selection.pair_projections:
            source = int(source)
            target = int(target)
            if (
                source in selected_indices
                and source != target
                and 0 <= target < frame_count
                and all(
                    hasattr(projection, name)
                    for name in (
                        "source_flat_indices",
                        "target_flat_indices",
                        "correspondence_weights",
                    )
                )
            ):
                supplied_projections_by_target.setdefault(target, []).append(
                    (source, projection)
                )

        refined_results: list[SegmentationResult] = []
        for target in range(frame_count):
            initial_labels = np.asarray(results[target].labels)
            regions_before = int(np.unique(initial_labels).size)
            is_keyframe = target in selected_indices
            mappings: list[tuple[int, float, dict[int, _RegionMapping]]] = []
            projected_samples = 0
            occluded_samples = 0
            depth_rejected_samples = 0

            supplied = supplied_projections_by_target.get(target)
            if supplied is not None:
                projection_items: Iterable[tuple[int, object]] = sorted(
                    supplied,
                    key=lambda item: item[0],
                )
            else:
                projection_items = (
                    (source, evaluate(source, target))
                    for source in sorted(selected_indices)
                    if source != target
                )

            for source, projection in projection_items:
                projected_samples += int(
                    getattr(projection, "projected_samples", 0)
                )
                occluded_samples += int(getattr(projection, "occluded_samples", 0))
                depth_rejected_samples += int(
                    getattr(projection, "depth_rejected_samples", 0)
                )
                pair_score = _projection_score(projection)
                if not is_keyframe and pair_score >= self.window_reference.min_reference_score:
                    source_labels = np.asarray(results[source].labels)
                    region_mappings = _projection_region_mappings(
                        source_labels=source_labels,
                        target_labels=initial_labels,
                        projection=projection,
                        sampling_stride=self.window_reference.sampling_stride,
                        min_region_correspondences=(
                            self.window_reference.min_region_correspondences
                        ),
                        min_region_coverage=self.window_reference.min_region_coverage,
                        min_region_purity=self.window_reference.min_region_purity,
                    )
                    mappings.append(
                        (
                            source,
                            pair_score,
                            region_mappings,
                        )
                    )
                del projection

            reference_votes = [
                (source, pair_score, region_mappings)
                for source, pair_score, region_mappings in mappings
            ]
            candidate_edges = (
                _merge_vote_edges(
                    initial_labels,
                    reference_votes,
                    merge_vote_threshold=self.window_reference.merge_vote_threshold,
                )
                if not is_keyframe
                else ()
            )
            signatures: dict[int, dict[int, int]] = {
                int(region): {}
                for region in np.unique(initial_labels)
            }
            for source, _, region_mappings in mappings:
                for region, region_mapping in region_mappings.items():
                    signatures[int(region)][source] = region_mapping.source_label
            roots, accepted_edges, conflict_edges = (
                _merge_region_components(
                    regions_before,
                    candidate_edges,
                    signatures,
                )
                if not is_keyframe
                else (
                    np.arange(regions_before, dtype=np.intp),
                    0,
                    0,
                )
            )

            if is_keyframe:
                output_labels = initial_labels.copy()
                fallback = "none"
            else:
                region_indices = np.asarray(initial_labels, dtype=np.intp)
                output_labels = compact_labels(roots[region_indices]).astype(
                    np.intp,
                    copy=False,
                )
                fallback = (
                    "none"
                    if accepted_edges > 0
                    else (
                        "conflict_only"
                        if conflict_edges > 0
                        else "insufficient_support"
                    )
                )
            regions_after = int(np.unique(output_labels).size)
            diagnostics = {
                **dict(results[target].diagnostics),
                "window_reference_applied": bool(accepted_edges > 0),
                "window_reference_keyframes": keyframes,
                "window_reference_keyframe_count": int(len(selection.indices)),
                "window_reference_is_keyframe": bool(is_keyframe),
                "window_reference_coverage_ratio": float(
                    np.clip(selection.coverage_ratio, 0.0, 1.0)
                ),
                "window_reference_regions_before": regions_before,
                "window_reference_regions_after": regions_after,
                "window_reference_candidate_edges": int(len(candidate_edges)),
                "window_reference_accepted_edges": int(accepted_edges),
                "window_reference_conflict_edges": int(conflict_edges),
                "window_reference_projected_samples": int(projected_samples),
                "window_reference_occluded_samples": int(occluded_samples),
                "window_reference_depth_rejected_samples": int(
                    depth_rejected_samples
                ),
                "window_reference_fallback": fallback,
                "region_count": regions_after,
            }
            refined_results.append(SegmentationResult(output_labels, diagnostics))

        return refined_results


def build_window_reference_refiner(
    config: SegmentationConfig,
) -> WindowReferenceRefinement:
    if config.window_reference.enabled:
        return WindowReferenceRefiner(config)
    return DisabledWindowReferenceRefiner()
