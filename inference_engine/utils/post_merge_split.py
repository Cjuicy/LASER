"""Conservative one-pass splitting of large post-merge geometry regions.

Normals are the only source of candidate boundaries.  RGB and normalized 3D
gaps can confirm a candidate, but never create one.  Every parent is processed
once and can produce at most four leaves.
"""

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .geometry import compute_normals_cross_np, compute_normals_sobel_np


NORMAL_BARRIER_RAD = np.deg2rad(30.0)
MAX_LEAVES = 4
MIN_CHILD_FRACTION = 0.02
EPS = 1e-8


@dataclass(frozen=True)
class SplitDiagnostics:
    split_parent_count: int = 0
    split_proposed_count: int = 0
    split_accepted_count: int = 0
    split_added_regions: int = 0
    split_score_mean: float | None = None
    split_score_max: float | None = None
    split_score_quantiles: dict[str, float] | None = None
    split_score_invalid_reason: str | None = "no_scored_candidates"
    split_reject_no_markers: int = 0
    split_reject_small_child: int = 0
    split_reject_low_score: int = 0
    split_runtime_ms: float = 0.0
    split_aux_confirmation: bool = True
    split_pre_segment_count: int = 0
    split_post_segment_count: int = 0
    split_segment_count_delta: int = 0
    split_segment_growth_ratio: float | None = None
    split_changed_pixel_count: int = 0
    split_changed_pixel_ratio: float | None = None
    split_child_count_mean: float | None = None
    split_child_count_max: int | None = None
    split_min_child_fraction_mean: float | None = None
    split_min_child_fraction_min: float | None = None
    split_child_summary_invalid_reason: str | None = "no_valid_child_proposals"
    split_parent_normal_dispersion_mean: float | None = None
    split_child_normal_dispersion_mean: float | None = None
    split_normal_dispersion_gain_mean: float | None = None
    split_normal_dispersion_gain_max: float | None = None
    split_normal_dispersion_invalid_reason: str | None = "no_valid_child_proposals"
    split_largest_segment_ratio_pre: float | None = None
    split_largest_segment_ratio_post: float | None = None
    split_largest_segment_ratio_delta: float | None = None
    split_area_entropy_pre: float | None = None
    split_area_entropy_post: float | None = None
    split_area_entropy_delta: float | None = None
    split_effective_segment_count_pre: float | None = None
    split_effective_segment_count_post: float | None = None
    split_effective_segment_count_delta: float | None = None
    split_tiny_child_area_ratio: float | None = None
    split_boundary_ratio_pre: float | None = None
    split_boundary_ratio_post: float | None = None
    split_boundary_ratio_delta: float | None = None
    split_fragmentation_signal: bool = False

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SplitTrace:
    labels: np.ndarray
    diagnostics: SplitDiagnostics
    changed_mask: np.ndarray
    parent_map: np.ndarray
    child_map: np.ndarray
    score_map: np.ndarray
    decision_map: np.ndarray


def _normal_map(point_map, normal_method):
    if normal_method == "cross":
        normals = compute_normals_cross_np(point_map)
    elif normal_method == "sobel":
        normals = compute_normals_sobel_np(point_map)
    else:
        raise ValueError(f"Unknown normal_method: {normal_method}")

    valid = np.isfinite(point_map).all(axis=-1)
    valid &= np.isfinite(normals).all(axis=-1)
    valid &= np.linalg.norm(normals, axis=-1) > EPS
    try:
        import cv2

        filtered = cv2.medianBlur(normals.astype(np.float32, copy=False), 3)
    except ImportError:
        filtered = np.stack(
            [
                ndimage.median_filter(normals[..., channel], size=3, mode="nearest")
                for channel in range(3)
            ],
            axis=-1,
        )
    norm = np.linalg.norm(filtered, axis=-1, keepdims=True)
    filtered = np.divide(
        filtered,
        norm,
        out=np.zeros_like(filtered),
        where=norm > EPS,
    )
    valid &= np.linalg.norm(filtered, axis=-1) > EPS
    filtered[~valid] = 0.0
    return filtered.astype(np.float32, copy=False), valid


def _normal_edge_map(normals, valid):
    edge = np.zeros(normals.shape[:2], dtype=np.float32)
    pairs = (
        (
            normals[:, :-1],
            normals[:, 1:],
            valid[:, :-1],
            valid[:, 1:],
            (slice(None), slice(None, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            normals[:-1],
            normals[1:],
            valid[:-1],
            valid[1:],
            (slice(None, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
    )
    for left, right, valid_left, valid_right, left_slice, right_slice in pairs:
        pair_valid = valid_left & valid_right
        dot = np.sum(left * right, axis=-1)
        angle = np.zeros(dot.shape, dtype=np.float32)
        angle[pair_valid] = np.arccos(
            np.clip(np.abs(dot[pair_valid]), 0.0, 1.0)
        )
        edge[left_slice] = np.maximum(edge[left_slice], angle)
        edge[right_slice] = np.maximum(edge[right_slice], angle)
    return edge


def _normal_dispersion(normals, mask):
    selected = normals[mask]
    if selected.size == 0:
        return np.nan
    moment = np.einsum("ni,nj->ij", selected, selected) / selected.shape[0]
    return float(max(0.0, 1.0 - np.linalg.eigvalsh(moment)[-1]))


def _minimum_child_area(parent_area, seg_min_size):
    return max(int(seg_min_size), int(np.ceil(MIN_CHILD_FRACTION * parent_area)))


def _markers(parent_mask, valid, normal_edge, min_marker_area):
    seed_mask = parent_mask & valid & (normal_edge < NORMAL_BARRIER_RAD)
    components, count = ndimage.label(seed_mask)
    if count < 2:
        return np.zeros(parent_mask.shape, dtype=np.int32), count

    sizes = np.bincount(components.reshape(-1), minlength=count + 1)[1:]
    component_ids = np.arange(1, count + 1)
    eligible = sizes >= min_marker_area
    sizes = sizes[eligible]
    component_ids = component_ids[eligible]
    if component_ids.size < 2:
        return np.zeros(parent_mask.shape, dtype=np.int32), component_ids.size
    # Stable ordering makes equal-area candidates deterministic.
    order = np.lexsort((component_ids, -sizes))[:MAX_LEAVES]
    selected = component_ids[order]
    markers = np.zeros(parent_mask.shape, dtype=np.int32)
    for marker_id, component_id in enumerate(selected, start=1):
        markers[components == component_id] = marker_id
    return markers, selected.size


def _rgb_float(rgb_image, shape):
    if rgb_image is None:
        return None
    rgb = np.asarray(rgb_image)
    if rgb.shape != (*shape, 3):
        raise ValueError("rgb_image must have shape (H, W, 3)")
    rgb = rgb.astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
    if max_value > 1.0:
        rgb = rgb / 255.0
    return rgb


def _edge_fields(point_map, rgb_image, atom_labels, atom_scales):
    """Return right/down RGB and locally normalized 3D edge strengths."""
    rgb = _rgb_float(rgb_image, point_map.shape[:2])
    fields = {}
    for axis, first_slice, second_slice in (
        ("right", (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ("down", (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        first_points = point_map[first_slice]
        second_points = point_map[second_slice]
        with np.errstate(invalid="ignore", over="ignore"):
            distance = np.linalg.norm(first_points - second_points, axis=-1)
        first_atom = atom_labels[first_slice]
        second_atom = atom_labels[second_slice]
        denominator = np.sqrt(atom_scales[first_atom] * atom_scales[second_atom])
        gap = np.zeros(distance.shape, dtype=np.float32)
        valid_gap = (
            np.isfinite(distance)
            & np.isfinite(denominator)
            & (denominator > EPS)
        )
        np.divide(distance, denominator, out=gap, where=valid_gap)
        fields[f"gap_{axis}"] = gap

        if rgb is None:
            fields[f"rgb_{axis}"] = None
        else:
            with np.errstate(invalid="ignore", over="ignore"):
                rgb_edge = np.linalg.norm(rgb[first_slice] - rgb[second_slice], axis=-1)
            fields[f"rgb_{axis}"] = np.nan_to_num(
                rgb_edge, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32, copy=False)
    return fields


def _field_contrast(candidate, parent_mask, horizontal, vertical):
    if horizontal is None or vertical is None:
        return 0.0

    pair_parent_h = parent_mask[:, :-1] & parent_mask[:, 1:]
    pair_parent_v = parent_mask[:-1] & parent_mask[1:]
    boundary_h = pair_parent_h & (candidate[:, :-1] != candidate[:, 1:])
    boundary_v = pair_parent_v & (candidate[:-1] != candidate[1:])
    interior_h = pair_parent_h & (candidate[:, :-1] == candidate[:, 1:])
    interior_v = pair_parent_v & (candidate[:-1] == candidate[1:])

    boundary = np.concatenate((horizontal[boundary_h], vertical[boundary_v]))
    interior = np.concatenate((horizontal[interior_h], vertical[interior_v]))
    boundary = boundary[np.isfinite(boundary)]
    interior = interior[np.isfinite(interior)]
    if boundary.size == 0:
        return 0.0
    boundary_strength = float(np.median(boundary))
    interior_strength = float(np.quantile(interior, 0.75)) if interior.size else 0.0
    ratio = boundary_strength / max(interior_strength, EPS)
    return float(np.clip((ratio - 1.0) / (ratio + 1.0), 0.0, 1.0))


def _normal_gain(normals, valid, parent_mask, candidate):
    parent_valid = parent_mask & valid
    parent_dispersion = _normal_dispersion(normals, parent_valid)
    if not np.isfinite(parent_dispersion) or parent_dispersion <= EPS:
        return 0.0

    child_total = 0
    child_weighted_dispersion = 0.0
    for child_id in np.unique(candidate[parent_mask]):
        child_valid = parent_valid & (candidate == child_id)
        child_count = int(np.count_nonzero(child_valid))
        child_dispersion = _normal_dispersion(normals, child_valid)
        if child_count == 0 or not np.isfinite(child_dispersion):
            continue
        child_total += child_count
        child_weighted_dispersion += child_count * child_dispersion
    if child_total == 0:
        return 0.0
    after = child_weighted_dispersion / child_total
    return float(np.clip((parent_dispersion - after) / parent_dispersion, 0.0, 1.0))


def _normal_dispersion_pair(normals, valid, parent_mask, candidate):
    parent_valid = parent_mask & valid
    parent_dispersion = _normal_dispersion(normals, parent_valid)
    total = 0
    weighted = 0.0
    for child_id in np.unique(candidate[parent_mask]):
        child_valid = parent_valid & (candidate == child_id)
        count = int(np.count_nonzero(child_valid))
        dispersion = _normal_dispersion(normals, child_valid)
        if count and np.isfinite(dispersion):
            total += count
            weighted += count * dispersion
    child_dispersion = weighted / total if total else np.nan
    return float(parent_dispersion), float(child_dispersion)


def _partition_structure(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0:
        return None
    _, compact = np.unique(labels, return_inverse=True)
    compact = compact.reshape(labels.shape)
    counts = np.bincount(compact.reshape(-1)).astype(np.float64)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    right = compact[:, 1:] != compact[:, :-1]
    down = compact[1:, :] != compact[:-1, :]
    possible = int(right.size + down.size)
    return {
        "segment_count": int(counts.size),
        "largest_segment_ratio": float(probabilities.max()),
        "area_entropy": entropy,
        "effective_segment_count": float(np.exp(entropy)),
        "boundary_ratio": (
            float((right.sum() + down.sum()) / possible) if possible else None
        ),
    }


def _split_score(
    normals,
    valid,
    parent_mask,
    candidate,
    edge_fields,
    normal_gain=None,
):
    if normal_gain is None:
        normal_gain = _normal_gain(normals, valid, parent_mask, candidate)
    if edge_fields is None:
        return normal_gain
    rgb_contrast = _field_contrast(
        candidate,
        parent_mask,
        edge_fields["rgb_right"],
        edge_fields["rgb_down"],
    )
    gap_contrast = _field_contrast(
        candidate,
        parent_mask,
        edge_fields["gap_right"],
        edge_fields["gap_down"],
    )
    return normal_gain * max(rgb_contrast, gap_contrast)


def _validate_inputs(point_map, rgb_image, auto_labels, atom_labels, atom_scales):
    point_map = np.asarray(point_map)
    auto_labels = np.asarray(auto_labels)
    atom_labels = np.asarray(atom_labels)
    atom_scales = np.asarray(atom_scales, dtype=np.float64)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError("point_map must have shape (H, W, 3)")
    if auto_labels.shape != point_map.shape[:2]:
        raise ValueError("auto_labels must have shape (H, W)")
    if auto_labels.size and np.min(auto_labels) < 0:
        raise ValueError("auto_labels must be non-negative")
    if atom_labels.shape != point_map.shape[:2]:
        raise ValueError("atom_labels must have shape (H, W)")
    if atom_labels.size and (
        np.min(atom_labels) < 0 or np.max(atom_labels) >= atom_scales.size
    ):
        raise ValueError("atom_scales must contain one entry per compact atom label")
    if rgb_image is not None and np.asarray(rgb_image).shape != (*point_map.shape[:2], 3):
        raise ValueError("rgb_image must have shape (H, W, 3)")
    return (
        point_map,
        auto_labels.astype(np.intp, copy=False),
        atom_labels.astype(np.intp, copy=False),
        atom_scales,
    )


def refine_auto_regions_with_trace(
    point_map,
    rgb_image,
    auto_labels,
    atom_labels,
    atom_scales,
    seg_min_size,
    normal_method,
    split_score_thresh,
    split_aux_confirmation=True,
):
    """Split eligible Auto regions once, returning compact labels and evidence."""
    start = time.perf_counter()
    if not np.isfinite(split_score_thresh) or split_score_thresh < 0:
        raise ValueError("split_score_thresh must be finite and non-negative")
    if int(seg_min_size) <= 0:
        raise ValueError("seg_min_size must be positive")
    point_map, labels, atom_labels, atom_scales = _validate_inputs(
        point_map, rgb_image, auto_labels, atom_labels, atom_scales
    )
    normals, valid = _normal_map(point_map, normal_method)
    normal_edge = _normal_edge_map(normals, valid)
    edge_fields = None

    result = labels.copy()
    parent_map = labels.copy()
    child_map = np.full(labels.shape, -1, dtype=np.intp)
    score_map = np.full(labels.shape, np.nan, dtype=np.float32)
    decision_map = np.zeros(labels.shape, dtype=np.uint8)
    next_label = int(result.max()) + 1 if result.size else 0
    parent_count = proposed_count = accepted_count = added_regions = 0
    reject_no_markers = reject_small_child = reject_low_score = 0
    scores = []
    had_non_finite_score = False
    child_counts = []
    min_child_fractions = []
    parent_dispersions = []
    child_dispersions = []
    normal_gains = []
    had_non_finite_dispersion = False
    tiny_child_pixels = 0

    # Iterate over the original Auto labels only: newly created leaves are never revisited.
    parent_areas = np.bincount(labels.reshape(-1))
    # ndimage reserves zero for background; offset compact Auto labels by one.
    parent_crops = ndimage.find_objects(labels + 1)
    for parent_id, crop in enumerate(parent_crops):
        if crop is None:
            continue
        parent_area = int(parent_areas[parent_id])
        min_child_area = _minimum_child_area(parent_area, seg_min_size)
        if parent_area < 2 * min_child_area:
            continue
        parent_count += 1

        crop_parent = labels[crop] == parent_id
        crop_valid = valid[crop]
        crop_edge = normal_edge[crop]
        markers, marker_count = _markers(
            crop_parent,
            crop_valid,
            crop_edge,
            min_child_area,
        )
        if marker_count < 2:
            reject_no_markers += 1
            decision_map[crop][crop_parent] = 2
            continue
        proposed_count += 1

        elevation = crop_edge.copy()
        invalid_crop = crop_parent & ~crop_valid
        valid_elevation = elevation[crop_parent & crop_valid]
        invalid_level = (
            float(np.max(valid_elevation))
            if valid_elevation.size
            else NORMAL_BARRIER_RAD
        )
        elevation[invalid_crop] = invalid_level
        candidate = watershed(elevation, markers=markers, mask=crop_parent)
        child_ids, child_sizes = np.unique(
            candidate[crop_parent], return_counts=True
        )
        nonzero = child_ids > 0
        child_ids = child_ids[nonzero]
        child_sizes = child_sizes[nonzero]
        if child_ids.size:
            child_counts.append(int(child_ids.size))
            min_child_fractions.append(float(child_sizes.min() / parent_area))
        if (
            child_ids.size < 2
            or child_ids.size > MAX_LEAVES
            or np.any(child_sizes < min_child_area)
        ):
            reject_small_child += 1
            child_map[crop][crop_parent] = candidate[crop_parent]
            decision_map[crop][crop_parent] = 3
            continue

        crop_normals = normals[crop]
        normal_gain = _normal_gain(
            crop_normals, crop_valid, crop_parent, candidate
        )
        parent_dispersion, child_dispersion = _normal_dispersion_pair(
            crop_normals, crop_valid, crop_parent, candidate
        )
        if (
            np.isfinite(parent_dispersion)
            and np.isfinite(child_dispersion)
            and np.isfinite(normal_gain)
        ):
            parent_dispersions.append(parent_dispersion)
            child_dispersions.append(child_dispersion)
            normal_gains.append(float(normal_gain))
        else:
            had_non_finite_dispersion = True
        if not np.isfinite(normal_gain):
            had_non_finite_score = True
            reject_low_score += 1
            child_map[crop][crop_parent] = candidate[crop_parent]
            decision_map[crop][crop_parent] = 4
            continue
        if normal_gain < split_score_thresh:
            scores.append(normal_gain)
            reject_low_score += 1
            child_map[crop][crop_parent] = candidate[crop_parent]
            score_map[crop][crop_parent] = normal_gain
            decision_map[crop][crop_parent] = 4
            continue

        if split_aux_confirmation and edge_fields is None:
            crop_rgb = None if rgb_image is None else np.asarray(rgb_image)[crop]
            edge_fields = _edge_fields(
                point_map[crop], crop_rgb, atom_labels[crop], atom_scales
            )
        elif split_aux_confirmation:
            crop_rgb = None if rgb_image is None else np.asarray(rgb_image)[crop]
            edge_fields = _edge_fields(
                point_map[crop], crop_rgb, atom_labels[crop], atom_scales
            )
        score = _split_score(
            crop_normals,
            crop_valid,
            crop_parent,
            candidate,
            edge_fields,
            normal_gain=normal_gain,
        )
        if not np.isfinite(score):
            had_non_finite_score = True
            reject_low_score += 1
            child_map[crop][crop_parent] = candidate[crop_parent]
            decision_map[crop][crop_parent] = 4
            continue
        scores.append(score)
        if score < split_score_thresh:
            reject_low_score += 1
            child_map[crop][crop_parent] = candidate[crop_parent]
            score_map[crop][crop_parent] = score
            decision_map[crop][crop_parent] = 4
            continue

        accepted_count += 1
        tiny_child_pixels += int(
            child_sizes[(child_sizes / parent_area) < 0.05].sum()
        )
        child_map[crop][crop_parent] = candidate[crop_parent]
        score_map[crop][crop_parent] = score
        decision_map[crop][crop_parent] = 1
        # Keep one child on the parent's label and append the others globally.
        ordered_children = sorted(
            zip(child_ids.tolist(), child_sizes.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        keep_child = ordered_children[0][0]
        result_crop = result[crop]
        result_crop[crop_parent & (candidate == keep_child)] = parent_id
        for child_id, _ in ordered_children[1:]:
            result_crop[crop_parent & (candidate == child_id)] = next_label
            next_label += 1
            added_regions += 1

    runtime_ms = (time.perf_counter() - start) * 1000.0
    score_values = np.asarray(scores, dtype=np.float64)
    score_quantiles = (
        {
            f"p{int(quantile * 100)}": float(np.quantile(score_values, quantile))
            for quantile in (0.10, 0.25, 0.50, 0.75, 0.90)
        }
        if score_values.size
        else None
    )
    pre_structure = _partition_structure(labels)
    post_structure = _partition_structure(result)
    changed_pixel_count = int(np.count_nonzero(result != labels))
    pixel_count = int(labels.size)
    boundary_pre = pre_structure["boundary_ratio"] if pre_structure else None
    boundary_post = post_structure["boundary_ratio"] if post_structure else None
    boundary_delta = (
        float(boundary_post - boundary_pre)
        if boundary_pre is not None and boundary_post is not None
        else None
    )
    tiny_child_ratio = float(tiny_child_pixels / pixel_count) if pixel_count else None
    segment_delta = (
        post_structure["segment_count"] - pre_structure["segment_count"]
        if pre_structure and post_structure else 0
    )
    segment_growth = (
        float(segment_delta / pre_structure["segment_count"])
        if pre_structure and pre_structure["segment_count"] else None
    )
    diagnostics = SplitDiagnostics(
        split_parent_count=parent_count,
        split_proposed_count=proposed_count,
        split_accepted_count=accepted_count,
        split_added_regions=added_regions,
        split_score_mean=float(np.mean(score_values)) if score_values.size else None,
        split_score_max=float(np.max(score_values)) if score_values.size else None,
        split_score_quantiles=score_quantiles,
        split_score_invalid_reason=(
            None if score_values.size else (
                "non_finite_split_score" if had_non_finite_score
                else "no_scored_candidates"
            )
        ),
        split_reject_no_markers=reject_no_markers,
        split_reject_small_child=reject_small_child,
        split_reject_low_score=reject_low_score,
        split_runtime_ms=runtime_ms,
        split_aux_confirmation=bool(split_aux_confirmation),
        split_pre_segment_count=pre_structure["segment_count"] if pre_structure else 0,
        split_post_segment_count=post_structure["segment_count"] if post_structure else 0,
        split_segment_count_delta=int(segment_delta),
        split_segment_growth_ratio=segment_growth,
        split_changed_pixel_count=changed_pixel_count,
        split_changed_pixel_ratio=(
            float(changed_pixel_count / pixel_count) if pixel_count else None
        ),
        split_child_count_mean=(float(np.mean(child_counts)) if child_counts else None),
        split_child_count_max=(int(max(child_counts)) if child_counts else None),
        split_min_child_fraction_mean=(
            float(np.mean(min_child_fractions)) if min_child_fractions else None
        ),
        split_min_child_fraction_min=(
            float(min(min_child_fractions)) if min_child_fractions else None
        ),
        split_child_summary_invalid_reason=(None if child_counts else "no_valid_child_proposals"),
        split_parent_normal_dispersion_mean=(
            float(np.mean(parent_dispersions)) if parent_dispersions else None
        ),
        split_child_normal_dispersion_mean=(
            float(np.mean(child_dispersions)) if child_dispersions else None
        ),
        split_normal_dispersion_gain_mean=(
            float(np.mean(normal_gains)) if normal_gains else None
        ),
        split_normal_dispersion_gain_max=(
            float(np.max(normal_gains)) if normal_gains else None
        ),
        split_normal_dispersion_invalid_reason=(
            None if normal_gains else (
                "non_finite_normal_dispersion" if had_non_finite_dispersion
                else "no_valid_child_proposals"
            )
        ),
        split_largest_segment_ratio_pre=(
            pre_structure["largest_segment_ratio"] if pre_structure else None
        ),
        split_largest_segment_ratio_post=(
            post_structure["largest_segment_ratio"] if post_structure else None
        ),
        split_largest_segment_ratio_delta=(
            post_structure["largest_segment_ratio"] - pre_structure["largest_segment_ratio"]
            if pre_structure and post_structure else None
        ),
        split_area_entropy_pre=(pre_structure["area_entropy"] if pre_structure else None),
        split_area_entropy_post=(post_structure["area_entropy"] if post_structure else None),
        split_area_entropy_delta=(
            post_structure["area_entropy"] - pre_structure["area_entropy"]
            if pre_structure and post_structure else None
        ),
        split_effective_segment_count_pre=(
            pre_structure["effective_segment_count"] if pre_structure else None
        ),
        split_effective_segment_count_post=(
            post_structure["effective_segment_count"] if post_structure else None
        ),
        split_effective_segment_count_delta=(
            post_structure["effective_segment_count"]
            - pre_structure["effective_segment_count"]
            if pre_structure and post_structure else None
        ),
        split_tiny_child_area_ratio=tiny_child_ratio,
        split_boundary_ratio_pre=boundary_pre,
        split_boundary_ratio_post=boundary_post,
        split_boundary_ratio_delta=boundary_delta,
        split_fragmentation_signal=bool(
            (segment_growth is not None and segment_growth > 0.5)
            or (tiny_child_ratio is not None and tiny_child_ratio > 0.05)
            or (boundary_delta is not None and boundary_delta > 0.10)
        ),
    )
    return SplitTrace(
        labels=result,
        diagnostics=diagnostics,
        changed_mask=result != labels,
        parent_map=parent_map,
        child_map=child_map,
        score_map=score_map,
        decision_map=decision_map,
    )


def refine_auto_regions(*args, **kwargs):
    """Compatibility wrapper returning the established labels/diagnostics tuple."""
    trace = refine_auto_regions_with_trace(*args, **kwargs)
    return trace.labels, trace.diagnostics
