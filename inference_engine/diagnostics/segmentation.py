"""Read-only segmentation metrics used by the comparison pipeline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _invalid(reason: str, fields: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "invalid_reason": reason}
    result.update({field: None for field in fields})
    return result


def _compact(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, inverse = np.unique(labels, return_inverse=True)
    return values, inverse.reshape(labels.shape)


def _boundary_mask(labels: np.ndarray) -> tuple[np.ndarray, int, int]:
    mask = np.zeros(labels.shape, dtype=bool)
    right = labels[:, 1:] != labels[:, :-1]
    down = labels[1:, :] != labels[:-1, :]
    mask[:, :-1] |= right
    mask[:, 1:] |= right
    mask[:-1, :] |= down
    mask[1:, :] |= down
    edges = int(right.sum() + down.sum())
    possible = int(right.size + down.size)
    return mask, edges, possible


def summarize_labels(labels: np.ndarray, initial_labels: np.ndarray | None = None) -> dict[str, Any]:
    """Return deterministic, JSON-safe partition and component-growth metrics."""
    fields = (
        "segment_count", "largest_segment_ratio", "top_k_area_ratio",
        "area_entropy", "effective_segment_count", "boundary_edges",
        "possible_boundary_edges", "boundary_ratio", "atom_count",
        "atom_compression_ratio", "atoms_per_segment_quantiles",
        "max_atoms_per_segment", "largest_growth_ratio",
    )
    labels = np.asarray(labels)
    if labels.ndim != 2:
        return _invalid("labels_must_be_2d", fields)
    if labels.size == 0:
        return _invalid("empty_labels", fields)
    if np.issubdtype(labels.dtype, np.floating) and not np.isfinite(labels).all():
        return _invalid("non_finite_labels", fields)

    _, compact = _compact(labels)
    counts = np.bincount(compact.reshape(-1)).astype(np.float64)
    probabilities = counts / counts.sum()
    ordered = np.sort(probabilities)[::-1]
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    _, boundary_edges, possible_edges = _boundary_mask(compact)
    result: dict[str, Any] = {
        "valid": True,
        "invalid_reason": None,
        "segment_count": int(counts.size),
        "largest_segment_ratio": float(ordered[0]),
        "top_k_area_ratio": {
            str(k): float(ordered[: min(k, ordered.size)].sum()) for k in (1, 3, 5)
        },
        "area_entropy": entropy,
        "effective_segment_count": float(math.exp(entropy)),
        "boundary_edges": boundary_edges,
        "possible_boundary_edges": possible_edges,
        "boundary_ratio": float(boundary_edges / possible_edges) if possible_edges else None,
        "atom_count": None,
        "atom_compression_ratio": None,
        "atoms_per_segment_quantiles": None,
        "max_atoms_per_segment": None,
        "largest_growth_ratio": None,
    }

    if initial_labels is None:
        return result
    initial = np.asarray(initial_labels)
    if initial.shape != labels.shape:
        result["atom_invalid_reason"] = "initial_label_shape_mismatch"
        return result
    _, atoms = _compact(initial)
    atom_count = int(atoms.max()) + 1
    atoms_per_segment: list[int] = []
    growth: list[float] = []
    for segment_id in range(counts.size):
        atom_ids, atom_areas = np.unique(atoms[compact == segment_id], return_counts=True)
        atoms_per_segment.append(int(atom_ids.size))
        growth.append(float(counts[segment_id] / atom_areas.max()))
    aps = np.asarray(atoms_per_segment, dtype=np.float64)
    result.update(
        atom_count=atom_count,
        atom_compression_ratio=float(atom_count / counts.size),
        atoms_per_segment_quantiles={
            "p50": float(np.quantile(aps, 0.50)),
            "p90": float(np.quantile(aps, 0.90)),
            "p95": float(np.quantile(aps, 0.95)),
        },
        max_atoms_per_segment=int(aps.max()),
        largest_growth_ratio=float(max(growth)),
    )
    return result


def compare_labelings(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Compare two partitions without assuming compatible numeric label ids."""
    fields = (
        "variation_of_information", "overmerge_conditional_entropy",
        "oversplit_conditional_entropy", "boundary_disagreement_ratio",
        "boundary_precision", "boundary_recall", "boundary_f1",
        "best_match_pixel_agreement", "contingency_shape",
    )
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return _invalid("shape_mismatch", fields)
    if left.ndim != 2 or left.size == 0:
        return _invalid("invalid_partition_shape", fields)
    if ((np.issubdtype(left.dtype, np.floating) and not np.isfinite(left).all()) or
            (np.issubdtype(right.dtype, np.floating) and not np.isfinite(right).all())):
        return _invalid("non_finite_labels", fields)

    _, l = _compact(left)
    _, r = _compact(right)
    nl = int(l.max()) + 1
    nr = int(r.max()) + 1
    contingency = np.bincount(l.reshape(-1) * nr + r.reshape(-1), minlength=nl * nr)
    contingency = contingency.reshape(nl, nr).astype(np.float64)
    total = contingency.sum()
    pxy = contingency / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    mutual = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))

    lb, _, _ = _boundary_mask(l)
    rb, _, _ = _boundary_mask(r)
    union = np.logical_or(lb, rb).sum()
    intersection = np.logical_and(lb, rb).sum()
    precision = float(intersection / lb.sum()) if lb.any() else (1.0 if not rb.any() else 0.0)
    recall = float(intersection / rb.sum()) if rb.any() else (1.0 if not lb.any() else 0.0)
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    rows, cols = linear_sum_assignment(-contingency)
    return {
        "valid": True,
        "invalid_reason": None,
        "variation_of_information": float(hx + hy - 2 * mutual),
        "overmerge_conditional_entropy": max(0.0, hy - mutual),
        "oversplit_conditional_entropy": max(0.0, hx - mutual),
        "boundary_disagreement_ratio": float((union - intersection) / union) if union else 0.0,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "best_match_pixel_agreement": float(contingency[rows, cols].sum() / total),
        "contingency_shape": [nl, nr],
    }


def trace_segmentation_frame(
    point_map: np.ndarray,
    formal_labels: np.ndarray,
    *,
    segment_mode: str,
    depth_merge_thresh: float = 0.1,
    conf_map: np.ndarray | None = None,
    top_conf_percentile: float | None = None,
    seg_scale: float = 300,
    seg_sigma: float = 1.1,
    seg_min_size: int = 500,
    normal_method: str = "cross",
    rgb_image: np.ndarray | None = None,
    split_score_thresh: float = .10,
    split_aux_confirmation: bool = True,
) -> dict[str, Any]:
    """Recompute diagnostic stages and assert parity with an already-used result."""
    from .merge import DiagnosticParityError, analyze_layer_atomic_merge
    from inference_engine.utils.depth import segment_depth_felzenszwalb_rag_stages
    from inference_engine.utils.geometry_segmentation import segment_geometry_felzenszwalb_rag_stages
    from inference_engine.utils.layer_atomic_geometry import (
        segment_point_map_layer_atomic_split_stages,
    )

    point_map = np.asarray(point_map)
    depth = point_map[..., -1]
    arrays: dict[str, np.ndarray] = {"final_labels": np.asarray(formal_labels)}
    merge_trace = None
    split_metrics = None
    if segment_mode in {"depth", "layer_atomic"}:
        depth_conf = None if conf_map is None else np.asarray(conf_map)[None]
        initial, coarse, merge_threshold = segment_depth_felzenszwalb_rag_stages(
            depth, depth_merge_thresh, depth_conf, top_conf_percentile,
            seg_scale, seg_sigma, seg_min_size, 0,
        )
        arrays.update(initial_labels=initial, coarse_labels=coarse)
        if segment_mode == "depth":
            recomputed = coarse
        else:
            merge_trace = analyze_layer_atomic_merge(
                point_map, initial, coarse, depth_merge_thresh, formal_labels,
            )
            recomputed = merge_trace.final_labels
    elif segment_mode == "geometry":
        stages = segment_geometry_felzenszwalb_rag_stages(
            depth, conf_map=conf_map, point_map=point_map,
            top_conf_percentile=top_conf_percentile,
            depth_merge_thresh=depth_merge_thresh, seg_scale=seg_scale,
            seg_sigma=seg_sigma, seg_min_size=seg_min_size,
            normal_method=normal_method,
        )
        initial, recomputed = stages.initial_labels, stages.merged_labels
        merge_threshold = None
        arrays.update(initial_labels=initial, coarse_labels=recomputed)
    elif segment_mode == "layer_atomic_split":
        split_conf = None if conf_map is None else np.asarray(conf_map)[None]
        stages = segment_point_map_layer_atomic_split_stages(
            point_map,
            depth_merge_thresh,
            rgb_images=rgb_image,
            normal_method=normal_method,
            split_score_thresh=split_score_thresh,
            split_aux_confirmation=split_aux_confirmation,
            conf_map=split_conf,
            top_conf_percentile=top_conf_percentile,
            seg_scale=seg_scale,
            seg_sigma=seg_sigma,
            seg_min_size=seg_min_size,
            batch_idx=0,
        )
        initial, recomputed = stages.initial_labels, stages.final_labels
        merge_threshold = None
        split_metrics = stages.split_trace.diagnostics.as_dict()
        changed_mask = np.asarray(stages.split_trace.changed_mask, dtype=bool)
        if changed_mask.size:
            split_metrics["split_changed_pixel_ratio"] = float(
                np.count_nonzero(changed_mask) / changed_mask.size
            )
        arrays.update(
            initial_labels=stages.initial_labels,
            coarse_labels=stages.coarse_labels,
            pre_split_labels=stages.pre_split_labels,
            final_labels=stages.final_labels,
            atom_labels=stages.atom_labels,
            atom_scales=stages.atom_scales,
            changed_mask=changed_mask,
            split_parent_map=stages.split_trace.parent_map,
            split_child_map=stages.split_trace.child_map,
            split_score_map=stages.split_trace.score_map,
            split_decision_map=stages.split_trace.decision_map,
        )
    else:
        raise ValueError(f"unknown segment mode: {segment_mode}")
    labels_match = (
        np.array_equal(recomputed, formal_labels)
        if segment_mode == "layer_atomic_split"
        else np.array_equal(_compact(recomputed)[1], _compact(formal_labels)[1])
    )
    if not labels_match:
        raise DiagnosticParityError("diagnostic recomputation differs from formal labels")
    metrics = {
        "mode": segment_mode,
        "merge_threshold": None if merge_threshold is None else float(merge_threshold),
        "confidence_mean": (
            float(np.nanmean(conf_map))
            if conf_map is not None and np.isfinite(conf_map).any()
            else None
        ),
        "confidence_p10": (
            float(np.nanquantile(conf_map, .10))
            if conf_map is not None and np.isfinite(conf_map).any()
            else None
        ),
        "initial": summarize_labels(initial),
        "final": summarize_labels(formal_labels, initial_labels=initial),
    }
    if merge_trace is not None:
        metrics["merge"] = merge_trace.metrics
        arrays["atom_scales"] = merge_trace.atom_scales
        arrays["merge_decision"] = merge_trace.decision_map
        arrays["normalized_gap_map"] = merge_trace.normalized_gap_map
        arrays["threshold_margin_map"] = merge_trace.threshold_margin_map
        arrays["component_growth_map"] = merge_trace.component_growth_map
    if split_metrics is not None:
        metrics["split"] = split_metrics
    return {"metrics": metrics, "arrays": arrays, "merge_trace": merge_trace}
