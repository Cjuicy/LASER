"""Observation-only reproduction of validated layer-atomic union decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inference_engine.utils.geometry import build_geometry_info_np


class DiagnosticParityError(RuntimeError):
    pass


@dataclass(frozen=True)
class LayerAtomicMergeTrace:
    final_labels: np.ndarray
    atom_scales: np.ndarray
    metrics: dict[str, Any]
    pair_table: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]


def _compact(labels: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape)


def _quantiles(values: np.ndarray, *, degrees: bool = False) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if degrees:
        values = np.rad2deg(values)
    keys = (("p10", .10), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90), ("p95", .95))
    if not values.size:
        return {key: None for key, _ in keys}
    return {key: float(np.quantile(values, q)) for key, q in keys}


class _DSU:
    def __init__(self, size: int, pixels: np.ndarray):
        self.parent = np.arange(size, dtype=np.int64)
        self.atoms = np.ones(size, dtype=np.int64)
        self.pixels = pixels.copy()
        self.depth = np.zeros(size, dtype=np.int64)

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = int(self.parent[root])
        while node != root:
            parent = int(self.parent[node])
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: int, right: int) -> tuple[int, bool]:
        a, b = self.find(left), self.find(right)
        if a == b:
            return a, False
        small, large = min(a, b), max(a, b)
        self.parent[large] = small
        self.atoms[small] += self.atoms[large]
        self.pixels[small] += self.pixels[large]
        self.depth[small] = max(self.depth[small], self.depth[large]) + 1
        return small, True


def _local_spacing(point_map: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    sums = np.zeros(atoms.shape, dtype=np.float64)
    counts = np.zeros(atoms.shape, dtype=np.int32)
    for ys, xs, yt, xt in (
        (slice(None), slice(None, -1), slice(None), slice(1, None)),
        (slice(None, -1), slice(None), slice(1, None), slice(None)),
    ):
        dist = np.linalg.norm(point_map[ys, xs] - point_map[yt, xt], axis=-1)
        valid = np.isfinite(dist) & (dist > 0) & (atoms[ys, xs] == atoms[yt, xt])
        sums[ys, xs][valid] += dist[valid]
        sums[yt, xt][valid] += dist[valid]
        counts[ys, xs][valid] += 1
        counts[yt, xt][valid] += 1
    result = np.full(atoms.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def analyze_layer_atomic_merge(
    point_map: np.ndarray,
    initial_labels: np.ndarray,
    coarse_labels: np.ndarray,
    depth_merge_thresh: float,
    formal_final_labels: np.ndarray | None = None,
) -> LayerAtomicMergeTrace:
    point_map = np.asarray(point_map)
    atoms = _compact(np.asarray(initial_labels))
    coarse = np.asarray(coarse_labels)
    if point_map.ndim != 3 or point_map.shape[-1] != 3 or point_map.shape[:2] != atoms.shape:
        raise ValueError("point_map/label shapes are incompatible")
    if coarse.shape != atoms.shape:
        raise ValueError("coarse_labels shape mismatch")
    n_atoms = int(atoms.max()) + 1
    scale_sums = np.zeros(n_atoms, dtype=np.float64)
    scale_counts = np.zeros(n_atoms, dtype=np.int64)
    codes: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    endpoint_local: list[tuple[np.ndarray, np.ndarray]] = []
    local = _local_spacing(point_map, atoms)

    for pa, pb, aa, ab, la, lb in (
        (point_map[:, :-1], point_map[:, 1:], atoms[:, :-1], atoms[:, 1:], local[:, :-1], local[:, 1:]),
        (point_map[:-1], point_map[1:], atoms[:-1], atoms[1:], local[:-1], local[1:]),
    ):
        with np.errstate(invalid="ignore", over="ignore"):
            dist = np.linalg.norm(pa - pb, axis=-1)
        finite = np.isfinite(dist)
        internal = aa == ab
        good = finite & internal & (dist > 0)
        scale_sums += np.bincount(aa[good], weights=dist[good], minlength=n_atoms)
        scale_counts += np.bincount(aa[good], minlength=n_atoms)
        boundary = finite & ~internal
        left = np.minimum(aa[boundary], ab[boundary])
        right = np.maximum(aa[boundary], ab[boundary])
        codes.append(left * n_atoms + right)
        distances.append(dist[boundary])
        endpoint_local.append((
            np.where(aa[boundary] <= ab[boundary], la[boundary], lb[boundary]),
            np.where(aa[boundary] <= ab[boundary], lb[boundary], la[boundary]),
        ))

    scales = np.zeros(n_atoms, dtype=np.float64)
    np.divide(scale_sums, scale_counts, out=scales, where=scale_counts > 0)
    valid_scale = (scale_counts > 0) & np.isfinite(scales) & (scales > 0)
    all_codes = np.concatenate(codes) if codes else np.empty(0, np.int64)
    all_dist = np.concatenate(distances) if distances else np.empty(0, np.float64)
    all_ll = np.concatenate([p[0] for p in endpoint_local]) if endpoint_local else np.empty(0)
    all_lr = np.concatenate([p[1] for p in endpoint_local]) if endpoint_local else np.empty(0)
    unique_codes, inverse = np.unique(all_codes, return_inverse=True)
    pair_count = np.bincount(inverse, minlength=unique_codes.size)
    pair_gap = np.zeros(unique_codes.size, dtype=np.float64)
    np.divide(np.bincount(inverse, weights=all_dist, minlength=unique_codes.size), pair_count,
              out=pair_gap, where=pair_count > 0)
    pair_left, pair_right = unique_codes // n_atoms, unique_codes % n_atoms

    flat_atoms = atoms.reshape(-1)
    _, first = np.unique(flat_atoms, return_index=True)
    atom_coarse = np.empty(n_atoms, dtype=coarse.dtype)
    atom_coarse[np.arange(n_atoms)] = coarse.reshape(-1)[first]
    denominator = np.sqrt(scales[pair_left]) * np.sqrt(scales[pair_right])
    valid_pair = valid_scale[pair_left] & valid_scale[pair_right] & np.isfinite(pair_gap) & np.isfinite(denominator) & (denominator > 0)
    normalized = np.full(pair_gap.shape, np.inf)
    np.divide(pair_gap, denominator, out=normalized, where=valid_pair)
    same = atom_coarse[pair_left] == atom_coarse[pair_right]
    limits = np.where(same, 1.0 + depth_merge_thresh, 1.0)
    accepted = valid_pair & (normalized <= limits)

    geometry = build_geometry_info_np(depth=point_map[..., -1], points=point_map, normal_method="cross")
    normals = geometry["normal"]
    atom_normals = np.zeros((n_atoms, 3), dtype=np.float64)
    atom_depth_iqr = np.full(n_atoms, np.nan)
    atom_normal_dispersion = np.full(n_atoms, np.nan)
    for atom in range(n_atoms):
        mask = atoms == atom
        mean = np.nanmean(normals[mask], axis=0)
        norm = np.linalg.norm(mean)
        if np.isfinite(norm) and norm > 0:
            atom_normals[atom] = mean / norm
            dots_atom = np.clip(normals[mask] @ atom_normals[atom], -1.0, 1.0)
            atom_normal_dispersion[atom] = float(np.nanmedian(np.arccos(dots_atom)))
        depth_values = point_map[..., -1][mask]
        depth_values = depth_values[np.isfinite(depth_values)]
        if depth_values.size:
            atom_depth_iqr[atom] = float(np.quantile(depth_values, .75) - np.quantile(depth_values, .25))
    dots = np.sum(atom_normals[pair_left] * atom_normals[pair_right], axis=1)
    angles = np.arccos(np.clip(dots, -1, 1))
    invalid_normal = (np.linalg.norm(atom_normals[pair_left], axis=1) == 0) | (np.linalg.norm(atom_normals[pair_right], axis=1) == 0)
    angles[invalid_normal] = np.nan

    dsu = _DSU(n_atoms, np.bincount(flat_atoms, minlength=n_atoms))
    events: list[dict[str, Any]] = []
    accepted_seen = 0
    total_accepted = int(accepted.sum())
    onset: dict[str, int | None] = {"p25": None, "p50": None, "p75": None}
    for index in np.flatnonzero(accepted):
        root, merged = dsu.union(int(pair_left[index]), int(pair_right[index]))
        accepted_seen += 1
        fraction = accepted_seen / max(total_accepted, 1)
        for key, threshold in (("p25", .25), ("p50", .50), ("p75", .75)):
            if onset[key] is None and fraction >= threshold:
                onset[key] = accepted_seen
        events.append({
            "pair_index": int(index), "left_atom": int(pair_left[index]),
            "right_atom": int(pair_right[index]), "same_coarse": bool(same[index]),
            "normalized_gap": float(normalized[index]), "limit": float(limits[index]),
            "merged_new_components": merged, "root_after": int(root),
            "component_atoms_after": int(dsu.atoms[root]),
            "component_pixels_after": int(dsu.pixels[root]),
            "merge_depth_after": int(dsu.depth[root]),
        })

    roots = np.asarray([dsu.find(atom) for atom in range(n_atoms)])
    final = _compact(roots[atoms])
    if formal_final_labels is not None and not np.array_equal(final, _compact(np.asarray(formal_final_labels))):
        raise DiagnosticParityError("diagnostic analyzer differs from formal layer-atomic labels")

    pair_table: list[dict[str, Any]] = []
    for i in range(unique_codes.size):
        mask = inverse == i
        ll, lr = all_ll[mask], all_lr[mask]
        local_denom = np.sqrt(ll) * np.sqrt(lr)
        with np.errstate(divide="ignore", invalid="ignore"):
            local_norm = all_dist[mask] / local_denom
        local_norm = local_norm[np.isfinite(local_norm) & (local_denom > 0)]
        if not valid_pair[i]:
            reason = "invalid_atom_scale" if not (valid_scale[pair_left[i]] and valid_scale[pair_right[i]]) else "invalid_boundary_gap"
        else:
            reason = None
        gaps = all_dist[mask]
        pair_table.append({
            "left_atom": int(pair_left[i]), "right_atom": int(pair_right[i]),
            "same_coarse": bool(same[i]), "boundary_edges": int(pair_count[i]),
            "boundary_gap_mean": float(pair_gap[i]), "boundary_gap_median": float(np.median(gaps)),
            "boundary_gap_p90": float(np.quantile(gaps, .90)), "boundary_gap_p95": float(np.quantile(gaps, .95)),
            "normalized_gap": None if not np.isfinite(normalized[i]) else float(normalized[i]),
            "limit": float(limits[i]),
            "threshold_margin": None if not np.isfinite(normalized[i]) else float(limits[i] - normalized[i]),
            "accepted": bool(accepted[i]), "invalid_reason": reason,
            "normal_angle_deg": None if not np.isfinite(angles[i]) else float(np.rad2deg(angles[i])),
            "boundary_local_normalized_gap_median": None if not local_norm.size else float(np.median(local_norm)),
            "whole_vs_local_scale_mismatch": None if not local_norm.size or not np.isfinite(normalized[i]) else float(np.median(local_norm) - normalized[i]),
        })

    component_atoms = [int(dsu.atoms[root]) for root in np.unique(roots)]
    metrics = {
        "candidate_count": int(unique_codes.size), "accepted_count": int(accepted.sum()),
        "same_coarse_candidate_count": int(same.sum()), "same_coarse_accepted_count": int((same & accepted).sum()),
        "cross_coarse_candidate_count": int((~same).sum()), "cross_coarse_accepted_count": int((~same & accepted).sum()),
        "accepted_ratio": float(accepted.mean()) if accepted.size else None,
        "cross_coarse_merge_ratio": float((~same & accepted).sum() / (~same).sum()) if (~same).any() else None,
        "normalized_gap_quantiles": _quantiles(normalized[valid_pair]),
        "threshold_margin_quantiles": _quantiles(limits[valid_pair] - normalized[valid_pair]),
        "boundary_gap_quantiles": _quantiles(all_dist),
        "normal_angle_quantiles_deg": _quantiles(angles, degrees=True),
        "atom_scale_quantiles": _quantiles(scales[valid_scale]),
        "atom_depth_iqr_quantiles": _quantiles(atom_depth_iqr),
        "atom_normal_dispersion_quantiles_deg": _quantiles(atom_normal_dispersion, degrees=True),
        "final_component_count": int(np.unique(roots).size),
        "max_atoms_per_component": max(component_atoms) if component_atoms else 0,
        "atoms_per_component_quantiles": _quantiles(np.asarray(component_atoms)),
        "longest_merge_chain": int(max((event["merge_depth_after"] for event in events), default=0)),
        "merge_onset_events": onset,
    }
    return LayerAtomicMergeTrace(final, scales, metrics, tuple(pair_table), tuple(events))
