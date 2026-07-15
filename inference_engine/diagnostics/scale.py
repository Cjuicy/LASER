"""Scale-coherence and propagation-source observations."""

from __future__ import annotations

from typing import Any

import numpy as np


def _quantiles(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in ("p50", "p75", "p90", "p95")}
    return {key: float(np.quantile(array, q)) for key, q in (("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95))}


def _weighted_scale(cache: dict[str, Any]) -> float:
    scales = np.asarray(cache.get("scale", ()), dtype=float)
    weights = np.asarray(cache.get("iou", ()), dtype=float)
    if not scales.size:
        return 1.0
    if weights.size != scales.size or not np.isfinite(weights).all() or weights.sum() <= 0:
        return float(np.nanmedian(scales))
    return float(np.dot(scales, weights / weights.sum()))


def summarize_scale_observations(graphs, *, direct_cache=None) -> dict[str, Any]:
    if direct_cache is None:
        direct_cache = [[{"scale": [], "iou": []} for _ in layer] for layer in graphs]
    shape = np.asarray(graphs[0][0].data).shape if graphs and graphs[0] else (0, 0)
    source_map = np.zeros((len(graphs), *shape), dtype=np.uint8)
    scale_map = np.ones((len(graphs), *shape), dtype=np.float32)
    dispersion_map = np.zeros((len(graphs), *shape), dtype=np.float32)
    segment_count = {"direct": 0, "propagated": 0, "fallback": 0}
    pixel_count = {"direct": 0, "propagated": 0, "fallback": 0}
    mads: list[float] = []
    iqrs: list[float] = []
    stds: list[float] = []
    support: list[int] = []
    residuals: list[float] = []
    for layer_index, layer in enumerate(graphs):
        for vertex_index, vertex in enumerate(layer):
            cache = vertex.cache
            direct = direct_cache[layer_index][vertex_index] if layer_index < len(direct_cache) else {"scale": []}
            if direct.get("scale"):
                source, code = "direct", 1
            elif cache.get("scale"):
                source, code = "propagated", 2
            else:
                source, code = "fallback", 0
            mask = np.asarray(vertex.data, dtype=bool)
            area = int(mask.sum())
            segment_count[source] += 1
            pixel_count[source] += area
            value = _weighted_scale(cache)
            source_map[layer_index][mask] = code
            scale_map[layer_index][mask] = value
            scales = np.asarray(cache.get("scale", ()), dtype=float)
            scales = scales[np.isfinite(scales) & (scales > 0)]
            if scales.size:
                logs = np.log(scales)
                median = np.median(logs)
                mad = float(np.median(np.abs(logs - median)))
                iqr = float(np.quantile(logs, .75) - np.quantile(logs, .25))
                std = float(np.std(logs))
                mads.append(mad); iqrs.append(iqr); stds.append(std)
                support.append(int(scales.size))
                residuals.extend(np.abs(logs - np.log(max(value, 1e-12))).tolist())
                dispersion_map[layer_index][mask] = mad
    total_pixels = sum(pixel_count.values())
    metrics = {
        "source_segment_count": segment_count,
        "source_pixel_count": pixel_count,
        "source_pixel_ratio": {key: (value / total_pixels if total_pixels else None) for key, value in pixel_count.items()},
        "scale_log_mad_quantiles": _quantiles(mads),
        "scale_log_iqr_quantiles": _quantiles(iqrs),
        "scale_log_std_quantiles": _quantiles(stds),
        "anchor_support_quantiles": _quantiles(support),
        "scale_residual_quantiles": _quantiles(residuals),
        "non_identity_pixel_ratio": float(np.mean(np.abs(scale_map - 1.0) > 1e-6)) if scale_map.size else None,
    }
    return {"metrics": metrics, "arrays": {"source_map": source_map, "scale_map": scale_map, "dispersion_map": dispersion_map}}
