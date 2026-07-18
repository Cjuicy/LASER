"""Headless deterministic renderers for selected diagnostic cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .storage import atomic_write_json


COLORS = {
    "depth": "#087f5b",
    "geometry_baseline": "#c92a2a",
    "layer_atomic_split": "#2459a9",
}


def _hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.asarray([int(value[index:index + 2], 16) for index in (0, 2, 4)], dtype=np.uint8)


def _palette(labels: np.ndarray) -> np.ndarray:
    values = labels.astype(np.uint64, copy=False)
    hashed = values * np.uint64(11400714819323198485) + np.uint64(0x9E3779B9)
    return np.stack([
        48 + ((hashed >> np.uint64(8)) & 207),
        48 + ((hashed >> np.uint64(24)) & 207),
        48 + ((hashed >> np.uint64(40)) & 207),
    ], axis=-1).astype(np.uint8)


def _write_png(path: Path, rgb: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = np.clip(image, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(encoded.tobytes())
    partial.replace(path)
    return path


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = image[..., :3].astype(np.float64)
    if np.nanmax(image) <= 1.5:
        image *= 255
    return np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)


def _heatmap(values: np.ndarray, *, cmap=cv2.COLORMAP_TURBO) -> tuple[np.ndarray, dict]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
        low, high = np.quantile(finite, [.02, .98])
        if high <= low:
            high = low + 1.0
    else:
        low, high = 0.0, 1.0
    normalized = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
    bgr = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {"p02": float(low), "p98": float(high)}


def _boundary(labels: np.ndarray) -> np.ndarray:
    result = np.zeros(labels.shape, bool)
    right = labels[:, 1:] != labels[:, :-1]
    down = labels[1:] != labels[:-1]
    result[:, :-1] |= right; result[:, 1:] |= right
    result[:-1] |= down; result[1:] |= down
    return result


def _unavailable(shape: tuple[int, int], label: str) -> np.ndarray:
    canvas = np.full((*shape, 3), 225, dtype=np.uint8)
    cv2.line(canvas, (0, 0), (shape[1] - 1, shape[0] - 1), (160, 160, 160), 3)
    cv2.line(canvas, (shape[1] - 1, 0), (0, shape[0] - 1), (160, 160, 160), 3)
    text = f"UNAVAILABLE: {label}"
    cv2.putText(
        canvas, text, (12, max(28, shape[0] // 2)),
        cv2.FONT_HERSHEY_SIMPLEX, .55, (65, 65, 65), 2, cv2.LINE_AA,
    )
    return canvas


def _load_traces(trace_paths) -> dict[str, np.ndarray]:
    if isinstance(trace_paths, (str, Path)):
        path = Path(trace_paths)
        paths = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    elif isinstance(trace_paths, dict):
        paths = [Path(value) for value in trace_paths.values()]
    else:
        paths = [Path(value) for value in trace_paths]
    arrays: dict[str, np.ndarray] = {}
    for path in paths:
        with np.load(path) as data:
            for name in data.files:
                arrays[name] = data[name]
    for name in list(arrays):
        if name.endswith("__packed"):
            base = name[:-8]
            shape_name = f"{base}__shape"
            if shape_name in arrays:
                arrays[base] = np.unpackbits(arrays[name])[: int(np.prod(arrays[shape_name]))].reshape(arrays[shape_name]).astype(bool)
    return arrays


def _project_points(points: np.ndarray, labels: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    canvas = np.full((480, 640, 3), 248, dtype=np.uint8)
    points = points.reshape(-1, 3)
    labels = labels.reshape(-1)
    finite = np.isfinite(points).all(axis=1)
    points, labels = points[finite][:: max(1, finite.sum() // 80000)], labels[finite][:: max(1, finite.sum() // 80000)]
    if not points.size:
        return canvas
    xy = points[:, axes]
    low = np.quantile(xy, .01, axis=0); high = np.quantile(xy, .99, axis=0)
    scale = np.maximum(high - low, 1e-9)
    pixels = np.clip((xy - low) / scale, 0, 1)
    px = np.round(20 + pixels[:, 0] * 599).astype(int)
    py = np.round(459 - pixels[:, 1] * 439).astype(int)
    colors = _palette(labels)
    canvas[py, px] = colors
    return canvas


def write_segment_ply(point_map: np.ndarray, labels: np.ndarray, path: str | Path, *, max_points: int = 250_000) -> Path:
    path = Path(path)
    points = np.asarray(point_map, dtype=float).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1)
    valid = np.isfinite(points).all(axis=1)
    points, labels = points[valid], labels[valid]
    stride = max(1, int(np.ceil(len(points) / max_points)))
    points, labels = points[::stride], labels[::stride]
    colors = _palette(labels)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header",
    ]
    lines.extend(
        f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {int(color[0])} {int(color[1])} {int(color[2])}"
        for point, color in zip(points, colors, strict=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    partial.replace(path)
    return path


def render_case(trace_paths, output_dir: str | Path, *, frame_index: int = 0) -> dict[str, Path]:
    arrays = _load_traces(trace_paths)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    required = ("final_labels", "initial_labels", "coarse_labels", "rgb", "point_map", "confidence")
    missing_required = [name for name in required if name not in arrays]
    if missing_required:
        raise ValueError("Missing required rendering arrays: " + ", ".join(missing_required))
    labels = np.asarray(arrays["final_labels"])
    if labels.ndim == 3:
        labels = labels[min(max(frame_index, 0), len(labels) - 1)]
    point_map = np.asarray(arrays["point_map"])
    if point_map.ndim == 4:
        point_map = point_map[min(max(frame_index, 0), len(point_map) - 1)]
    rgb = _normalize_rgb(arrays["rgb"])
    depth = point_map[..., -1]
    confidence = np.asarray(arrays["confidence"])
    if confidence.ndim == 3:
        confidence = confidence[min(max(frame_index, 0), len(confidence) - 1)]
    def select_frame(value):
        value = np.asarray(value)
        return value[min(max(frame_index, 0), len(value) - 1)] if value.ndim == 3 else value
    initial = select_frame(arrays["initial_labels"])
    coarse = select_frame(arrays["coarse_labels"])
    availability = {name: name in arrays for name in (
        "merge_decision", "component_growth_map", "source_map", "scale_map",
        "dispersion_map", "propagation_hop_map", "temporal_best_iou_map",
        "pre_split_labels", "changed_mask", "split_score_map",
        "split_decision_map",
    )}
    legends = {}
    depth_rgb, legends["depth"] = _heatmap(depth)
    conf_rgb, legends["confidence"] = _heatmap(confidence)
    if availability["scale_map"]:
        scale = select_frame(arrays["scale_map"])
        scale_rgb, legends["scale"] = _heatmap(np.log(np.maximum(scale, 1e-9)))
    else:
        scale_rgb = _unavailable(labels.shape, "scale_map")
    if availability["dispersion_map"]:
        dispersion = select_frame(arrays["dispersion_map"])
        dispersion_rgb, legends["dispersion"] = _heatmap(dispersion)
    else:
        dispersion_rgb = _unavailable(labels.shape, "dispersion_map")
    merge_rgb = _unavailable(labels.shape, "merge_decision")
    decision_colors = np.asarray([
        [0, 0, 0],       # not a boundary
        [20, 190, 110],  # accepted, same coarse
        [18, 130, 220],  # accepted, cross coarse
        [245, 175, 35],  # rejected, same coarse
        [235, 65, 65],   # rejected, cross coarse
        [145, 151, 160], # invalid geometry
    ], dtype=np.uint8)
    if availability["merge_decision"]:
        decision = np.asarray(arrays["merge_decision"])
        if decision.ndim == 3: decision = decision[min(max(frame_index, 0), len(decision) - 1)]
        merge_rgb = _normalize_rgb(rgb)
        for code in range(1, len(decision_colors)):
            merge_rgb[decision == code] = decision_colors[code]
    if availability["component_growth_map"]:
        growth = np.asarray(arrays["component_growth_map"], dtype=float)
        if growth.ndim == 3: growth = growth[min(max(frame_index, 0), len(growth) - 1)]
        growth_rgb, legends["growth"] = _heatmap(np.log1p(growth))
    else:
        growth_rgb = _unavailable(labels.shape, "component_growth_map")
    source_colors = np.asarray([[145, 151, 160], [18, 158, 140], [124, 77, 190]], dtype=np.uint8)
    if availability["source_map"]:
        source = select_frame(arrays["source_map"])
        source_rgb = source_colors[np.clip(source.astype(int), 0, 2)]
    else:
        source_rgb = _unavailable(labels.shape, "source_map")
    if availability["propagation_hop_map"]:
        hops = select_frame(arrays["propagation_hop_map"])
        hop_rgb, legends["propagation_hops"] = _heatmap(np.where(hops >= 0, hops, np.nan))
        hop_rgb[hops < 0] = [145, 151, 160]
    else:
        hop_rgb = _unavailable(labels.shape, "propagation_hop_map")
    if availability["temporal_best_iou_map"]:
        temporal_values = np.asarray(arrays["temporal_best_iou_map"])
        if temporal_values.ndim == 3: temporal_values = temporal_values[min(max(frame_index, 0), len(temporal_values) - 1)]
        temporal_rgb, legends["temporal_iou"] = _heatmap(temporal_values)
    else:
        temporal_rgb = _unavailable(labels.shape, "temporal_best_iou_map")

    if availability["pre_split_labels"]:
        pre_split_rgb = _palette(select_frame(arrays["pre_split_labels"]))
    else:
        pre_split_rgb = _unavailable(labels.shape, "pre_split_labels")

    if availability["changed_mask"]:
        changed = select_frame(arrays["changed_mask"]).astype(bool, copy=False)
        changed_rgb = rgb.copy()
        changed_rgb[changed] = np.round(
            .35 * changed_rgb[changed] + .65 * np.asarray([235, 65, 65])
        ).astype(np.uint8)
        legends["split_changed_regions"] = {
            "false": "unchanged RGB",
            "true": "changed after split (red overlay)",
        }
    else:
        changed_rgb = _unavailable(labels.shape, "changed_mask")

    if availability["split_score_map"]:
        split_scores = np.asarray(select_frame(arrays["split_score_map"]), dtype=float)
        split_score_rgb, legends["split_scores"] = _heatmap(split_scores)
        split_score_rgb[~np.isfinite(split_scores)] = [145, 151, 160]
        legends["split_scores"]["unavailable"] = "NaN"
    else:
        split_score_rgb = _unavailable(labels.shape, "split_score_map")

    legends["split_decisions"] = {
        "0": "none",
        "1": "accepted",
        "2": "no-markers",
        "3": "small-child",
        "4": "low-score",
    }
    if availability["split_decision_map"]:
        split_decision = np.asarray(select_frame(arrays["split_decision_map"]), dtype=np.int64)
        split_decision_rgb = rgb.copy()
        split_decision_colors = np.asarray([
            [0, 0, 0],
            [20, 190, 110],
            [245, 175, 35],
            [124, 77, 190],
            [235, 65, 65],
        ], dtype=np.uint8)
        for code in range(1, len(split_decision_colors)):
            split_decision_rgb[split_decision == code] = split_decision_colors[code]
        split_decision_rgb[(split_decision < 0) | (split_decision > 4)] = [145, 151, 160]
    else:
        split_decision_rgb = _unavailable(labels.shape, "split_decision_map")

    images = {
        "rgb": rgb, "depth": depth_rgb, "confidence": conf_rgb,
        "initial_atoms": _palette(initial), "coarse_layers": _palette(coarse),
        "pre_split_segments": pre_split_rgb, "final_segments": _palette(labels),
        "split_changed_regions": changed_rgb, "split_scores": split_score_rgb,
        "split_decisions": split_decision_rgb, "merge_decisions": merge_rgb,
        "component_growth": growth_rgb, "scale_source": source_rgb,
        "scale_map": scale_rgb, "scale_dispersion": dispersion_rgb,
        "propagation_hops": hop_rgb,
        "temporal": temporal_rgb,
        "pointcloud_top": _project_points(point_map, labels, (0, 2)),
        "pointcloud_side": _project_points(point_map, labels, (2, 1)),
    }
    result = {name: _write_png(output / f"{name}.png", image) for name, image in images.items()}
    write_segment_ply(point_map, labels, output / "segments.ply")
    atomic_write_json(output / "rendering.json", {
        "legends": legends,
        "availability": availability,
        "artifacts": {name: path.name for name, path in result.items()},
    })
    return result


def render_method_comparison(
    method_labels: dict[str, np.ndarray],
    output_dir: str | Path,
    *,
    method_scales: dict[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    """Render the strict three-method partition and split/geometry scale comparison."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if set(method_labels) != set(COLORS):
        raise ValueError(
            "method comparison requires exactly depth, geometry_baseline, "
            "and layer_atomic_split"
        )
    labels = {
        name: np.asarray(method_labels[name])
        for name in COLORS
    }
    shapes = {value.shape for value in labels.values()}
    if len(shapes) != 1:
        raise ValueError("comparison label shapes differ")
    shape = next(iter(shapes))
    panels = []
    for method, values in labels.items():
        panel = np.full((*shape, 3), 242, dtype=np.uint8)
        panel[_boundary(values)] = _hex_rgb(COLORS[method])
        panels.append(panel)
    image = np.concatenate(panels, axis=1)
    result = {"segmentation_disagreement": _write_png(output / "segmentation_disagreement.png", image)}
    comparison_manifest = {
        "methods": list(COLORS),
        "colors": COLORS,
        "availability": {"scale_log_ratio": False},
    }
    method_scales = method_scales or {}
    left_scale = method_scales.get("layer_atomic_split")
    right_scale = method_scales.get("geometry_baseline")
    if left_scale is not None and right_scale is not None:
        left_scale = np.asarray(left_scale)
        right_scale = np.asarray(right_scale)
        if left_scale.shape == right_scale.shape == shape:
            valid = np.isfinite(left_scale) & np.isfinite(right_scale)
            if valid.any():
                ratio = np.full(left_scale.shape, np.nan, dtype=float)
                ratio[valid] = np.log(
                    np.maximum(left_scale[valid], 1e-9)
                    / np.maximum(right_scale[valid], 1e-9)
                )
                ratio_rgb, legend = _heatmap(ratio)
                ratio_rgb[~valid] = [145, 151, 160]
                comparison_manifest["availability"]["scale_log_ratio"] = True
                comparison_manifest["scale_log_ratio"] = {
                    **legend,
                    "numerator": "layer_atomic_split",
                    "denominator": "geometry_baseline",
                    "unavailable": "NaN",
                }
            else:
                ratio_rgb = _unavailable(shape, "finite scale_log_ratio")
        else:
            ratio_rgb = _unavailable(shape, "shape-matched scale_log_ratio")
    else:
        ratio_rgb = _unavailable(shape, "scale_log_ratio")
    result["scale_log_ratio"] = _write_png(
        output / "scale_log_ratio.png", ratio_rgb
    )
    atomic_write_json(output / "comparison-rendering.json", comparison_manifest)
    return result
