"""Headless deterministic renderers for selected diagnostic cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .storage import atomic_write_json


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
    path.write_bytes(encoded.tobytes())
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_case(trace_paths, output_dir: str | Path) -> dict[str, Path]:
    arrays = _load_traces(trace_paths)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(arrays.get("final_labels", arrays.get("coarse_labels")))
    if labels.ndim == 3:
        labels = labels[0]
    if "point_map" in arrays:
        point_map = np.asarray(arrays["point_map"])
    else:
        grid_y, grid_x = np.mgrid[:labels.shape[0], :labels.shape[1]]
        point_map = np.stack([grid_x, grid_y, np.ones(labels.shape)], axis=-1)
    if point_map.ndim == 4:
        point_map = point_map[0]
    rgb = _normalize_rgb(arrays.get("rgb", _palette(labels)))
    depth = point_map[..., -1]
    confidence = np.asarray(arrays.get("confidence", np.ones(labels.shape)))
    if confidence.ndim == 3:
        confidence = confidence[0]
    initial = np.asarray(arrays.get("initial_labels", labels)); initial = initial[0] if initial.ndim == 3 else initial
    coarse = np.asarray(arrays.get("coarse_labels", labels)); coarse = coarse[0] if coarse.ndim == 3 else coarse
    source = np.asarray(arrays.get("source_map", np.zeros(labels.shape, np.uint8))); source = source[0] if source.ndim == 3 else source
    scale = np.asarray(arrays.get("scale_map", np.ones(labels.shape))); scale = scale[0] if scale.ndim == 3 else scale
    dispersion = np.asarray(arrays.get("dispersion_map", np.zeros(labels.shape))); dispersion = dispersion[0] if dispersion.ndim == 3 else dispersion
    legends = {}
    depth_rgb, legends["depth"] = _heatmap(depth)
    conf_rgb, legends["confidence"] = _heatmap(confidence)
    scale_rgb, legends["scale"] = _heatmap(np.log(np.maximum(scale, 1e-9)))
    dispersion_rgb, legends["dispersion"] = _heatmap(dispersion)
    decision = np.asarray(arrays.get("merge_decision", _boundary(coarse).astype(np.uint8)))
    if decision.ndim == 3: decision = decision[0]
    merge_rgb = _normalize_rgb(rgb)
    merge_rgb[decision > 0] = [255, 55, 45]
    atom_counts = {int(segment): np.unique(initial[labels == segment]).size for segment in np.unique(labels)}
    growth = np.zeros(labels.shape, dtype=float)
    for segment, count in atom_counts.items(): growth[labels == segment] = count
    growth_rgb, legends["growth"] = _heatmap(np.log1p(growth))
    source_colors = np.asarray([[145, 151, 160], [18, 158, 140], [124, 77, 190]], dtype=np.uint8)
    source_rgb = source_colors[np.clip(source.astype(int), 0, 2)]
    temporal_rgb = _palette(labels)
    temporal_rgb[_boundary(labels)] = [255, 255, 255]
    images = {
        "rgb": rgb, "depth": depth_rgb, "confidence": conf_rgb,
        "initial_atoms": _palette(initial), "coarse_layers": _palette(coarse),
        "final_segments": _palette(labels), "merge_decisions": merge_rgb,
        "component_growth": growth_rgb, "scale_source": source_rgb,
        "scale_map": scale_rgb, "scale_dispersion": dispersion_rgb,
        "temporal": temporal_rgb,
        "pointcloud_top": _project_points(point_map, labels, (0, 2)),
        "pointcloud_side": _project_points(point_map, labels, (2, 1)),
    }
    result = {name: _write_png(output / f"{name}.png", image) for name, image in images.items()}
    write_segment_ply(point_map, labels, output / "segments.ply")
    atomic_write_json(output / "rendering.json", {"legends": legends, "artifacts": {name: path.name for name, path in result.items()}})
    return result
