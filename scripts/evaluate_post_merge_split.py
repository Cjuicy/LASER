#!/usr/bin/env python3
"""Evaluate conservative post-merge splitting on saved segmentation traces."""

import argparse
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference_engine.utils.layer_atomic_geometry import (  # noqa: E402
    segment_point_map_layer_atomic,
    segment_point_map_layer_atomic_split,
)
from inference_engine.utils.post_merge_split import refine_auto_regions  # noqa: E402


TRACE_KEYS = {
    "rgb": "layer_atomic__inputs__rgb",
    "point_map": "layer_atomic__inputs__point_map",
    "auto_labels": "layer_atomic__segmentation__final_labels",
    "atom_labels": "layer_atomic__segmentation__initial_labels",
    "atom_scales": "layer_atomic__segmentation__atom_scales",
}
GEOMETRY_KEY = "geometry_baseline__segmentation__final_labels"
PANEL_TRACES = {
    "laser-case-00-003825-004079-trace.npz",
    "laser-case-05-002160-002414-trace.npz",
    "laser-tum-freiburg1_360-000000-000254-trace.npz",
    "laser-tum-freiburg1_desk-000135-000389-trace.npz",
}


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in TRACE_KEYS.values() if key not in archive]
        if missing:
            raise KeyError(f"{path}: missing trace keys {missing}")
        trace = {name: np.asarray(archive[key]) for name, key in TRACE_KEYS.items()}
        trace["geometry_labels"] = (
            np.asarray(archive[GEOMETRY_KEY]) if GEOMETRY_KEY in archive else None
        )

    rgb = trace["rgb"]
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.shape != (*trace["point_map"].shape[:2], 3):
        raise ValueError(f"{path}: RGB and point-map shapes are not aligned")
    trace["rgb"] = rgb
    return trace


def summarize(records):
    if not records:
        raise ValueError("records must not be empty")
    region_growth = np.asarray(
        [
            record["split_regions"] / max(record["auto_regions"], 1)
            for record in records
        ],
        dtype=np.float64,
    )
    runtime_overhead = np.asarray(
        [
            (record["split_ms"] - record["auto_ms"])
            / max(record["auto_ms"], 1e-8)
            for record in records
        ],
        dtype=np.float64,
    )
    median_growth = float(np.median(region_growth))
    max_growth = float(np.max(region_growth))
    median_overhead = float(np.median(runtime_overhead))
    return {
        "trace_count": len(records),
        "median_region_growth": median_growth,
        "max_region_growth": max_growth,
        "median_runtime_overhead": median_overhead,
        "p90_runtime_overhead": float(np.quantile(runtime_overhead, 0.90)),
        "accepted_splits": int(
            sum(record.get("split_accepted_count", 0) for record in records)
        ),
        "mean_geometry_boundary_support": float(
            np.mean(
                [
                    record["geometry_boundary_support"]
                    for record in records
                    if record.get("geometry_boundary_support") is not None
                ]
            )
        )
        if any(record.get("geometry_boundary_support") is not None for record in records)
        else None,
        "passes_region_budget": bool(median_growth <= 1.30 and max_growth <= 1.50),
        "passes_runtime_budget": bool(median_overhead <= 0.20),
    }


def _paired_runtimes(auto_operation, split_operation, repeats):
    """Alternate operation order so CPU drift is shared by both measurements."""
    auto_operation()
    split_operation()
    auto_durations = []
    split_durations = []
    for repeat in range(repeats):
        operations = (
            ((auto_operation, auto_durations), (split_operation, split_durations))
            if repeat % 2 == 0
            else ((split_operation, split_durations), (auto_operation, auto_durations))
        )
        for operation, durations in operations:
            start = time.perf_counter()
            operation()
            durations.append((time.perf_counter() - start) * 1000.0)
    return (
        float(np.median(auto_durations)),
        float(np.quantile(auto_durations, 0.90)),
        float(np.median(split_durations)),
        float(np.quantile(split_durations, 0.90)),
    )


def _boundary(labels):
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, :-1] != labels[:, 1:]
    vertical = labels[:-1] != labels[1:]
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1] |= vertical
    boundary[1:] |= vertical
    return boundary


def _geometry_boundary_support(auto_labels, split_labels, geometry_labels):
    if geometry_labels is None:
        return None
    new_boundary = _boundary(split_labels) & ~_boundary(auto_labels)
    count = int(np.count_nonzero(new_boundary))
    if count == 0:
        return 0.0
    geometry_boundary = ndimage.binary_dilation(_boundary(geometry_labels), iterations=1)
    return float(np.count_nonzero(new_boundary & geometry_boundary) / count)


def _rgb_panel(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    finite = rgb[np.isfinite(rgb)]
    if finite.size and np.max(finite) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8)


def _label_panel(labels):
    labels = np.asarray(labels, dtype=np.uint64)
    colors = np.stack(
        (
            (labels * 37 + 17) % 251,
            (labels * 67 + 43) % 253,
            (labels * 97 + 71) % 255,
        ),
        axis=-1,
    ).astype(np.uint8)
    colors[labels == 0] = (28, 28, 28)
    return colors


def _save_panel(path, rgb, auto_labels, aux_on_labels, aux_off_labels):
    panels = [
        ("RGB", _rgb_panel(rgb)),
        ("Auto merge", _label_panel(auto_labels)),
        ("Split + auxiliary", _label_panel(aux_on_labels)),
        ("Split normals only", _label_panel(aux_off_labels)),
    ]
    height, width = panels[0][1].shape[:2]
    title_height = 24
    canvas = Image.new("RGB", (width * len(panels), height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, pixels) in enumerate(panels):
        canvas.paste(Image.fromarray(pixels), (index * width, title_height))
        draw.text((index * width + 6, 5), title, fill="black")
    canvas.save(path)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _paired_delta(on_summary, off_summary):
    def delta(name):
        left = on_summary.get(name)
        right = off_summary.get(name)
        if left is None or right is None:
            return None
        return float(left - right)

    return {
        "accepted_splits": int(
            on_summary["accepted_splits"] - off_summary["accepted_splits"]
        ),
        "median_region_growth": delta("median_region_growth"),
        "median_runtime_overhead": delta("median_runtime_overhead"),
        "mean_geometry_boundary_support": delta("mean_geometry_boundary_support"),
    }


def evaluate(paths, thresholds, aux_states, repeats, output_dir, seg_min_size=500):
    traces = [(Path(path), load_trace(path)) for path in paths]
    if not traces:
        raise ValueError("no trace files matched")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_trace = []
    grouped = {}
    labels_for_panels = {}
    for threshold in thresholds:
        for aux_state in aux_states:
            aux_enabled = aux_state == "on"
            group_records = []
            for path, trace in traces:
                refined, diagnostics = refine_auto_regions(
                    trace["point_map"],
                    trace["rgb"],
                    trace["auto_labels"],
                    trace["atom_labels"],
                    trace["atom_scales"],
                    seg_min_size=seg_min_size,
                    normal_method="cross",
                    split_score_thresh=threshold,
                    split_aux_confirmation=aux_enabled,
                )
                auto_operation = lambda trace=trace: segment_point_map_layer_atomic(
                    trace["point_map"],
                    depth_merge_thresh=0.1,
                    seg_min_size=seg_min_size,
                )
                split_operation = lambda trace=trace, threshold=threshold, aux_enabled=aux_enabled: segment_point_map_layer_atomic_split(
                    trace["point_map"],
                    depth_merge_thresh=0.1,
                    rgb_images=trace["rgb"],
                    normal_method="cross",
                    split_score_thresh=threshold,
                    split_aux_confirmation=aux_enabled,
                    seg_min_size=seg_min_size,
                )
                auto_ms, auto_p90_ms, split_ms, split_p90_ms = _paired_runtimes(
                    auto_operation, split_operation, repeats
                )
                record = {
                    "trace": path.name,
                    "threshold": float(threshold),
                    "aux_state": aux_state,
                    "auto_regions": int(np.unique(trace["auto_labels"]).size),
                    "split_regions": int(np.unique(refined).size),
                    "auto_ms": auto_ms,
                    "auto_p90_ms": auto_p90_ms,
                    "split_ms": split_ms,
                    "split_p90_ms": split_p90_ms,
                    "geometry_boundary_support": _geometry_boundary_support(
                        trace["auto_labels"], refined, trace["geometry_labels"]
                    ),
                    **diagnostics.as_dict(),
                }
                group_records.append(record)
                per_trace.append(record)
                if path.name in PANEL_TRACES:
                    labels_for_panels[(path.name, float(threshold), aux_state)] = refined
            grouped[(float(threshold), aux_state)] = summarize(group_records)

    summaries = {
        f"{threshold:.2f}": {
            state: grouped[(float(threshold), state)] for state in aux_states
        }
        for threshold in thresholds
    }
    paired = {}
    if "on" in aux_states and "off" in aux_states:
        paired = {
            f"{threshold:.2f}": _paired_delta(
                grouped[(float(threshold), "on")],
                grouped[(float(threshold), "off")],
            )
            for threshold in thresholds
        }

    selected_threshold = None
    if "on" in aux_states:
        preferred = [candidate for candidate in (0.10, 0.15, 0.20) if candidate in thresholds]
        preferred.extend(candidate for candidate in thresholds if candidate not in preferred)
        for threshold in preferred:
            production = grouped[(float(threshold), "on")]
            if production["passes_region_budget"] and production["passes_runtime_budget"]:
                selected_threshold = float(threshold)
                break

    summary = {
        "selected_production_threshold": selected_threshold,
        "production_aux_state": "on",
        "thresholds": summaries,
        "paired_aux_on_minus_off": paired,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_trace.json").write_text(
        json.dumps(_json_ready(per_trace), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if selected_threshold is not None and {"on", "off"}.issubset(aux_states):
        for path, trace in traces:
            if path.name not in PANEL_TRACES:
                continue
            aux_on = labels_for_panels[(path.name, selected_threshold, "on")]
            aux_off = labels_for_panels[(path.name, selected_threshold, "off")]
            _save_panel(
                output_dir / f"{path.stem}-comparison.png",
                trace["rgb"],
                trace["auto_labels"],
                aux_on,
                aux_off,
            )
    return summary


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    parser.add_argument("--aux-states", nargs="+", choices=("on", "off"), default=["on", "off"])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seg-min-size", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main():
    args = get_args_parser().parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    paths = sorted(
        {
            match
            for pattern in args.trace_glob
            for match in glob.glob(pattern)
        }
    )
    if not paths:
        raise SystemExit("no trace files matched")
    summary = evaluate(
        paths,
        args.thresholds,
        args.aux_states,
        args.repeats,
        args.output_dir,
        args.seg_min_size,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    if "on" in args.aux_states and summary["selected_production_threshold"] is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
