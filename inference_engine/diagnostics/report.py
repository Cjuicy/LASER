"""Offline two-level HTML/CSV/JSON report builder."""

from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .storage import atomic_write_json


STYLE = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f4f7fb;color:#172033}main{max-width:1320px;margin:auto;padding:28px}
h1,h2{margin:.35em 0}.card{background:white;border:1px solid #d9e2ef;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 2px 8px #1720330d}
table{border-collapse:collapse;width:100%}th,td{padding:8px 10px;border-bottom:1px solid #e4e9f1;text-align:left}th{background:#edf3fa}
.ok{color:#087f5b;font-weight:700}.bad{color:#c92a2a;font-weight:700}.warn{background:#fff3bf;padding:10px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}.grid img{width:100%;border:1px solid #d9e2ef;border-radius:6px}code{background:#edf3fa;padding:2px 5px;border-radius:4px}a{color:#2459a9}
"""

COLORS = {
    "depth": "#087f5b",
    "geometry_baseline": "#c92a2a",
    "layer_atomic_split": "#2459a9",
}
REGRET_FIELDS = (
    "split_minus_depth_regret",
    "split_minus_geometry_regret",
)
CASE_REASON_LABELS = {
    "trajectory_degradation": "Split trajectory degradation",
    "trajectory_improvement": "Split trajectory improvement",
    "trajectory_change": "Split trajectory change point",
    "split_anomaly": "Split activity anomaly",
    "split_no_trajectory_effect": "Split without observed trajectory effect",
    "matched_control": "Motion-matched no-split control",
}


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
    return path


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _strict_summary(summary: dict) -> dict:
    """Return report data restricted to the registered official methods."""
    result = dict(summary)
    metrics = summary.get("sequence_metrics", {})
    result["sequence_metrics"] = {
        method: metrics[method] for method in COLORS if method in metrics
    }
    if "official_aggregate" in summary:
        aggregate = summary.get("official_aggregate", {})
        result["official_aggregate"] = {
            method: aggregate[method] for method in COLORS if method in aggregate
        }
    if "official_ranking" in summary:
        result["official_ranking"] = {
            sequence: [
                row for row in rows if row.get("config_id") in COLORS
            ]
            for sequence, rows in summary.get("official_ranking", {}).items()
        }
    return result


def _svg_heatmap(sequence_metrics: dict) -> str:
    configs = sorted(sequence_metrics)
    sequences = sorted({seq for values in sequence_metrics.values() for seq in values})
    if not configs or not sequences:
        return "<p>No complete trajectory metrics yet.</p>"
    values = [sequence_metrics[c][s].get("ate_rmse") for c in configs for s in sequences if s in sequence_metrics[c] and sequence_metrics[c][s].get("ate_rmse") is not None]
    maximum = max(values, default=1) or 1
    cells = []
    for row, config in enumerate(configs):
        cells.append(f'<text x="4" y="{45 + row*34}" font-size="12">{escape(config)}</text>')
        for col, seq in enumerate(sequences):
            value = sequence_metrics.get(config, {}).get(seq, {}).get("ate_rmse")
            intensity = 0 if value is None else int(220 * float(value) / maximum)
            color = f"rgb({245-intensity//3},{250-intensity//2},{255-intensity})"
            x = 170 + col * 72; y = 22 + row * 34
            cells.append(f'<rect x="{x}" y="{y}" width="68" height="30" fill="{color}"/><text x="{x+4}" y="{y+20}" font-size="11">{"—" if value is None else f"{value:.2f}"}</text>')
    for col, seq in enumerate(sequences):
        cells.append(f'<text x="{188+col*72}" y="16" font-size="12">{seq}</text>')
    return f'<svg viewBox="0 0 {190+72*len(sequences)} {35+34*len(configs)}" role="img">{"".join(cells)}</svg>'


def _case_page(case_dir: Path, report_cases: Path, run_dir: Path) -> tuple[str, float, dict]:
    metrics = _load(case_dir / "metrics.json", {})
    relative = case_dir.relative_to(run_dir)
    page_name = "-".join(relative.parts) + ".html"
    image_names = [
        "rgb.png", "initial_atoms.png", "coarse_layers.png", "pre_split_segments.png",
        "final_segments.png", "split_changed_regions.png", "split_scores.png",
        "split_decisions.png", "split_parent_map.png", "split_child_map.png",
        "merge_decisions.png", "component_growth.png", "scale_source.png",
        "scale_map.png", "scale_dispersion.png", "propagation_hops.png", "temporal.png",
        "pointcloud_top.png", "pointcloud_side.png", "segmentation_disagreement.png", "scale_log_ratio.png",
    ]
    images = "".join(
        f'<figure><img src="../../{escape(str(relative / name))}" alt="{escape(name)}"><figcaption>{escape(name)}</figcaption></figure>'
        for name in image_names if (case_dir / name).exists()
    )
    method_sections = []
    for method in COLORS:
        method_dir = case_dir / method
        method_images = "".join(
            f'<figure><img src="../../{escape(str(relative / method / name))}" alt="{escape(method + ": " + name)}"><figcaption>{escape(name)}</figcaption></figure>'
            for name in image_names if (method_dir / name).exists()
        )
        if method_dir.is_dir():
            method_metrics = _load(method_dir / "metrics.json", {})
            method_sections.append(
                f'<section class="card"><h2 style="color:{COLORS[method]}">{escape(method)}</h2>'
                f'<pre>{escape(json.dumps(method_metrics, indent=2, ensure_ascii=False))}</pre>'
                f'<div class="grid">{method_images}</div></section>'
            )
    ply_link = f'<p><a href="../../{escape(str(relative / "segments.ply"))}">segments.ply</a></p>' if (case_dir / "segments.ply").exists() else ""
    trace_link = f'<p><a href="../../{escape(str(relative / "trace.npz"))}">trace.npz</a></p>' if (case_dir / "trace.npz").exists() else ""
    timeline = _load(case_dir / "trajectory-timeline.json", {})
    if not timeline:
        raise ValueError(f"Missing required local trajectory timeline: {case_dir}")
    geometry_evidence = metrics.get("geometry_split_comparison")
    structural = metrics.get("split_structural_summary")
    if not isinstance(geometry_evidence, dict) or not isinstance(structural, dict):
        raise ValueError(f"Missing required split case summaries: {case_dir}")
    html = (
        f"<!doctype html><meta charset='utf-8'><style>{STYLE}</style><main>"
        f"<a href='../index.html'>← Overview</a><h1>{escape(str(relative))}</h1>"
        "<h2>Segmentation → Merge → Scale → Trajectory</h2>"
        f"<div class='card'><pre>{escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>{ply_link}{trace_link}</div>"
        "<section class='card'><h2>Aligned local three-error / two-regret timeline</h2>"
        f"{_local_timeline_svg(timeline)}</section>"
        "<section class='card'><h2>Geometry boundary disagreement pre/post</h2>"
        f"<pre>{escape(json.dumps(geometry_evidence, indent=2, ensure_ascii=False))}</pre></section>"
        "<section class='card'><h2>Structural split summary</h2>"
        f"<pre>{escape(json.dumps(structural, indent=2, ensure_ascii=False))}</pre></section>"
        f"<div class='grid'>{images}</div>{''.join(method_sections)}</main>"
    )
    report_cases.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(report_cases / page_name, html)
    score = float(metrics.get("selection_score", metrics.get("score", 0)) or 0)
    return page_name, score, metrics


def _selection_diagnostics(run_dir: Path) -> dict:
    selected = _load(run_dir / "selected_intervals.json", [])
    if not selected:
        selected = []
        for metrics_path in sorted((run_dir / "cases").glob("*/*/metrics.json")):
            metrics = _load(metrics_path, {})
            interval = metrics.get("interval", {})
            selected.append({
                "sequence_id": interval.get("sequence_id"),
                "start_frame": interval.get("start_frame"),
                "end_frame": interval.get("end_frame"),
                "reasons": metrics.get(
                    "selection_reasons", interval.get("reasons", [])
                ),
            })
    normalized_intervals = []
    for interval in selected:
        sequence = interval.get("sequence_id")
        start = interval.get("start_frame")
        end = interval.get("end_frame")
        if sequence is None or start is None or end is None:
            continue
        normalized_intervals.append({
            "sequence_id": str(sequence),
            "start_frame": int(start),
            "end_frame": int(end),
            "reasons": sorted(set(interval.get("reasons", []))),
        })
    normalized_intervals.sort(
        key=lambda item: (
            item["sequence_id"], item["start_frame"], item["end_frame"]
        )
    )
    reason_counts: dict[str, int] = {}
    for interval in normalized_intervals:
        for reason in interval["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    records = []
    for record in _load(run_dir / "selection_records.json", []):
        if record.get("config_id") not in COLORS:
            continue
        sequence = str(record.get("sequence_id", ""))
        start = record.get("frame_start")
        end = record.get("frame_end")
        reasons = sorted({
            reason
            for interval in normalized_intervals
            if interval["sequence_id"] == sequence
            and start is not None and end is not None
            and int(start) <= interval["end_frame"]
            and int(end) >= interval["start_frame"]
            for reason in interval["reasons"]
        })
        records.append({
            "config_id": record.get("config_id"),
            "sequence_id": record.get("sequence_id"),
            "window_id": record.get("window_id"),
            "frame_start": start,
            "frame_end": end,
            "split_minus_depth_regret": record.get("split_minus_depth_regret"),
            "split_minus_geometry_regret": record.get(
                "split_minus_geometry_regret"
            ),
            "selection_reasons": reasons,
        })
    records.sort(key=lambda item: (
        str(item["sequence_id"]),
        -1 if item["frame_start"] is None else int(item["frame_start"]),
        -1 if item["frame_end"] is None else int(item["frame_end"]),
        str(item["config_id"]),
        -1 if item["window_id"] is None else int(item["window_id"]),
    ))
    return {
        "selection_reasons": sorted(reason_counts),
        "reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
    }


def _write_csv(run_dir: Path, summary: dict, selection_diagnostics: dict):
    rows = [["config_id", "sequence_id", "ATE_RMSE", "RPE_translation_RMSE", "RPE_rotation_RMSE_deg"]]
    for config in COLORS:
        sequences = summary.get("sequence_metrics", {}).get(config, {})
        for sequence, metrics in sorted(sequences.items()):
            rows.append([config, sequence, metrics.get("ate_rmse"), metrics.get("rpe_translation_rmse"), metrics.get("rpe_rotation_rmse_deg")])
    path = run_dir / "report" / "sequence_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    partial.replace(path)

    fields = [
        "config_id", "sequence_id", "window_id", "frame_start", "frame_end",
        "split_minus_depth_regret", "split_minus_geometry_regret",
        "selection_reasons",
    ]
    path = run_dir / "report" / "metrics.csv"
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in selection_diagnostics["records"]:
            writer.writerow({
                **record,
                "selection_reasons": json.dumps(
                    record["selection_reasons"], separators=(",", ":")
                ),
            })
    partial.replace(path)


def _correlations(run_dir: Path) -> list[dict]:
    records = _load(run_dir / "selection_records.json", [])
    split_signals = (
        "split_score_mean", "split_accepted_count",
        "split_changed_pixel_ratio", "split_segment_count_delta",
    )
    signals = split_signals + (
        "merge_anomaly", "atom_anomaly", "scale_dispersion", "temporal_churn",
    )
    grouped: dict[tuple[str, str], list[dict]] = {}
    split_grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (record.get("config_id", ""), record.get("sequence_id", ""))
        grouped.setdefault(key, []).append(record)
        if record.get("config_id") == "layer_atomic_split":
            split_grouped.setdefault(key, []).append(record)
    result = []
    for target in REGRET_FIELDS:
        for signal in signals:
            for lag in range(4):
                left, right = [], []
                total = 0
                source_groups = split_grouped if signal in split_signals else grouped
                for rows in source_groups.values():
                    rows.sort(key=lambda row: row.get("window_id", 0))
                    if len(rows) <= lag:
                        continue
                    for index in range(len(rows) - lag):
                        total += 1
                        x = rows[index].get(signal)
                        y = rows[index + lag].get(target)
                        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                            left.append(float(x)); right.append(float(y))
                if len(left) >= 2 and np.std(left) > 0 and np.std(right) > 0:
                    pearson = float(np.corrcoef(left, right)[0, 1])
                    spearman = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
                else:
                    pearson = spearman = None
                result.append({
                    "target": target, "signal": signal, "lag_windows": lag,
                    "sample_count": len(left),
                    "missing_rate": float(1 - len(left) / total) if total else None,
                    "pearson": pearson, "spearman": spearman,
                })
    return result


def _svg_points(values, *, width, height, minimum, maximum):
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    points = []
    for index, value in enumerate(array):
        if not finite[index]:
            continue
        x = 12 + index * (width - 24) / max(len(array) - 1, 1)
        y = height - 18 - (value - minimum) * (height - 36) / max(
            maximum - minimum, 1e-9
        )
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _local_timeline_svg(timeline: dict) -> str:
    series = {
        **timeline.get("errors", {}),
        **timeline.get("regrets", {}),
    }
    values = [
        float(value)
        for item in series.values()
        for value in item
        if value is not None and np.isfinite(value)
    ]
    if not values:
        return "<p>UNAVAILABLE: local aligned errors and regrets.</p>"
    low, high = min(values), max(values)
    width, height = 720, 190
    colors = {
        **COLORS,
        "split_minus_depth_regret": "#7048e8",
        "split_minus_geometry_regret": "#e67700",
    }
    paths = []
    for name, item in series.items():
        points = _svg_points(
            item, width=width, height=height, minimum=low, maximum=high
        )
        paths.append(
            f'<polyline fill="none" stroke="{colors.get(name, "#495057")}" '
            f'stroke-width="1.7" points="{points}"/><text x="12" '
            f'y="{16 + 14 * len(paths)}" font-size="10">{escape(name)}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(paths)}</svg>'


def _timeline_svg(summary: dict, records: list[dict], selected: list[dict]) -> str:
    metrics = summary.get("sequence_metrics", {})
    panels = []
    sequences = sorted({
        sequence for config in COLORS
        for sequence in metrics.get(config, {})
    })
    for sequence in sequences:
        split = metrics.get("layer_atomic_split", {}).get(sequence, {}).get(
            "per_frame_translation_error"
        ) or []
        depth = metrics.get("depth", {}).get(sequence, {}).get(
            "per_frame_translation_error"
        ) or []
        geometry = metrics.get("geometry_baseline", {}).get(sequence, {}).get(
            "per_frame_translation_error"
        ) or []
        count = min(len(split), len(depth), len(geometry))
        if not count:
            continue
        regret_series = {
            "split_minus_depth_regret": (
                np.asarray(split[:count], dtype=float)
                - np.asarray(depth[:count], dtype=float)
            ),
            "split_minus_geometry_regret": (
                np.asarray(split[:count], dtype=float)
                - np.asarray(geometry[:count], dtype=float)
            ),
        }
        all_values = np.concatenate(list(regret_series.values()))
        finite = all_values[np.isfinite(all_values)]
        low, high = (
            (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
        )
        width, height = 620, 150
        overlays = []
        for interval in selected:
            if str(interval.get("sequence_id")) != sequence:
                continue
            start = int(interval.get("start_frame", 0))
            end = int(interval.get("end_frame", start))
            x = 12 + start * (width - 24) / max(count - 1, 1)
            overlay_width = max(
                2.0, (end - start + 1) * (width - 24) / max(count, 1)
            )
            overlays.append(
                f'<rect x="{x:.1f}" y="20" width="{overlay_width:.1f}" '
                'height="112" fill="#ffd43b" opacity="0.20"/>'
            )
        activity = [
            row for row in records
            if row.get("config_id") == "layer_atomic_split"
            and str(row.get("sequence_id")) == sequence
        ]
        bars = []
        max_activity = max(
            (float(row.get("split_accepted_count") or 0) for row in activity),
            default=0.0,
        )
        for row in activity:
            value = float(row.get("split_accepted_count") or 0)
            if value <= 0:
                continue
            x = 12 + int(row.get("frame_start", 0)) * (width - 24) / max(count - 1, 1)
            bar_height = 22 * value / max(max_activity, 1e-9)
            bars.append(
                f'<rect x="{x:.1f}" y="{132-bar_height:.1f}" width="3" '
                f'height="{bar_height:.1f}" fill="#343a40"/>'
            )
        lines = []
        for name, values in regret_series.items():
            color = "#7048e8" if name.endswith("depth_regret") else "#e67700"
            points = _svg_points(
                values, width=width, height=height, minimum=low, maximum=high
            )
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
                f'points="{points}"/><text x="12" y="{36 + 13*len(lines)}" '
                f'font-size="10">{name}</text>'
            )
        panels.append(
            f'<svg viewBox="0 0 {width} {height}" role="img">'
            f'<text x="10" y="13">KITTI {sequence}: two regrets + split activity</text>'
            f'{"".join(overlays)}{"".join(lines)}{"".join(bars)}</svg>'
        )
    return "".join(panels) or "<p>No per-frame regret series yet.</p>"


def _scatter_svg(records: list[dict]) -> str:
    signals = (
        "split_score_mean", "split_accepted_count",
        "split_changed_pixel_ratio", "split_segment_count_delta",
    )
    panels = []
    for signal in signals:
        for target in REGRET_FIELDS:
            pairs = []
            for row in records:
                if row.get("config_id") != "layer_atomic_split":
                    continue
                x, y = row.get(signal), row.get(target)
                if x is None or y is None:
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    pairs.append((float(x), float(y)))
            width, height = 260, 150
            circles = []
            if pairs:
                xs, ys = zip(*pairs, strict=True)
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                for x, y in pairs:
                    px = 12 + (x - xmin) * (width - 24) / max(xmax - xmin, 1e-9)
                    py = height - 16 - (y - ymin) * (height - 38) / max(ymax - ymin, 1e-9)
                    circles.append(
                        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="#2459a9"/>'
                    )
            panels.append(
                f'<svg viewBox="0 0 {width} {height}" role="img">'
                f'<text x="8" y="12" font-size="9">{signal} vs {target}</text>'
                f'{"".join(circles)}</svg>'
            )
    return "".join(panels)


def _ranking_tables(summary: dict) -> tuple[str, str]:
    ranking_rows = "".join(
        f"<tr><td>{escape(sequence)}</td><td>{rank}</td>"
        f"<td>{escape(row.get('config_id', ''))}</td><td>{row.get('ate_rmse', '—')}</td></tr>"
        for sequence, rows in sorted(summary.get("official_ranking", {}).items())
        for rank, row in enumerate(rows, 1)
    )
    aggregate_rows = "".join(
        f"<tr><td>{escape(config)}</td><td>{values.get('mean_ate', '—')}</td>"
        f"<td>{values.get('median_ate', '—')}</td><td>{values.get('wins', '—')}</td>"
        f"<td>{values.get('max_sequence_regression', '—')}</td></tr>"
        for config, values in summary.get("official_aggregate", {}).items()
    )
    return ranking_rows, aggregate_rows


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"; report_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load(run_dir / "manifest.json", {})
    summary = _strict_summary(_load(run_dir / "summary.json", {}))
    selection_diagnostics = _selection_diagnostics(run_dir)
    summary["selection_diagnostics"] = selection_diagnostics
    _write_csv(run_dir, summary, selection_diagnostics)
    atomic_write_json(report_dir / "summary.json", summary)
    correlations = _correlations(run_dir)
    atomic_write_json(report_dir / "correlations.json", correlations)
    cases = []
    for metrics_path in sorted((run_dir / "cases").glob("*/*/metrics.json")) if (run_dir / "cases").exists() else []:
        case_dir = metrics_path.parent
        page, score, metrics = _case_page(case_dir, report_dir / "cases", run_dir)
        cases.append((score, str(case_dir.relative_to(run_dir)), page, metrics))
    cases.sort(key=lambda item: (-item[0], item[1]))
    case_rows = "".join(
        "<tr><td>{}</td><td><a href='cases/{}'>{}</a></td><td>{:.3g}</td>"
        "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
            rank, escape(page), escape(label), score,
            metrics.get("split_minus_depth_regret", "—"),
            metrics.get("split_minus_geometry_regret", "—"),
            escape(", ".join(metrics.get("selection_reasons", metrics.get("interval", {}).get("reasons", [])))),
        )
        for rank, (score, label, page, metrics) in enumerate(cases, 1)
    ) or "<tr><td colspan='6'>No selected cases rendered.</td></tr>"
    guard = summary.get("stability_guard", {})
    passed = guard.get("passed")
    guard_text = "PASS" if passed else "FAIL / incomplete"
    warning = "" if manifest.get("status") == "complete" else "<p class='warn'>Incomplete run: available artifacts are shown, but official comparison is not complete.</p>"
    recovery = escape(json.dumps(summary.get("recovery", summary.get("recovery_gap", {})), ensure_ascii=False, indent=2))
    provenance = escape(json.dumps({key: manifest.get(key) for key in ("run_id", "git_commit", "checkpoint_sha256", "budget", "status")}, ensure_ascii=False, indent=2))
    heatmap = _svg_heatmap(summary.get("sequence_metrics", {}))
    correlation_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(row["target"]), escape(row["signal"]), row["lag_windows"],
            row["sample_count"],
            "—" if row["pearson"] is None else f"{row['pearson']:.3f}",
            "—" if row["spearman"] is None else f"{row['spearman']:.3f}",
        )
        for row in correlations
    )
    rpe_rows = "".join(
        f"<tr><td>{escape(config)}</td><td>{escape(sequence)}</td><td>{metrics.get('rpe_translation_rmse', '—')}</td><td>{metrics.get('rpe_rotation_rmse_deg', '—')}</td></tr>"
        for config in COLORS
        for sequences in (summary.get("sequence_metrics", {}).get(config, {}),)
        for sequence, metrics in sorted(sequences.items())
    )
    method_legend = "".join(
        f'<li><code style="color:{color}">{escape(method)}</code></li>'
        for method, color in COLORS.items()
    )
    reason_legend = "".join(
        f'<li><code>{escape(reason)}</code>: {escape(label)}</li>'
        for reason, label in CASE_REASON_LABELS.items()
    )
    raw_records = _load(run_dir / "selection_records.json", [])
    selected = _load(run_dir / "selected_intervals.json", [])
    ranking_rows, aggregate_rows = _ranking_tables(summary)
    coverage = escape(json.dumps(
        summary.get("selection_coverage", {}), ensure_ascii=False, indent=2
    ))
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>LASER segmentation diagnostics</title><style>{STYLE}</style></head><body><main>
<h1>LASER segmentation diagnostics</h1>{warning}
<section class='card'><h2>Official methods</h2><ul>{method_legend}</ul></section>
<div class='card'><h2>Run provenance &amp; storage budget</h2><pre>{provenance}</pre></div>
<div class='grid'><section class='card'><h2>Stability Guard</h2><p class='{"ok" if passed else "bad"}'>{guard_text}</p><pre>{escape(json.dumps(guard, ensure_ascii=False, indent=2))}</pre></section><section class='card'><h2>Recovery</h2><pre>{recovery}</pre></section></div>
	<section class='card'><h2>ATE / RPE overview heatmap</h2>{heatmap}<table><tr><th>Config</th><th>Sequence</th><th>RPE trans</th><th>RPE rot deg</th></tr>{rpe_rows}</table><p>One global Sim(3) alignment per sequence; RPE delta=1 frame, all pairs.</p></section>
	<section class='card'><h2>Official per-sequence ranking</h2><table><tr><th>Sequence</th><th>Rank</th><th>Method</th><th>ATE RMSE</th></tr>{ranking_rows}</table><h3>Aggregate mean / median / wins</h3><table><tr><th>Method</th><th>Mean ATE</th><th>Median ATE</th><th>Wins</th><th>Maximum sequence regression</th></tr>{aggregate_rows}</table></section>
	<section class='card'><h2>Two regret timelines for every requested sequence</h2>{_timeline_svg(summary, raw_records, selected)}<p>Yellow regions are the selected interval overlay; black bars show split activity aligned in time.</p></section>
	<section class='card'><h2>Selection coverage</h2><pre>{coverage}</pre></section>
	<section class='card'><h2>Selected case ranking</h2><table><tr><th>#</th><th>Case</th><th>Selection score</th><th>split_minus_depth_regret</th><th>split_minus_geometry_regret</th><th>Selection reasons</th></tr>{case_rows}</table><h3>Case reason labels</h3><ul>{reason_legend}</ul></section>
	<section class='card'><h2>Split-specific scatter / correlation evidence</h2><div class='grid'>{_scatter_svg(raw_records)}</div><p>Signals: split_score_mean, split_accepted_count, split_changed_pixel_ratio, and split_segment_count_delta (region growth). Geometry is a comparator, not ground truth.</p></section>
	<section class='card'><h2>Correlation &amp; lag diagnostics</h2><table><tr><th>Regret</th><th>Signal</th><th>Lag</th><th>N</th><th>Pearson</th><th>Spearman</th></tr>{correlation_rows}</table><p>Correlation tables (Pearson/Spearman, lag 0–3) support checking hypotheses; they do not establish causality. Missing rates and sample counts are exported in <a href='correlations.json'>correlations.json</a>.</p></section>
<section class='card'><h2>Exports</h2><a href='sequence_metrics.csv'>sequence_metrics.csv</a></section>
</main></body></html>"""
    index = report_dir / "index.html"
    _atomic_write_text(index, html)
    return index
