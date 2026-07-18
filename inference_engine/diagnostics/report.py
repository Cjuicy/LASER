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
        "split_decisions.png",
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
    html = f"<!doctype html><meta charset='utf-8'><style>{STYLE}</style><main><a href='../index.html'>← Overview</a><h1>{escape(str(relative))}</h1><h2>Segmentation → Merge → Scale → Trajectory</h2><div class='card'><pre>{escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre>{ply_link}{trace_link}</div><div class='grid'>{images}</div>{''.join(method_sections)}</main>"
    report_cases.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(report_cases / page_name, html)
    score = float(metrics.get("selection_score", metrics.get("score", 0)) or 0)
    return page_name, score, metrics


def _write_csv(run_dir: Path, summary: dict):
    rows = [["config_id", "sequence_id", "ATE_RMSE", "RPE_translation_RMSE", "RPE_rotation_RMSE_deg"]]
    for config in COLORS:
        sequences = summary.get("sequence_metrics", {}).get(config, {})
        for sequence, metrics in sorted(sequences.items()):
            rows.append([config, sequence, metrics.get("ate_rmse"), metrics.get("rpe_translation_rmse"), metrics.get("rpe_rotation_rmse_deg")])
    for filename in ("sequence_metrics.csv", "metrics.csv"):
        path = run_dir / "report" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        with partial.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        partial.replace(path)


def _correlations(run_dir: Path) -> list[dict]:
    records = _load(run_dir / "selection_records.json", [])
    signals = ("merge_anomaly", "atom_anomaly", "scale_dispersion", "temporal_churn")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record.get("config_id", ""), record.get("sequence_id", "")), []).append(record)
    result = []
    for target in REGRET_FIELDS:
        for signal in signals:
            for lag in range(4):
                left, right = [], []
                total = 0
                for rows in grouped.values():
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


def _timeline_svg(summary: dict) -> str:
    metrics = summary.get("sequence_metrics", {})
    panels = []
    for sequence in ("02", "04", "10"):
        series = []
        maximum = 0.0
        for config in COLORS:
            values = metrics.get(config, {}).get(sequence, {}).get("per_frame_translation_error") or []
            if values:
                array = np.asarray(values, dtype=float)
                finite = array[np.isfinite(array)]
                maximum = max(maximum, float(finite.max()) if finite.size else 0)
                series.append((config, array))
        if not series:
            continue
        width, height = 440, 125
        paths = []
        for config, values in series:
            sample = values[np.linspace(0, len(values) - 1, min(len(values), 300)).astype(int)]
            points = " ".join(
                f"{10 + index * (width-20)/max(len(sample)-1,1):.1f},{height-15 - min(max(float(value),0),maximum)*(height-30)/max(maximum,1e-9):.1f}"
                for index, value in enumerate(sample)
            )
            paths.append(f'<polyline fill="none" stroke="{COLORS[config]}" stroke-width="1.5" points="{points}"/>')
        panels.append(f'<svg viewBox="0 0 {width} {height}" role="img"><text x="10" y="13">KITTI {sequence}</text>{"".join(paths)}</svg>')
    return "".join(panels) or "<p>No per-frame trajectory series yet.</p>"


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"; report_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load(run_dir / "manifest.json", {})
    summary = _strict_summary(_load(run_dir / "summary.json", {}))
    _write_csv(run_dir, summary)
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
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>LASER segmentation diagnostics</title><style>{STYLE}</style></head><body><main>
<h1>LASER segmentation diagnostics</h1>{warning}
<section class='card'><h2>Official methods</h2><ul>{method_legend}</ul></section>
<div class='card'><h2>Run provenance &amp; storage budget</h2><pre>{provenance}</pre></div>
<div class='grid'><section class='card'><h2>Stability Guard</h2><p class='{"ok" if passed else "bad"}'>{guard_text}</p><pre>{escape(json.dumps(guard, ensure_ascii=False, indent=2))}</pre></section><section class='card'><h2>Recovery</h2><pre>{recovery}</pre></section></div>
<section class='card'><h2>ATE / RPE overview heatmap</h2>{heatmap}<table><tr><th>Config</th><th>Sequence</th><th>RPE trans</th><th>RPE rot deg</th></tr>{rpe_rows}</table><p>One global Sim(3) alignment per sequence; RPE delta=1 frame, all pairs.</p></section>
<section class='card'><h2>Error timeline and selected intervals</h2>{_timeline_svg(summary)}<p>The ranked cases show trajectory regret alongside segmentation, merge, scale and temporal evidence.</p></section>
<section class='card'><h2>Selected case ranking</h2><table><tr><th>#</th><th>Case</th><th>Selection score</th><th>split_minus_depth_regret</th><th>split_minus_geometry_regret</th><th>Selection reasons</th></tr>{case_rows}</table><h3>Case reason labels</h3><ul>{reason_legend}</ul></section>
<section class='card'><h2>Correlation &amp; lag diagnostics</h2><table><tr><th>Regret</th><th>Signal</th><th>Lag</th><th>N</th><th>Pearson</th><th>Spearman</th></tr>{correlation_rows}</table><p>Correlation tables (Pearson/Spearman, lag 0–3) support checking hypotheses; they do not establish causality. Missing rates and sample counts are exported in <a href='correlations.json'>correlations.json</a>.</p></section>
<section class='card'><h2>Exports</h2><a href='sequence_metrics.csv'>sequence_metrics.csv</a></section>
</main></body></html>"""
    index = report_dir / "index.html"
    _atomic_write_text(index, html)
    return index
