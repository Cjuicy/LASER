"""Offline two-level HTML/CSV/JSON report builder."""

from __future__ import annotations

import csv
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np


STYLE = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f4f7fb;color:#172033}main{max-width:1320px;margin:auto;padding:28px}
h1,h2{margin:.35em 0}.card{background:white;border:1px solid #d9e2ef;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 2px 8px #1720330d}
table{border-collapse:collapse;width:100%}th,td{padding:8px 10px;border-bottom:1px solid #e4e9f1;text-align:left}th{background:#edf3fa}
.ok{color:#087f5b;font-weight:700}.bad{color:#c92a2a;font-weight:700}.warn{background:#fff3bf;padding:10px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px}.grid img{width:100%;border:1px solid #d9e2ef;border-radius:6px}code{background:#edf3fa;padding:2px 5px;border-radius:4px}a{color:#2459a9}
"""


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


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


def _case_page(case_dir: Path, report_cases: Path, run_dir: Path) -> tuple[str, float]:
    metrics = _load(case_dir / "metrics.json", {})
    relative = case_dir.relative_to(run_dir)
    page_name = "-".join(relative.parts) + ".html"
    image_names = [
        "rgb.png", "initial_atoms.png", "coarse_layers.png", "final_segments.png",
        "merge_decisions.png", "component_growth.png", "scale_source.png",
        "scale_map.png", "scale_dispersion.png", "temporal.png", "pointcloud_top.png", "pointcloud_side.png",
    ]
    images = "".join(
        f'<figure><img src="../../{escape(str(relative / name))}" alt="{escape(name)}"><figcaption>{escape(name)}</figcaption></figure>'
        for name in image_names if (case_dir / name).exists()
    )
    html = f"<!doctype html><meta charset='utf-8'><style>{STYLE}</style><main><a href='../index.html'>← Overview</a><h1>{escape(str(relative))}</h1><h2>Segmentation → Merge → Scale → Trajectory</h2><div class='card'><pre>{escape(json.dumps(metrics, indent=2, ensure_ascii=False))}</pre></div><div class='grid'>{images}</div></main>"
    report_cases.mkdir(parents=True, exist_ok=True)
    (report_cases / page_name).write_text(html, encoding="utf-8")
    score = float(metrics.get("trajectory_regret", metrics.get("score", 0)) or 0)
    return page_name, score


def _write_csv(run_dir: Path, summary: dict):
    path = run_dir / "report" / "sequence_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config_id", "sequence_id", "ATE_RMSE", "RPE_translation_RMSE", "RPE_rotation_RMSE_deg"])
        for config, sequences in sorted(summary.get("sequence_metrics", {}).items()):
            for sequence, metrics in sorted(sequences.items()):
                writer.writerow([config, sequence, metrics.get("ate_rmse"), metrics.get("rpe_translation_rmse"), metrics.get("rpe_rotation_rmse_deg")])


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"; report_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load(run_dir / "manifest.json", {})
    summary = _load(run_dir / "summary.json", {})
    _write_csv(run_dir, summary)
    cases = []
    for metrics_path in sorted((run_dir / "cases").glob("**/metrics.json")) if (run_dir / "cases").exists() else []:
        case_dir = metrics_path.parent
        page, score = _case_page(case_dir, report_dir / "cases", run_dir)
        cases.append((score, str(case_dir.relative_to(run_dir)), page))
    cases.sort(key=lambda item: (-item[0], item[1]))
    case_rows = "".join(f"<tr><td>{rank}</td><td><a href='cases/{escape(page)}'>{escape(label)}</a></td><td>{score:.3g}</td></tr>" for rank, (score, label, page) in enumerate(cases, 1)) or "<tr><td colspan='3'>No selected cases rendered.</td></tr>"
    guard = summary.get("stability_guard", {})
    passed = guard.get("passed")
    guard_text = "PASS" if passed else "FAIL / incomplete"
    warning = "" if manifest.get("status") == "complete" else "<p class='warn'>Incomplete run: available artifacts are shown, but official comparison is not complete.</p>"
    recovery = escape(json.dumps(summary.get("recovery", summary.get("recovery_gap", {})), ensure_ascii=False, indent=2))
    provenance = escape(json.dumps({key: manifest.get(key) for key in ("run_id", "git_commit", "checkpoint_sha256", "budget", "status")}, ensure_ascii=False, indent=2))
    heatmap = _svg_heatmap(summary.get("sequence_metrics", {}))
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>LASER segmentation diagnostics</title><style>{STYLE}</style></head><body><main>
<h1>LASER segmentation diagnostics</h1>{warning}
<div class='card'><h2>Run provenance &amp; storage budget</h2><pre>{provenance}</pre></div>
<div class='grid'><section class='card'><h2>Stability Guard</h2><p class='{"ok" if passed else "bad"}'>{guard_text}</p><pre>{escape(json.dumps(guard, ensure_ascii=False, indent=2))}</pre></section><section class='card'><h2>Recovery</h2><pre>{recovery}</pre></section></div>
<section class='card'><h2>ATE / RPE overview heatmap</h2>{heatmap}<p>One global Sim(3) alignment per sequence; RPE delta=1 frame, all pairs.</p></section>
<section class='card'><h2>Error timeline and selected intervals</h2><p>The ranked cases link trajectory regret to segmentation, merge, scale and temporal evidence.</p></section>
<section class='card'><h2>Selected case ranking</h2><table><tr><th>#</th><th>Case</th><th>Score / regret</th></tr>{case_rows}</table></section>
<section class='card'><h2>Correlation &amp; lag diagnostics</h2><p>Correlation tables (Pearson/Spearman, lag 0–3) support checking hypotheses; they do not establish causality. Missing rates and sample counts remain explicit in exported JSON.</p></section>
<section class='card'><h2>Exports</h2><a href='sequence_metrics.csv'>sequence_metrics.csv</a></section>
</main></body></html>"""
    index = report_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index
