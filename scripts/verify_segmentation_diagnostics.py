#!/usr/bin/env python3
"""CPU-only end-to-end verification that needs no model weights or KITTI data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine.diagnostics.merge import analyze_layer_atomic_merge
from inference_engine.diagnostics.rendering import render_case
from inference_engine.diagnostics.report import build_report
from inference_engine.diagnostics.schema import DiagnosticContext, SelectedInterval
from inference_engine.diagnostics.segmentation import summarize_labels
from inference_engine.diagnostics.selection import select_intervals
from inference_engine.diagnostics.sink import FileDiagnosticSink
from inference_engine.diagnostics.storage import StorageBudget, atomic_write_json
from inference_engine.utils.layer_atomic_geometry import merge_layer_atoms


def _pass(name: str):
    print(f"[PASS] {name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="diagnostic-verification-output")
    args = parser.parse_args(argv)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)

    context1 = DiagnosticContext("synthetic", "layer_atomic", "02", 1, 0, 0)
    assert DiagnosticContext.from_dict(context1.to_dict()) == context1
    _pass("schema")

    top = np.asarray([(x, 0.0, 1.0) for x in (0, 1, 2.05, 3.05, 7, 8)], dtype=float)
    points = np.stack([top + (0, y, 0) for y in range(4)])
    atoms = np.tile(np.asarray([0, 0, 1, 1, 2, 2]), (4, 1))
    coarse = np.tile(np.asarray([0, 0, 0, 0, 1, 1]), (4, 1))
    formal = merge_layer_atoms(points, atoms, coarse, .1)
    trace = analyze_layer_atomic_merge(points, atoms, coarse, .1, formal)
    np.testing.assert_array_equal(trace.final_labels, formal)
    _pass("parity")

    budget = StorageBudget(output, max_bytes=50_000_000, warn_bytes=40_000_000, min_free_bytes=0)
    assert budget.state(used_bytes=39_999_999).level == "ok"
    assert budget.state(used_bytes=40_000_000).level == "warning"
    sink1 = FileDiagnosticSink(output / "artifacts", budget=budget)
    sink1.emit_segmentation(context1, 0, summarize_labels(formal, atoms), {"final_labels": formal})
    sink1.close()
    _pass("storage")

    records = []
    for sequence in ("00", "02", "04", "05", "09", "10"):
        for window in range(3):
            spike = 8.0 if window == 1 else .1
            records.append({
                "config_id": "layer_atomic", "sequence_id": sequence,
                "window_id": window, "frame_start": window * 5, "frame_end": window * 5 + 9,
                "trajectory_regret": spike, "merge_anomaly": spike * .8,
                "scale_dispersion": spike * .6, "temporal_churn": spike * .4,
                "gt_speed": window + 1, "gt_turn": .1 * window, "confidence": .9,
            })
    selected = select_intervals(records, limit=48)
    assert selected and any(item.sequence_id == "02" for item in selected)
    _pass("selection")

    selected_case = SelectedInterval("02", 0, 20, ("trajectory", "merge_atom", "scale", "temporal"), 8.0)
    context2 = DiagnosticContext("synthetic", "layer_atomic", "02", 2, 0, 0)
    sink2 = FileDiagnosticSink(output / "artifacts", selected_intervals=[selected_case], budget=budget)
    rgb = np.zeros((1, 4, 6, 3), dtype=np.float32)
    rgb[0, ..., 0] = np.linspace(0, 1, 6)
    confidence = np.linspace(0, 1, 24).reshape(1, 4, 6)
    sink2.emit_inputs(context2, rgb, points[None], confidence)
    sink2.emit_segmentation(context2, 0, {"final": summarize_labels(formal, atoms), "merge": trace.metrics}, {
        "initial_labels": atoms, "coarse_labels": coarse, "final_labels": formal,
        "merge_decision": (coarse != formal).astype(np.uint8),
    })
    sink2.emit_scale(context2, {"source": "synthetic"}, {
        "source_map": np.ones((1, 4, 6), np.uint8),
        "scale_map": np.ones((1, 4, 6), np.float32) * 1.1,
        "dispersion_map": np.ones((1, 4, 6), np.float32) * .03,
    })
    sink2.close()
    trace_dir = output / "artifacts" / "layer_atomic" / "02" / "pass2" / "traces"
    case_dir = output / "cases" / "02" / "000000-000020" / "layer_atomic"
    render_case(sorted(trace_dir.glob("*.npz")), case_dir)
    atomic_write_json(case_dir / "metrics.json", {"trajectory_regret": 8.0, "largest_segment_ratio": summarize_labels(formal)["largest_segment_ratio"]})
    assert (case_dir / "final_segments.png").exists() and (case_dir / "segments.ply").exists()
    _pass("rendering")

    atomic_write_json(output / "manifest.json", {
        "run_id": "synthetic", "git_commit": "verification", "checkpoint_sha256": "not-required",
        "status": "complete", "budget": {"max_gib": 50, "warn_gib": 40, "min_free_gib": 10},
    })
    atomic_write_json(output / "summary.json", {
        "stability_guard": {"passed": True}, "recovery": {"02": .5},
        "sequence_metrics": {"layer_atomic": {"02": {"ate_rmse": 1.0, "rpe_translation_rmse": .1, "rpe_rotation_rmse_deg": .1}}},
    })
    report = build_report(output)
    html = report.read_text(encoding="utf-8")
    assert "Stability Guard" in html and "000000-000020" in html
    _pass("report")
    print("Synthetic two-pass verification complete.")
    print("Budget defaults for cloud runs: warning=40 GiB, hard=50 GiB, free-space reserve=10 GiB.")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
