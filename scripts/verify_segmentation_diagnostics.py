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
from inference_engine.diagnostics.rendering import render_case, render_method_comparison
from inference_engine.diagnostics.report import build_report
from inference_engine.diagnostics.schema import (
    SCHEMA_VERSION,
    DiagnosticContext,
    RunManifest,
    SelectedInterval,
)
from inference_engine.diagnostics.segmentation import summarize_labels
from inference_engine.diagnostics.selection import select_intervals
from inference_engine.diagnostics.sink import FileDiagnosticSink
from inference_engine.diagnostics.storage import (
    FreeSpaceReserveExceeded,
    StorageBudget,
    StorageLimitExceeded,
    atomic_write_json,
)
from inference_engine.utils.layer_atomic_geometry import (
    merge_layer_atoms,
    segment_point_map_layer_atomic_split,
    segment_point_map_layer_atomic_split_stages,
)


METHODS = ("depth", "geometry_baseline", "layer_atomic_split")
ALL_REASONS = {
    "trajectory_degradation", "trajectory_improvement", "trajectory_change",
    "split_anomaly", "split_no_trajectory_effect", "matched_control",
    "merge", "immutable_atom", "scale", "temporal",
}


def _pass(name: str) -> None:
    print(f"[PASS] {name}")


def _selection_records() -> list[dict]:
    """Synthetic records that deliberately cover every selector reason."""
    records = []
    for sequence in ("00", "02", "04", "05", "09", "10"):
        regrets = [(-4.0, -2.0), (0.0, 0.0), (5.0, 3.0), (5.2, 3.1),
                   (0.1, 0.1), (0.2, 0.2), (1.0, 0.8), (0.3, 0.3)]
        split_activity = [(0, 0.0), (0, 0.0), (1, .15), (0, 0.0),
                          (2, .35), (0, 0.0), (10, .90), (0, 0.0)]
        for window, ((depth_regret, geometry_regret), (accepted, changed)) in enumerate(
            zip(regrets, split_activity, strict=True)
        ):
            common = {
                "sequence_id": sequence, "window_id": window,
                "frame_start": window * 5, "frame_end": window * 5 + 9,
                "split_minus_depth_regret": depth_regret,
                "split_minus_geometry_regret": geometry_regret,
                "merge_anomaly": .1 * window, "atom_anomaly": .08 * window,
                "scale_dispersion": .06 * window, "temporal_churn": .04 * window,
                "gt_speed": window + 1, "gt_turn": .1 * window,
                "confidence": .9 - .01 * window,
            }
            for config_id in METHODS:
                is_split = config_id == "layer_atomic_split"
                records.append({
                    **common, "config_id": config_id,
                    "split_accepted_count": accepted if is_split else 100,
                    "split_changed_pixel_ratio": changed if is_split else 1.0,
                })
    return records


def _split_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 8, 10
    y, x = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((x, y, 1.0 + x / width), axis=-1)
    rgb = np.stack((x / width, y / height, np.full_like(x, .5)), axis=-1)
    confidence = np.linspace(0, 1, height * width, dtype=np.float32).reshape(height, width)
    return points, rgb, confidence


def _render_case_arrays(stages, points, rgb, confidence) -> dict[str, np.ndarray]:
    changed = stages.split_trace.changed_mask.astype(np.uint8)
    return {
        "rgb": rgb, "point_map": points[None], "confidence": confidence[None],
        "initial_labels": stages.initial_labels, "coarse_labels": stages.coarse_labels,
        "pre_split_labels": stages.pre_split_labels, "final_labels": stages.final_labels,
        "changed_mask": changed, "split_score_map": stages.split_trace.score_map,
        "split_decision_map": stages.split_trace.decision_map,
        "merge_decision": np.where(changed, 2, 0).astype(np.uint8),
        "source_map": np.ones_like(stages.final_labels, dtype=np.uint8),
        "scale_map": np.ones_like(stages.final_labels, dtype=np.float32) * 1.1,
        "dispersion_map": np.ones_like(stages.final_labels, dtype=np.float32) * .03,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU verifier for the strict depth, geometry_baseline, and layer_atomic_split diagnostics contract."
    )
    parser.add_argument("--output-dir", default="diagnostic-verification-output")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    context = DiagnosticContext("synthetic", "layer_atomic_split", "02", 1, 0, 0)
    manifest = RunManifest(
        "synthetic", "verification", "not-required", "synthetic-config",
        "synthetic-data", 0, {}, {"max_gib": 50, "warn_gib": 40, "min_free_gib": 10},
    )
    assert context.to_dict()["schema_version"] == SCHEMA_VERSION == "2.0"
    assert DiagnosticContext.from_dict(context.to_dict()) == context
    assert RunManifest.from_dict(manifest.to_dict()).schema_version == SCHEMA_VERSION
    _pass("schema 2.0")

    top = np.asarray([(x, 0.0, 1.0) for x in (0, 1, 2.05, 3.05, 7, 8)], dtype=float)
    merge_points = np.stack([top + (0, y, 0) for y in range(4)])
    atoms = np.tile(np.asarray([0, 0, 1, 1, 2, 2]), (4, 1))
    coarse = np.tile(np.asarray([0, 0, 0, 0, 1, 1]), (4, 1))
    merged = merge_layer_atoms(merge_points, atoms, coarse, .1)
    merge_trace = analyze_layer_atomic_merge(merge_points, atoms, coarse, .1, merged)
    np.testing.assert_array_equal(merge_trace.final_labels, merged)

    points, rgb, confidence = _split_fixture()
    stages = segment_point_map_layer_atomic_split_stages(
        points, .1, rgb_images=rgb, conf_map=confidence[None], top_conf_percentile=.5,
        seg_scale=2, seg_sigma=0, seg_min_size=2, normal_method="cross",
        split_score_thresh=.10, split_aux_confirmation=True, batch_idx=0,
    )
    public = segment_point_map_layer_atomic_split(
        points, .1, rgb_images=rgb, conf_map=confidence[None], top_conf_percentile=.5,
        seg_scale=2, seg_sigma=0, seg_min_size=2, normal_method="cross",
        split_score_thresh=.10, split_aux_confirmation=True, batch_idx=0,
    )
    np.testing.assert_array_equal(stages.final_labels, public)
    _pass("parity")

    budget = StorageBudget(output, max_bytes=100, warn_bytes=40, min_free_bytes=10)
    assert budget.state(used_bytes=39, free_bytes=100).level == "ok"
    assert budget.enforce(used_bytes=40, free_bytes=100).level == "warning"
    try:
        budget.enforce(used_bytes=101, free_bytes=100)
    except StorageLimitExceeded:
        pass
    else:
        raise AssertionError("hard storage limit was not enforced")
    try:
        budget.enforce(used_bytes=0, free_bytes=10, estimated_bytes=1)
    except FreeSpaceReserveExceeded:
        pass
    else:
        raise AssertionError("free-space reserve was not enforced")
    sink = FileDiagnosticSink(output / "artifacts", budget=StorageBudget(output, max_bytes=50_000_000, warn_bytes=40_000_000, min_free_bytes=0))
    sink.emit_segmentation(context, 0, summarize_labels(stages.final_labels, stages.initial_labels), _render_case_arrays(stages, points, rgb, confidence))
    sink.close()
    _pass("storage")

    records = _selection_records()
    selected = select_intervals(records, limit=48)
    assert selected and ALL_REASONS <= {reason for item in selected for reason in item.reasons}
    assert all(
        "split_minus_depth_regret" in record and "split_minus_geometry_regret" in record
        for record in records
    )
    atomic_write_json(output / "selection_records.json", records)
    atomic_write_json(output / "selected_intervals.json", [item.to_dict() for item in selected])
    _pass("selection")

    selected_case = SelectedInterval("02", 0, 20, tuple(sorted(ALL_REASONS)), 8.0)
    interval_dir = output / "cases" / "02" / "000000-000020"
    arrays = _render_case_arrays(stages, points, rgb, confidence)
    for index, method in enumerate(METHODS):
        method_arrays = dict(arrays)
        method_arrays["final_labels"] = (stages.final_labels + index).astype(np.int32)
        trace_path = output / "artifacts" / method / "02" / "pass2" / "traces" / "000000.npz"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, **method_arrays)
        method_dir = interval_dir / method
        render_case(trace_path, method_dir)
        atomic_write_json(method_dir / "metrics.json", {
            "selection_score": selected_case.score,
            "selection_reasons": list(selected_case.reasons),
            "split_minus_depth_regret": 8.0,
            "split_minus_geometry_regret": 7.2,
        })
    render_method_comparison(
        {method: stages.final_labels + index for index, method in enumerate(METHODS)},
        interval_dir,
        method_scales={
            "geometry_baseline": np.ones_like(stages.final_labels, dtype=float),
            "layer_atomic_split": np.ones_like(stages.final_labels, dtype=float) * 1.1,
        },
    )
    atomic_write_json(interval_dir / "metrics.json", {
        "selection_score": selected_case.score,
        "selection_reasons": list(selected_case.reasons),
        "split_minus_depth_regret": 8.0,
        "split_minus_geometry_regret": 7.2,
        "largest_segment_ratio": summarize_labels(stages.final_labels)["largest_segment_ratio"],
    })
    assert (interval_dir / "layer_atomic_split" / "split_decisions.png").is_file()
    assert (interval_dir / "comparison-rendering.json").is_file()
    _pass("rendering")

    manifest.status = "complete"
    atomic_write_json(output / "manifest.json", manifest.to_dict())
    atomic_write_json(output / "summary.json", {
        "stability_guard": {"passed": True}, "recovery": {"02": {"valid": True, "score": .5}},
        "sequence_metrics": {
            "depth": {"02": {"ate_rmse": 1.2, "rpe_translation_rmse": .2, "rpe_rotation_rmse_deg": .2}},
            "geometry_baseline": {"02": {"ate_rmse": 1.1, "rpe_translation_rmse": .15, "rpe_rotation_rmse_deg": .15}},
            "layer_atomic_split": {"02": {"ate_rmse": 1.0, "rpe_translation_rmse": .1, "rpe_rotation_rmse_deg": .1}},
        },
    })
    report = build_report(output)
    html = report.read_text(encoding="utf-8")
    report_summary = json.loads((output / "report" / "summary.json").read_text(encoding="utf-8"))
    assert set(report_summary["sequence_metrics"]) == set(METHODS)
    assert all(method in html for method in METHODS)
    assert "geometry_legacy_reference" not in html and ">layer_atomic<" not in html
    _pass("report")
    print(f"Synthetic three-method verification complete (schema {SCHEMA_VERSION}; three methods: {', '.join(METHODS)}).")
    print("Budget defaults for cloud runs: warning=40 GiB, hard=50 GiB, free-space reserve=10 GiB.")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
