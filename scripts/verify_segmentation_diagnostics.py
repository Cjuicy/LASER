#!/usr/bin/env python3
"""CPU-only end-to-end verification that needs no model weights or KITTI data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

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
from inference_engine.diagnostics.selection import (
    REQUIRED_COVERAGE_REASONS,
    select_intervals,
    select_intervals_with_coverage,
)
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


def _accepted_staged_split_fixture():
    """Create a deterministic accepted split through both production entry points."""
    height, width = 24, 32
    y, x = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((x, y, np.ones_like(x)), axis=-1)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[:, width // 2:] = 1.0
    normals = np.zeros_like(points)
    normals[:, : width // 2, 2] = 1.0
    normals[:, width // 2 :, 0] = 1.0
    kwargs = {
        "depth_merge_thresh": .1,
        "rgb_images": rgb,
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 20,
        "normal_method": "cross",
        "split_score_thresh": .10,
        "split_aux_confirmation": True,
    }
    with patch(
        "inference_engine.utils.post_merge_split._normal_map",
        return_value=(normals, np.ones((height, width), dtype=bool)),
    ):
        stages = segment_point_map_layer_atomic_split_stages(points, **kwargs)
        public = segment_point_map_layer_atomic_split(points, **kwargs)
    np.testing.assert_array_equal(stages.final_labels, public)
    assert stages.split_trace.diagnostics.split_accepted_count > 0
    assert np.any(stages.split_trace.changed_mask)
    assert np.any(stages.split_trace.decision_map == 1)
    return points, rgb, np.ones((height, width), dtype=np.float32), stages


def _single_regret_records(field: str) -> list[dict]:
    if field not in {
        "split_minus_depth_regret", "split_minus_geometry_regret",
    }:
        raise ValueError(f"unexpected regret field: {field}")
    records = []
    for window, extreme in enumerate((0.0, 20.0, 0.0)):
        for config_id in METHODS:
            records.append({
                "config_id": config_id, "sequence_id": "01", "window_id": window,
                "frame_start": window * 10, "frame_end": window * 10 + 9,
                "split_minus_depth_regret": (
                    extreme if field == "split_minus_depth_regret" else 0.0
                ),
                "split_minus_geometry_regret": (
                    extreme if field == "split_minus_geometry_regret" else 0.0
                ),
                "split_accepted_count": 0, "split_changed_pixel_ratio": 0.0,
                "merge_anomaly": 0.0, "atom_anomaly": 0.0,
                "scale_dispersion": 0.0, "temporal_churn": 0.0,
                "gt_speed": 1.0, "gt_turn": 0.0, "confidence": .9,
            })
    return records


def _render_case_arrays(
    *, points, rgb, confidence, initial_labels, coarse_labels,
    pre_split_labels, final_labels, split_trace,
) -> dict[str, np.ndarray]:
    changed = split_trace.changed_mask.astype(np.uint8)
    return {
        "rgb": rgb, "point_map": points[None], "confidence": confidence[None],
        "initial_labels": initial_labels, "coarse_labels": coarse_labels,
        "pre_split_labels": pre_split_labels, "final_labels": final_labels,
        "changed_mask": changed, "split_score_map": split_trace.score_map,
        "split_decision_map": split_trace.decision_map,
        "atom_labels": split_trace.parent_map,
        "atom_scales": np.ones(int(split_trace.parent_map.max()) + 1),
        "split_parent_map": split_trace.parent_map,
        "split_child_map": split_trace.child_map,
        "merge_decision": np.where(changed, 2, 0).astype(np.uint8),
        "source_map": np.ones_like(final_labels, dtype=np.uint8),
        "scale_map": np.ones_like(final_labels, dtype=np.float32) * 1.1,
        "dispersion_map": np.ones_like(final_labels, dtype=np.float32) * .03,
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
        {
            "profiles": [
                {"config_id": method, "effective_parameters": {}}
                for method in METHODS
            ],
            "sequences": ["02"],
            "window_size": 20,
            "overlap": 5,
            "evaluation_signature": {
                "align": True, "correct_scale": True,
                "rpe_delta": 1, "all_pairs": True,
            },
        },
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

    dense_points, dense_rgb, dense_confidence, dense_stages = _accepted_staged_split_fixture()
    dense_trace = dense_stages.split_trace
    dense_initial = dense_stages.initial_labels
    print("Verified accepted staged/public split parity.")
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
    dense_arrays = _render_case_arrays(
        points=dense_points, rgb=dense_rgb, confidence=dense_confidence,
        initial_labels=dense_initial, coarse_labels=dense_stages.coarse_labels,
        pre_split_labels=dense_stages.pre_split_labels, final_labels=dense_trace.labels,
        split_trace=dense_trace,
    )
    sink = FileDiagnosticSink(output / "artifacts", budget=StorageBudget(output, max_bytes=50_000_000, warn_bytes=40_000_000, min_free_bytes=0))
    sink.emit_segmentation(context, 0, summarize_labels(dense_trace.labels, dense_initial), dense_arrays)
    sink.close()
    _pass("storage")

    records = _selection_records()
    selected, coverage = select_intervals_with_coverage(records, limit=48)
    assert selected and ALL_REASONS <= {reason for item in selected for reason in item.reasons}
    assert set(coverage) == set(REQUIRED_COVERAGE_REASONS)
    assert all(item["available"] and item["selected"] for item in coverage.values())
    assert all(
        "split_minus_depth_regret" in record and "split_minus_geometry_regret" in record
        for record in records
    )
    geometry_selected = select_intervals(
        _single_regret_records("split_minus_geometry_regret"), context_windows=0
    )
    assert any(
        item.start_frame == 10 and "trajectory_degradation" in item.reasons
        for item in geometry_selected
    )
    depth_selected = select_intervals(
        _single_regret_records("split_minus_depth_regret"), context_windows=0
    )
    assert any(
        item.start_frame == 10 and "trajectory_degradation" in item.reasons
        for item in depth_selected
    )
    print("Verified depth-only regret selection.")
    atomic_write_json(output / "selection_records.json", records)
    atomic_write_json(output / "selected_intervals.json", [item.to_dict() for item in selected])
    _pass("selection")

    selected_case = SelectedInterval("02", 0, 20, tuple(sorted(ALL_REASONS)), 8.0)
    interval_dir = output / "cases" / "02" / "000000-000020"
    arrays = dense_arrays
    for index, method in enumerate(METHODS):
        method_arrays = dict(arrays)
        method_arrays["final_labels"] = (dense_trace.labels + index).astype(np.int32)
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
        {method: dense_trace.labels + index for index, method in enumerate(METHODS)},
        interval_dir,
        method_scales={
            "geometry_baseline": np.ones_like(dense_trace.labels, dtype=float),
            "layer_atomic_split": np.ones_like(dense_trace.labels, dtype=float) * 1.1,
        },
    )
    atomic_write_json(interval_dir / "metrics.json", {
        "selection_score": selected_case.score,
        "selection_reasons": list(selected_case.reasons),
        "split_minus_depth_regret": 8.0,
        "split_minus_geometry_regret": 7.2,
        "largest_segment_ratio": summarize_labels(dense_trace.labels)["largest_segment_ratio"],
        "geometry_split_comparison": {
            "boundary_disagreement_pre": .4,
            "boundary_disagreement_post": .2,
            "boundary_disagreement_delta": -.2,
        },
        "split_structural_summary": {
            "accepted_count": 1,
            "score_mean": .3,
            "changed_pixel_ratio": .5,
            "segment_count_delta": 1,
            "normal_dispersion_gain_mean": .4,
        },
    })
    atomic_write_json(interval_dir / "trajectory-timeline.json", {
        "frame_start": 0,
        "frame_end": 2,
        "errors": {
            "depth": [1.0, 1.2, 1.1],
            "geometry_baseline": [.9, 1.0, 1.0],
            "layer_atomic_split": [1.1, .8, 1.2],
        },
        "regrets": {
            "split_minus_depth_regret": [.1, -.4, .1],
            "split_minus_geometry_regret": [.2, -.2, .2],
        },
    })
    assert (interval_dir / "layer_atomic_split" / "split_decisions.png").is_file()
    assert (interval_dir / "comparison-rendering.json").is_file()
    print("Verified accepted split dense evidence.")
    _pass("rendering")

    manifest.status = "complete"
    atomic_write_json(output / "manifest.json", manifest.to_dict())
    atomic_write_json(output / "summary.json", {
        "stability_guard": {"passed": True}, "recovery": {"02": {"valid": True, "score": .5}},
        "selection_coverage": coverage,
        "official_ranking": {
            "02": [
                {"config_id": "layer_atomic_split", "ate_rmse": 1.0},
                {"config_id": "geometry_baseline", "ate_rmse": 1.1},
                {"config_id": "depth", "ate_rmse": 1.2},
            ],
        },
        "official_aggregate": {
            "depth": {"mean_ate": 1.2, "median_ate": 1.2, "wins": 0, "max_sequence_regression": 0.0},
            "geometry_baseline": {"mean_ate": 1.1, "median_ate": 1.1, "wins": 0, "max_sequence_regression": -.08},
            "layer_atomic_split": {"mean_ate": 1.0, "median_ate": 1.0, "wins": 1, "max_sequence_regression": -.17},
        },
        "sequence_metrics": {
            "depth": {"02": {"ate_rmse": 1.2, "rpe_translation_rmse": .2, "rpe_rotation_rmse_deg": .2, "per_frame_translation_error": [1.0, 1.2, 1.1]}},
            "geometry_baseline": {"02": {"ate_rmse": 1.1, "rpe_translation_rmse": .15, "rpe_rotation_rmse_deg": .15, "per_frame_translation_error": [.9, 1.0, 1.0]}},
            "layer_atomic_split": {"02": {"ate_rmse": 1.0, "rpe_translation_rmse": .1, "rpe_rotation_rmse_deg": .1, "per_frame_translation_error": [1.1, .8, 1.2]}},
            "geometry_legacy_reference": {"02": {"ate_rmse": .9, "rpe_translation_rmse": .05, "rpe_rotation_rmse_deg": .05}},
        },
    })
    report = build_report(output)
    html = report.read_text(encoding="utf-8")
    report_summary = json.loads((output / "report" / "summary.json").read_text(encoding="utf-8"))
    assert set(report_summary["sequence_metrics"]) == set(METHODS)
    assert all(method in html for method in METHODS)
    assert "geometry_legacy_reference" not in html and ">layer_atomic<" not in html
    print("Verified strict report rejects an injected fourth method.")
    _pass("report")
    print(f"Synthetic three-method verification complete (schema {SCHEMA_VERSION}; three methods: {', '.join(METHODS)}).")
    print("Budget defaults for cloud runs: warning=40 GiB, hard=50 GiB, free-space reserve=10 GiB.")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
