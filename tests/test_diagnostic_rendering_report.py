import csv
import json
from pathlib import Path

import cv2
import numpy as np

from inference_engine.diagnostics.rendering import (
    render_case,
    render_method_comparison,
    write_segment_ply,
)
from inference_engine.diagnostics.report import build_report


def _trace(path, *, include_split=True):
    height, width = 8, 10
    y, x = np.mgrid[:height, :width]
    point_map = np.stack([(x - 5) / 10, (y - 4) / 10, 1 + x / 20], axis=-1)
    labels = (x >= 5).astype(np.int32) + 2 * (y >= 4).astype(np.int32)
    arrays = dict(
        rgb=np.stack([x / width, y / height, np.ones_like(x) * .5], axis=-1),
        point_map=point_map,
        confidence=(x + y).astype(float),
        initial_labels=(x // 2 + 5 * (y // 2)).astype(np.int32),
        coarse_labels=(x >= 5).astype(np.int32),
        final_labels=labels,
        merge_decision=(x == 5).astype(np.uint8),
        source_map=(labels % 3).astype(np.uint8),
        scale_map=1 + labels * .1,
        dispersion_map=labels * .02,
    )
    if include_split:
        changed = (x >= 7) & (y >= 4)
        arrays.update(
            pre_split_labels=(x >= 5).astype(np.int32),
            changed_mask=changed,
            split_score_map=np.where(changed, .35, np.nan).astype(np.float32),
            split_decision_map=np.where(changed, 1, 0).astype(np.uint8),
        )
    np.savez_compressed(path, **arrays)


def test_renderer_writes_complete_deterministic_headless_case(tmp_path):
    trace = tmp_path / "trace.npz"
    _trace(trace)
    first = render_case(trace, tmp_path / "case")
    expected = {
        "rgb", "depth", "confidence", "initial_atoms", "coarse_layers",
        "final_segments", "merge_decisions", "component_growth", "scale_source",
        "scale_map", "scale_dispersion", "temporal", "pointcloud_top", "pointcloud_side",
        "pre_split_segments", "split_changed_regions", "split_scores",
        "split_decisions",
    }
    assert expected <= set(first)
    for path in first.values():
        assert cv2.imread(str(path)) is not None
    before = {name: path.read_bytes() for name, path in first.items()}
    second = render_case(trace, tmp_path / "case")
    assert before == {name: path.read_bytes() for name, path in second.items()}

    rendering = json.loads((tmp_path / "case" / "rendering.json").read_text())
    assert rendering["availability"]["split_score_map"] is True
    assert rendering["legends"]["split_decisions"] == {
        "0": "none",
        "1": "accepted",
        "2": "no-markers",
        "3": "small-child",
        "4": "low-score",
    }


def test_renderer_writes_explicit_unavailable_split_evidence(tmp_path):
    trace = tmp_path / "trace.npz"
    _trace(trace, include_split=False)

    rendered = render_case(trace, tmp_path / "case")

    assert {
        "pre_split_segments", "split_changed_regions", "split_scores",
        "split_decisions",
    } <= set(rendered)
    rendering = json.loads((tmp_path / "case" / "rendering.json").read_text())
    assert rendering["availability"]["pre_split_labels"] is False
    assert rendering["availability"]["changed_mask"] is False
    assert rendering["availability"]["split_score_map"] is False
    assert rendering["availability"]["split_decision_map"] is False


def test_method_comparison_replaces_stale_scale_ratio_with_unavailable_panel(tmp_path):
    labels = np.zeros((4, 6), dtype=np.int32)
    labels[:, 3:] = 1
    methods = {
        "depth": labels,
        "geometry_baseline": labels,
        "layer_atomic_split": labels,
    }
    valid_scales = {
        "geometry_baseline": np.ones(labels.shape),
        "layer_atomic_split": np.ones(labels.shape) * 1.2,
    }
    first = render_method_comparison(
        methods, tmp_path / "comparison", method_scales=valid_scales
    )
    before = first["scale_log_ratio"].read_bytes()

    second = render_method_comparison(
        methods,
        tmp_path / "comparison",
        method_scales={
            "geometry_baseline": np.ones((2, 2)),
            "layer_atomic_split": np.ones((3, 3)),
        },
    )

    assert "scale_log_ratio" in second
    assert second["scale_log_ratio"].read_bytes() != before
    image = cv2.imread(str(second["scale_log_ratio"]))
    assert image.shape[:2] == labels.shape
    rendering = json.loads(
        (tmp_path / "comparison" / "comparison-rendering.json").read_text()
    )
    assert rendering["availability"]["scale_log_ratio"] is False


def test_ply_has_deterministic_vertex_colors_and_count(tmp_path):
    points = np.array([[[0., 0., 1.], [1., 0., 1.]], [[0., 1., 1.], [np.nan, 0., 1.]]])
    labels = np.array([[0, 1], [0, 1]])
    path = write_segment_ply(points, labels, tmp_path / "segments.ply")
    text = path.read_text()
    assert "element vertex 3" in text
    assert "property uchar red" in text
    original = path.read_bytes()
    assert write_segment_ply(points, labels, path).read_bytes() == original


def test_static_report_has_overview_guard_recovery_and_case_links(tmp_path):
    (tmp_path / "cases" / "02" / "000010-000020").mkdir(parents=True)
    case = tmp_path / "cases" / "02" / "000010-000020"
    _trace(case / "trace.npz")
    render_case(case / "trace.npz", case)
    reasons = [
        "trajectory_degradation", "trajectory_improvement", "trajectory_change",
        "split_anomaly", "split_no_trajectory_effect", "matched_control",
    ]
    (case / "metrics.json").write_text(json.dumps({
        "selection_score": 8.2,
        "split_minus_depth_regret": 1.2,
        "split_minus_geometry_regret": -.4,
        "selection_reasons": reasons,
        "largest_segment_ratio": .7,
    }))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "test", "git_commit": "abc", "checkpoint_sha256": "123",
        "status": "complete", "budget": {"max_gib": 50, "warn_gib": 40, "min_free_gib": 10},
    }))
    (tmp_path / "summary.json").write_text(json.dumps({
        "stability_guard": {"passed": True}, "recovery": {"02": .5},
        "official_ranking": {"02": [
            {"config_id": "depth", "ate_rmse": 11.0},
            {"config_id": "geometry_baseline", "ate_rmse": 10.5},
            {"config_id": "layer_atomic_split", "ate_rmse": 10.0},
            {"config_id": "geometry_legacy_reference", "ate_rmse": 9.0},
            {"config_id": "layer_atomic", "ate_rmse": 8.0},
        ]},
        "sequence_metrics": {
            "depth": {"02": {"ate_rmse": 11, "rpe_translation_rmse": .3, "rpe_rotation_rmse_deg": .2}},
            "geometry_baseline": {"02": {"ate_rmse": 10.5, "rpe_translation_rmse": .25, "rpe_rotation_rmse_deg": .15}},
            "layer_atomic_split": {"02": {"ate_rmse": 10, "rpe_translation_rmse": .2, "rpe_rotation_rmse_deg": .1}},
            "geometry_legacy_reference": {"02": {"ate_rmse": 9}},
            "layer_atomic": {"02": {"ate_rmse": 8}},
        },
    }))
    (tmp_path / "selection_records.json").write_text(json.dumps([
        {
            "config_id": "layer_atomic_split", "sequence_id": "02",
            "window_id": 2, "frame_start": 10, "frame_end": 20,
            "split_minus_depth_regret": 1.2,
            "split_minus_geometry_regret": None,
        },
    ]))
    (tmp_path / "selected_intervals.json").write_text(json.dumps([
        {
            "sequence_id": "02", "start_frame": 10, "end_frame": 20,
            "reasons": ["trajectory_degradation", "split_anomaly"],
            "score": 8.2,
        },
    ]))

    index = build_report(tmp_path)
    html = index.read_text()
    assert "Stability Guard" in html
    assert "Recovery" in html
    assert "ATE / RPE" in html
    assert "Selected case ranking" in html
    assert "Correlation" in html
    assert "000010-000020" in html
    assert "layer_atomic_split" in html
    assert "split_minus_depth_regret" in html
    assert "split_minus_geometry_regret" in html
    assert all(reason in html for reason in reasons)
    assert "geometry_legacy_reference" not in html
    assert ">layer_atomic</" not in html
    assert "http://" not in html and "https://" not in html
    pages = list((index.parent / "cases").glob("*.html"))
    assert len(pages) == 1
    detail = pages[0].read_text()
    assert "Segmentation → Merge → Scale → Trajectory" in detail
    assert "final_segments.png" in detail
    assert "split_changed_regions.png" in detail
    assert "split_minus_depth_regret" in detail
    assert "split_minus_geometry_regret" in detail
    summary_export = json.loads((index.parent / "summary.json").read_text())
    assert set(summary_export["sequence_metrics"]) == {
        "depth", "geometry_baseline", "layer_atomic_split",
    }
    diagnostics = summary_export["selection_diagnostics"]
    assert diagnostics["reason_counts"] == {
        "split_anomaly": 1,
        "trajectory_degradation": 1,
    }
    assert diagnostics["selection_reasons"] == [
        "split_anomaly", "trajectory_degradation",
    ]
    assert diagnostics["records"] == [{
        "config_id": "layer_atomic_split",
        "sequence_id": "02",
        "window_id": 2,
        "frame_start": 10,
        "frame_end": 20,
        "split_minus_depth_regret": 1.2,
        "split_minus_geometry_regret": None,
        "selection_reasons": ["split_anomaly", "trajectory_degradation"],
    }]
    with (index.parent / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{
        "config_id": "layer_atomic_split",
        "sequence_id": "02",
        "window_id": "2",
        "frame_start": "10",
        "frame_end": "20",
        "split_minus_depth_regret": "1.2",
        "split_minus_geometry_regret": "",
        "selection_reasons": '["split_anomaly","trajectory_degradation"]',
    }]
    with (index.parent / "sequence_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        trajectory_rows = list(csv.DictReader(handle))
    assert {row["config_id"] for row in trajectory_rows} == {
        "depth", "geometry_baseline", "layer_atomic_split",
    }
