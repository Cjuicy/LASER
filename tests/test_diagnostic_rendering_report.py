import json
from pathlib import Path

import cv2
import numpy as np

from inference_engine.diagnostics.rendering import render_case, write_segment_ply
from inference_engine.diagnostics.report import build_report


def _trace(path):
    height, width = 8, 10
    y, x = np.mgrid[:height, :width]
    point_map = np.stack([(x - 5) / 10, (y - 4) / 10, 1 + x / 20], axis=-1)
    labels = (x >= 5).astype(np.int32) + 2 * (y >= 4).astype(np.int32)
    np.savez_compressed(
        path,
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


def test_renderer_writes_complete_deterministic_headless_case(tmp_path):
    trace = tmp_path / "trace.npz"
    _trace(trace)
    first = render_case(trace, tmp_path / "case")
    expected = {
        "rgb", "depth", "confidence", "initial_atoms", "coarse_layers",
        "final_segments", "merge_decisions", "component_growth", "scale_source",
        "scale_map", "scale_dispersion", "temporal", "pointcloud_top", "pointcloud_side",
    }
    assert expected <= set(first)
    for path in first.values():
        assert cv2.imread(str(path)) is not None
    before = {name: path.read_bytes() for name, path in first.items()}
    second = render_case(trace, tmp_path / "case")
    assert before == {name: path.read_bytes() for name, path in second.items()}


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
    (case / "metrics.json").write_text(json.dumps({"trajectory_regret": 8.2, "largest_segment_ratio": .7}))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "test", "git_commit": "abc", "checkpoint_sha256": "123",
        "status": "complete", "budget": {"max_gib": 50, "warn_gib": 40, "min_free_gib": 10},
    }))
    (tmp_path / "summary.json").write_text(json.dumps({
        "stability_guard": {"passed": True}, "recovery": {"02": .5},
        "official_ranking": {"02": [{"config_id": "layer_atomic", "ate_rmse": 10.0}]},
        "sequence_metrics": {"layer_atomic": {"02": {"ate_rmse": 10, "rpe_translation_rmse": .2, "rpe_rotation_rmse_deg": .1}}},
    }))

    index = build_report(tmp_path)
    html = index.read_text()
    assert "Stability Guard" in html
    assert "Recovery" in html
    assert "ATE / RPE" in html
    assert "Selected case ranking" in html
    assert "Correlation" in html
    assert "000010-000020" in html
    assert "http://" not in html and "https://" not in html
    pages = list((index.parent / "cases").glob("*.html"))
    assert len(pages) == 1
    detail = pages[0].read_text()
    assert "Segmentation → Merge → Scale → Trajectory" in detail
    assert "final_segments.png" in detail
