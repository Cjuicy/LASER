from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "tools" / "disposable_experiments"
SUMMARY_PATH = SCRIPT_ROOT / "summarize_results.py"

POINTCLOUD_METHODS = (
    "depth",
    "geometry",
    "atomic-original",
    "atomic-split-assisted",
    "atomic-split-no-assisted",
)
ATE_METHODS = (
    "depth-traditional",
    "depth-corrected",
    "geometry-corrected",
    "atomic-original-corrected",
    "atomic-split-assisted-corrected",
    "atomic-split-no-assisted-corrected",
)


def load_module():
    spec = importlib.util.spec_from_file_location("disposable_summary", SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def pc_method(method):
    table = {
        "depth": ("depth", "none"),
        "geometry": ("geometry", "none"),
        "atomic-original": ("atomic", "none"),
        "atomic-split-assisted": ("atomic", "conservative"),
        "atomic-split-no-assisted": ("atomic", "normal_only"),
    }
    return table[method]


def ate_method(method):
    table = {
        "depth-traditional": ("depth", "none", "traditional"),
        "depth-corrected": ("depth", "none", "corrected"),
        "geometry-corrected": ("geometry", "none", "corrected"),
        "atomic-original-corrected": ("atomic", "none", "corrected"),
        "atomic-split-assisted-corrected": (
            "atomic",
            "conservative",
            "corrected",
        ),
        "atomic-split-no-assisted-corrected": (
            "atomic",
            "normal_only",
            "corrected",
        ),
    }
    return table[method]


def write_complete_fixture(root: Path, seven_map: dict, nrgbd_map: dict) -> Path:
    metric_root = root / "metrics"
    prediction_index = 1
    for dataset, mapping in (("7scenes", seven_map), ("nrgbd", nrgbd_map)):
        for scene in mapping:
            safe_scene = scene.replace("/", "__")
            for method_index, method in enumerate(POINTCLOUD_METHODS):
                segmentation, split_mode = pc_method(method)
                value = 0.01 + method_index * 0.001
                payload = {
                    "schema_version": 1,
                    "identity": {
                        "evaluation": "pointcloud",
                        "dataset": dataset,
                        "scene": scene,
                        "method": method,
                        "segmentation_method": segmentation,
                        "split_mode": split_mode,
                        "reconstruction_mode": "no_loop",
                        "sample_stride": 1,
                        "window_size": 20,
                        "overlap": 5,
                        "registration_confidence_keep_ratio": 0.5,
                    },
                    "prediction_key": f"{prediction_index:064x}",
                    "frame_count": len(mapping[scene]),
                    "metrics": {
                        "accuracy_mean_m": value,
                        "accuracy_median_m": value + 0.001,
                        "completion_mean_m": value + 0.002,
                        "completion_median_m": value + 0.003,
                        "normal_consistency_mean": 0.8,
                        "normal_consistency_median": 0.9,
                        "chamfer_l1_m": value + 0.001,
                        "thresholds": [
                            {
                                "threshold_m": threshold,
                                "precision": 0.7,
                                "recall": 0.6,
                                "fscore": 0.646153846,
                            }
                            for threshold in (0.01, 0.02, 0.05)
                        ],
                    },
                }
                write_json(
                    metric_root
                    / "pointcloud"
                    / dataset
                    / safe_scene
                    / f"{method}.json",
                    payload,
                )
            prediction_index += 1

    for sequence_index in range(11):
        sequence = f"{sequence_index:02d}"
        for method_index, method in enumerate(ATE_METHODS):
            segmentation, split_mode, reconstruction = ate_method(method)
            payload = {
                "schema_version": 1,
                "identity": {
                    "evaluation": "ate",
                    "dataset": "kitti",
                    "scene": sequence,
                    "method": method,
                    "segmentation_method": segmentation,
                    "split_mode": split_mode,
                    "reconstruction_mode": reconstruction,
                    "sample_stride": 1,
                    "window_size": 75,
                    "overlap": 30,
                    "registration_confidence_keep_ratio": 0.5,
                },
                "prediction_key": f"{prediction_index:064x}",
                "frame_count": 100,
                "metrics": {
                    "ate_rmse_m": 1.0 + method_index,
                    "rpe_translation_rmse_m": 0.1 + method_index * 0.01,
                    "rpe_rotation_rmse_deg": 0.2 + method_index * 0.01,
                    "matched_frame_count": 100,
                },
            }
            write_json(
                metric_root / "ate" / "kitti" / sequence / f"{method}.json",
                payload,
            )
        prediction_index += 1
    return metric_root


def tiny_maps():
    return {"chess/seq-03": [0, 10]}, {"breakfast_room": [0, 10]}


def test_collection_has_exact_coverage_and_numeric_metrics(tmp_path):
    module = load_module()
    seven, nrgbd = tiny_maps()
    metric_root = write_complete_fixture(tmp_path, seven, nrgbd)

    pointcloud = module.collect_pointcloud_rows(metric_root, seven, nrgbd)
    ate = module.collect_ate_rows(metric_root)

    assert len(pointcloud) == 2 * 5
    assert len(ate) == 11 * 6
    assert isinstance(pointcloud[0]["accuracy_mean_m"], float)
    assert isinstance(ate[0]["matched_frame_count"], int)
    assert pointcloud[0]["dataset"] == "7scenes"
    assert ate[-1]["sequence"] == "10"


def test_collection_rejects_missing_method(tmp_path):
    module = load_module()
    seven, nrgbd = tiny_maps()
    metric_root = write_complete_fixture(tmp_path, seven, nrgbd)
    missing = (
        metric_root
        / "pointcloud"
        / "nrgbd"
        / "breakfast_room"
        / "atomic-split-no-assisted.json"
    )
    missing.unlink()

    with pytest.raises(ValueError, match="missing"):
        module.collect_pointcloud_rows(metric_root, seven, nrgbd)


def test_summary_writes_long_csv_and_two_populated_workbooks(tmp_path):
    module = load_module()
    seven, nrgbd = tiny_maps()
    metric_root = write_complete_fixture(tmp_path, seven, nrgbd)
    seven_map = tmp_path / "seven.json"
    nrgbd_map = tmp_path / "nrgbd.json"
    write_json(seven_map, seven)
    write_json(nrgbd_map, nrgbd)
    output = tmp_path / "summary"

    exit_code = module.main(
        [
            "--metric-root",
            str(metric_root),
            "--seven-map",
            str(seven_map),
            "--nrgbd-map",
            str(nrgbd_map),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    with (output / "pointcloud_long.csv").open(newline="", encoding="utf-8-sig") as source:
        pointcloud_rows = list(csv.DictReader(source))
    with (output / "kitti_ate_long.csv").open(newline="", encoding="utf-8-sig") as source:
        ate_rows = list(csv.DictReader(source))
    assert len(pointcloud_rows) == 10
    assert len(ate_rows) == 66
    assert pointcloud_rows[0]["accuracy_mean_m"] == "0.01"
    assert ate_rows[0]["ate_rmse_m"] == "1.0"
    pointcloud_workbook = output / "pointcloud_summary.xlsx"
    ate_workbook = output / "kitti_ate_summary.xlsx"
    assert pointcloud_workbook.stat().st_size > 5000
    assert ate_workbook.stat().st_size > 5000
    with zipfile.ZipFile(pointcloud_workbook) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "7-Scenes" in workbook_xml
    assert "Raw Data" in workbook_xml
    assert "Acc mean : 0.010000 m" in sheet_xml
    with zipfile.ZipFile(ate_workbook) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "KITTI 00-10" in workbook_xml
    assert "ATE RMSE : 1.000000 m" in sheet_xml
