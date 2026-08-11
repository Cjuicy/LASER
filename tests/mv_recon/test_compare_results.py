import csv
import hashlib
import json
from pathlib import Path

import pytest

from mv_recon.compare_results import (
    compare_run_directories,
    main,
    validate_depth_gate,
)


EXPECTED_NAMES = (
    "breakfast_room",
    "complete_kitchen",
    "green_room",
    "grey_white_room",
    "kitchen",
    "morning_apartment",
    "staircase",
    "thin_geometry",
    "whiteroom",
)
PAPER_PRIMARY = {
    "accuracy_mean_m": 0.0201,
    "accuracy_median_m": 0.0101,
    "completion_mean_m": 0.0121,
    "completion_median_m": 0.0041,
    "normal_consistency_mean": 0.7131,
    "normal_consistency_median": 0.8561,
}
PAPER_REFERENCE = {
    "accuracy_mean_m": 0.020,
    "accuracy_median_m": 0.010,
    "completion_mean_m": 0.012,
    "completion_median_m": 0.004,
    "normal_consistency_mean": 0.713,
    "normal_consistency_median": 0.856,
}
METRIC_VERSION = "laser-pointmap-metrics-v2"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_run(
    run_dir: Path,
    method: str,
    *,
    names: tuple[str, ...] = EXPECTED_NAMES,
    primary: dict[str, float] | None = None,
    chamfer_base: float = 0.016,
    precision_base: float = 0.50,
    recall_base: float = 0.60,
) -> Path:
    selected_primary = dict(primary or PAPER_PRIMARY)
    cache_mode = "auto" if method == "depth" else "readonly"
    sequences = []
    for index, name in enumerate(names):
        thresholds = []
        for threshold_index, threshold in enumerate((0.01, 0.02, 0.05)):
            precision = precision_base + index * 0.01 + threshold_index * 0.10
            recall = recall_base + index * 0.01 + threshold_index * 0.05
            thresholds.append(
                {
                    "threshold_m": threshold,
                    "precision": precision,
                    "recall": recall,
                    "fscore": 2.0 * precision * recall / (precision + recall),
                }
            )
        sequences.append(
            {
                "dataset": "NRGBD-dense",
                "sequence": name,
                "frame_count": 10 + index,
                "input_manifest_sha256": _digest(f"input:{name}"),
                "ground_truth_sha256": _digest(f"ground-truth:{name}"),
                "ordinary_prediction_key": _digest(f"ordinary:{name}"),
                "primary": selected_primary,
                "diagnostics": {
                    "umeyama_scale": 1.0,
                    "icp_transformation": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "icp_fitness": 1.0,
                    "icp_inlier_rmse": 0.001,
                    "predicted_point_count": 100,
                    "ground_truth_point_count": 100,
                    "directional_normals": {
                        "nc1_mean": 0.7,
                        "nc1_median": 0.8,
                        "nc2_mean": 0.7,
                        "nc2_median": 0.8,
                    },
                    "chamfer_l1_m": chamfer_base + index * 0.001,
                    "thresholds": thresholds,
                },
                "cache_diagnostics": {
                    "ordinary_hits": 1 if method != "depth" else 0,
                    "ordinary_misses": 0 if method != "depth" else 1,
                },
            }
        )
    identity = {
        "protocol_identity_sha256": _digest(f"protocol:{method}"),
        "pipeline_sha256": _digest(f"pipeline:{method}"),
        "checkpoint_sha256": "c" * 64,
        "sequence_map_sha256": {"NRGBD-dense": "d" * 64},
        "metric_version": METRIC_VERSION,
    }
    results = {
        "schema_version": METRIC_VERSION,
        "state": "subset",
        "identity": identity,
        "expected_sequences": {"NRGBD-dense": list(names)},
        "full_sequence_counts": {"NRGBD-dense": 9},
        "subset": True,
        "paper_reference": {"NRGBD-dense": PAPER_REFERENCE},
        "sequences": sequences,
        "failures": [],
        "datasets": [
            {
                "dataset": "NRGBD-dense",
                "status": "subset",
                "expected_sequences": 9,
                "selected_sequences": len(names),
                "completed_sequences": len(names),
                "primary": selected_primary,
                "paper_reference": PAPER_REFERENCE,
                "delta_to_paper": {
                    key: value - PAPER_REFERENCE[key]
                    for key, value in selected_primary.items()
                },
            }
        ],
    }
    manifest = {
        "schema_version": 1,
        "metric_schema_version": METRIC_VERSION,
        "git_commit": "test-commit",
        "resolved_protocol_sha256": _digest(f"resolved:{method}"),
        "protocol_identity_sha256": identity["protocol_identity_sha256"],
        "resolved_pipeline_sha256": identity["pipeline_sha256"],
        "evaluation_mode": "comparison",
        "segmentation_method": method,
        "checkpoint_sha256": "c" * 64,
        "sequence_map_sha256": {"NRGBD-dense": "d" * 64},
        "run_state": "subset",
        "runtime": {
            "python": "3.11.15",
            "torch": "2.12.0+cu130",
            "open3d": "0.19.0",
            "numpy": "1.26.4",
            "scipy": "1.17.1",
            "cuda_available": True,
            "cuda_version": "13.0",
            "cuda_capability": [12, 0],
            "gpu_name": "NVIDIA GeForce RTX 5090",
        },
        "pipeline": {
            "prediction_cache": {"mode": cache_mode},
            "segmentation": {"method": method},
        },
        "selected_sequences": {"NRGBD-dense": list(names)},
        "selected_sequence_counts": {"NRGBD-dense": len(names)},
        "expected_sequence_counts": {"NRGBD-dense": 9},
        "attempted_sequences": len(names),
        "successful_sequences": len(names),
        "failed_sequences": 0,
    }
    _write_json(run_dir / "results.json", results)
    _write_json(run_dir / "protocol_manifest.json", manifest)
    return run_dir


def _rewrite_results(run_dir: Path, mutate) -> None:
    path = run_dir / "results.json"
    payload = _read_json(path)
    mutate(payload)
    _write_json(path, payload)


def _rewrite_manifest(run_dir: Path, mutate) -> None:
    path = run_dir / "protocol_manifest.json"
    payload = _read_json(path)
    mutate(payload)
    _write_json(path, payload)


def test_depth_gate_accepts_three_decimal_paper_match(tmp_path):
    run = _write_run(tmp_path / "depth", "depth")

    values = validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)

    assert values["accuracy_mean_m"] == pytest.approx(0.0201)
    assert values["normal_consistency_median"] == pytest.approx(0.8561)


def test_depth_gate_rejects_changed_metric(tmp_path):
    run = _write_run(
        tmp_path / "depth",
        "depth",
        primary={**PAPER_PRIMARY, "accuracy_mean_m": 0.021},
    )

    with pytest.raises(ValueError, match="reproduction gate"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)


@pytest.mark.parametrize("bad_state", ("complete", "incomplete", "failed"))
def test_depth_gate_rejects_wrong_run_state(tmp_path, bad_state):
    run = _write_run(tmp_path / "depth", "depth")
    _rewrite_results(run, lambda payload: payload.update(state=bad_state))

    with pytest.raises(ValueError, match="successful subset"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)


def test_depth_gate_rejects_failures(tmp_path):
    run = _write_run(tmp_path / "depth", "depth")
    _rewrite_results(
        run,
        lambda payload: payload["failures"].append(
            {"dataset": "NRGBD-dense", "sequence": "whiteroom"}
        ),
    )

    with pytest.raises(ValueError, match="failures"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)


@pytest.mark.parametrize("coverage_error", ("missing", "duplicate"))
def test_depth_gate_rejects_bad_sequence_coverage(tmp_path, coverage_error):
    run = _write_run(tmp_path / "depth", "depth")

    def mutate(payload):
        if coverage_error == "missing":
            payload["sequences"].pop()
        else:
            payload["sequences"].append(payload["sequences"][0])

    _rewrite_results(run, mutate)

    with pytest.raises(ValueError, match="sequence coverage"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)


def test_depth_gate_rejects_wrong_method(tmp_path):
    run = _write_run(tmp_path / "depth", "depth")
    _rewrite_manifest(
        run,
        lambda payload: payload.update(segmentation_method="geometry"),
    )

    with pytest.raises(ValueError, match="method mismatch"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)


def test_depth_gate_rejects_nonfinite_metric(tmp_path):
    run = _write_run(tmp_path / "depth", "depth")

    def mutate(payload):
        payload["datasets"][0]["primary"]["accuracy_mean_m"] = float("nan")

    _rewrite_results(run, mutate)

    with pytest.raises(ValueError, match="finite"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)


def test_depth_gate_rejects_results_manifest_identity_disagreement(tmp_path):
    run = _write_run(tmp_path / "depth", "depth")
    _rewrite_manifest(
        run, lambda payload: payload.update(checkpoint_sha256="e" * 64)
    )

    with pytest.raises(ValueError, match="checkpoint.*identity"):
        validate_depth_gate(run, expected_sequences=EXPECTED_NAMES)


def test_full_comparison_writes_macro_metrics_and_deltas(tmp_path):
    depth = _write_run(tmp_path / "depth", "depth")
    geometry = _write_run(
        tmp_path / "geometry",
        "geometry",
        primary={**PAPER_PRIMARY, "accuracy_mean_m": 0.0191},
        chamfer_base=0.015,
        precision_base=0.51,
        recall_base=0.61,
    )
    atomic = _write_run(
        tmp_path / "atomic",
        "atomic",
        primary={**PAPER_PRIMARY, "accuracy_mean_m": 0.0206},
        chamfer_base=0.017,
        precision_base=0.49,
        recall_base=0.59,
    )
    output = tmp_path / "comparison"

    report = compare_run_directories(depth, geometry, atomic, output)

    assert [row["method"] for row in report["methods"]] == [
        "depth",
        "geometry",
        "atomic",
    ]
    assert report["methods"][0]["diagnostics"]["chamfer_l1_m"] \
        == pytest.approx(0.020)
    assert report["methods"][0]["diagnostics"]["thresholds"][0][
        "precision"
    ] == pytest.approx(0.54)
    assert report["methods"][0]["diagnostics"]["thresholds"][0][
        "recall"
    ] == pytest.approx(0.64)
    assert report["methods"][0]["diagnostics"]["thresholds"][0][
        "fscore"
    ] == pytest.approx(0.5857545691677362)
    assert report["methods"][1]["state"] == "subset"
    assert report["methods"][1]["sequence_count"] == 9
    assert report["methods"][1]["diagnostics"]["cache"] == {
        "ordinary_hits": 9,
        "ordinary_misses": 0,
    }
    assert report["methods"][1]["delta_to_depth"][
        "accuracy_mean_m"
    ] == pytest.approx(-0.001)
    assert report["metric_directions"]["accuracy_mean_m"] == "lower"
    assert report["metric_directions"]["normal_consistency_mean"] == "higher"
    assert report["provenance"]["git_commit"] == "test-commit"
    assert report["provenance"]["runtime"]["python"] == "3.11.15"
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.csv").is_file()
    with (output / "comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["method"] for row in rows] == ["depth", "geometry", "atomic"]
    assert float(rows[0]["precision_at_1cm"]) == pytest.approx(0.54)


def test_comparison_rejects_runtime_mismatch(tmp_path):
    depth = _write_run(tmp_path / "depth", "depth")
    geometry = _write_run(tmp_path / "geometry", "geometry")
    atomic = _write_run(tmp_path / "atomic", "atomic")

    def mutate(payload):
        payload["runtime"]["torch"] = "different"

    _rewrite_manifest(geometry, mutate)

    with pytest.raises(ValueError, match="runtime.*mismatch"):
        compare_run_directories(depth, geometry, atomic, tmp_path / "output")


def test_comparison_rejects_different_checkpoint(tmp_path):
    depth = _write_run(tmp_path / "depth", "depth")
    geometry = _write_run(tmp_path / "geometry", "geometry")
    atomic = _write_run(tmp_path / "atomic", "atomic")
    _rewrite_results(
        geometry,
        lambda payload: payload["identity"].update(
            checkpoint_sha256="e" * 64
        ),
    )
    _rewrite_manifest(
        geometry, lambda payload: payload.update(checkpoint_sha256="e" * 64)
    )

    with pytest.raises(ValueError, match="checkpoint.*mismatch"):
        compare_run_directories(depth, geometry, atomic, tmp_path / "output")


def test_comparison_rejects_different_map_hash(tmp_path):
    depth = _write_run(tmp_path / "depth", "depth")
    geometry = _write_run(tmp_path / "geometry", "geometry")
    atomic = _write_run(tmp_path / "atomic", "atomic")

    def change_map(payload):
        payload["sequence_map_sha256"]["NRGBD-dense"] = "e" * 64

    _rewrite_results(geometry, lambda payload: change_map(payload["identity"]))
    _rewrite_manifest(geometry, change_map)

    with pytest.raises(ValueError, match="sequence map.*mismatch"):
        compare_run_directories(depth, geometry, atomic, tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("input_manifest_sha256", "input manifest"),
        ("ground_truth_sha256", "ground truth"),
        ("ordinary_prediction_key", "ordinary prediction"),
    ),
)
def test_comparison_rejects_per_sequence_identity_mismatch(
    tmp_path, field, message
):
    depth = _write_run(tmp_path / "depth", "depth")
    geometry = _write_run(tmp_path / "geometry", "geometry")
    atomic = _write_run(tmp_path / "atomic", "atomic")

    def mutate(payload):
        payload["sequences"][0][field] = "e" * 64

    _rewrite_results(geometry, mutate)

    with pytest.raises(ValueError, match=message):
        compare_run_directories(depth, geometry, atomic, tmp_path / "output")


def test_comparison_rejects_threshold_mismatch(tmp_path):
    depth = _write_run(tmp_path / "depth", "depth")
    geometry = _write_run(tmp_path / "geometry", "geometry")
    atomic = _write_run(tmp_path / "atomic", "atomic")

    def mutate(payload):
        payload["sequences"][0]["diagnostics"]["thresholds"][0][
            "threshold_m"
        ] = 0.015

    _rewrite_results(geometry, mutate)

    with pytest.raises(ValueError, match="threshold"):
        compare_run_directories(depth, geometry, atomic, tmp_path / "output")


def test_depth_gate_cli_writes_atomic_report(tmp_path):
    depth = _write_run(tmp_path / "depth", "depth")
    output = tmp_path / "comparison"

    exit_code = main(
        [
            "--depth-run",
            str(depth),
            "--depth-gate-only",
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = _read_json(output / "depth_gate.json")
    assert payload["passed"] is True
    assert payload["sequence_count"] == 9
    assert not list(output.glob("*.tmp"))
