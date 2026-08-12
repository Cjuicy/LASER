from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from mv_recon.compare_loop_results import (
    compare_loop_run_directories,
    main,
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
PRIMARY = {
    "accuracy_mean_m": 0.0201,
    "accuracy_median_m": 0.0101,
    "completion_mean_m": 0.0121,
    "completion_median_m": 0.0041,
    "normal_consistency_mean": 0.7131,
    "normal_consistency_median": 0.8561,
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


def _pipeline(loop: bool) -> dict[str, object]:
    return {
        "window": {"size": 20, "overlap": 5},
        "prediction_cache": {"mode": "auto"},
        "segmentation": {
            "method": "depth",
            "confidence_keep_ratio": 0.5,
            "depth_merge_threshold": 0.1,
            "temporal_iou_threshold": 0.3,
            "felzenszwalb": {
                "scale": 300,
                "sigma": 1.1,
                "min_size": 500,
            },
        },
        "anchor_propagation": {
            "enabled": True,
            "correspondence_iou_threshold": 0.4,
        },
        "loop": {
            "enabled": loop,
            "method": "traditional",
            "registration": {"confidence_keep_ratio": 0.5},
        },
    }


def _write_run(
    run_dir: Path,
    *,
    loop: bool,
    accuracy_delta: float = 0.0,
) -> Path:
    resolved_protocol = (
        "geometry:\n"
        "  alignment: umeyama_sim3_then_icp\n"
        "  center_crop_size: 224\n"
        "  fscore_thresholds_m:\n"
        "  - 0.01\n"
        "  - 0.02\n"
        "  - 0.05\n"
        "  icp_threshold_m: 0.1\n"
        "  icp_type: point_to_point\n"
        "  normal_estimation: open3d_default\n"
    )
    primary = {**PRIMARY, "accuracy_mean_m": 0.0201 + accuracy_delta}
    sequences = []
    for index, name in enumerate(EXPECTED_NAMES):
        thresholds = []
        for threshold_index, threshold in enumerate((0.01, 0.02, 0.05)):
            precision = 0.50 + index * 0.01 + threshold_index * 0.10
            recall = 0.60 + index * 0.01 + threshold_index * 0.05
            thresholds.append(
                {
                    "threshold_m": threshold,
                    "precision": precision,
                    "recall": recall,
                    "fscore": 2.0 * precision * recall / (precision + recall),
                }
            )
        cache = {
            "ordinary_hits": 1,
            "ordinary_misses": 0,
            "ordinary_prediction_key": _digest(f"ordinary:{name}"),
        }
        if loop:
            cache.update(
                {
                    "candidate_count": 1,
                    "constraint_count": 1,
                    "rejected_candidate_count": 0,
                    "used_no_loop_path": False,
                    "joint_forward_count": 1,
                    "loop_method": "traditional",
                }
            )
        sequences.append(
            {
                "dataset": "NRGBD-dense",
                "sequence": name,
                "frame_count": 10 + index,
                "input_manifest_sha256": _digest(f"input:{name}"),
                "ground_truth_sha256": _digest(f"gt:{name}"),
                "ordinary_prediction_key": _digest(f"ordinary:{name}"),
                "primary": primary,
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
                    "chamfer_l1_m": 0.016 + index * 0.001,
                    "thresholds": thresholds,
                },
                "cache_diagnostics": cache,
            }
        )
    protocol = _digest(f"protocol:{loop}")
    pipeline = _digest(f"pipeline:{loop}")
    auxiliary = {"salad": "a" * 64, "dino": "b" * 64} if loop else {}
    identity = {
        "protocol_identity_sha256": protocol,
        "pipeline_sha256": pipeline,
        "checkpoint_sha256": "c" * 64,
        "sequence_map_sha256": {"NRGBD-dense": "d" * 64},
        "auxiliary_checkpoint_sha256": auxiliary,
        "metric_version": METRIC_VERSION,
    }
    results = {
        "schema_version": METRIC_VERSION,
        "state": "subset",
        "identity": identity,
        "expected_sequences": {"NRGBD-dense": list(EXPECTED_NAMES)},
        "full_sequence_counts": {"NRGBD-dense": 9},
        "subset": True,
        "paper_reference": {"NRGBD-dense": PRIMARY},
        "sequences": sequences,
        "failures": [],
        "datasets": [
            {
                "dataset": "NRGBD-dense",
                "status": "subset",
                "expected_sequences": 9,
                "selected_sequences": 9,
                "completed_sequences": 9,
                "primary": primary,
                "paper_reference": PRIMARY,
                "delta_to_paper": PRIMARY,
            }
        ],
    }
    manifest = {
        "schema_version": 1,
        "metric_schema_version": METRIC_VERSION,
        "git_commit": "loop-commit" if loop else "no-loop-commit",
        "resolved_protocol_sha256": hashlib.sha256(
            resolved_protocol.encode("utf-8")
        ).hexdigest(),
        "protocol_identity_sha256": protocol,
        "resolved_pipeline_sha256": pipeline,
        "evaluation_mode": "experiment" if loop else "comparison",
        "pointmap_assembly": (
            "pipeline-loop-aggregate-v1"
            if loop
            else "laser-incremental-global-map-v1"
        ),
        "segmentation_method": "depth",
        "loop_enabled": loop,
        "loop_method": "traditional",
        "geometry": {
            "center_crop_size": 224,
            "alignment": "umeyama_sim3_then_icp",
            "icp_type": "point_to_point",
            "icp_threshold_m": 0.1,
            "normal_estimation": "open3d_default",
            "fscore_thresholds_m": [0.01, 0.02, 0.05],
        },
        "checkpoint_sha256": "c" * 64,
        "auxiliary_checkpoint_sha256": auxiliary,
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
        "pipeline": _pipeline(loop),
        "selected_sequences": {"NRGBD-dense": list(EXPECTED_NAMES)},
        "selected_sequence_counts": {"NRGBD-dense": 9},
        "expected_sequence_counts": {"NRGBD-dense": 9},
        "attempted_sequences": 9,
        "successful_sequences": 9,
        "failed_sequences": 0,
    }
    _write_json(run_dir / "results.json", results)
    _write_json(run_dir / "protocol_manifest.json", manifest)
    (run_dir / "resolved_protocol.yaml").write_text(
        resolved_protocol,
        encoding="utf-8",
    )
    return run_dir


def _mutate(run: Path, filename: str, mutation) -> None:
    path = run / filename
    payload = _read_json(path)
    mutation(payload)
    _write_json(path, payload)


def test_compare_loop_runs_writes_metrics_deltas_and_warning(tmp_path):
    no_loop = _write_run(tmp_path / "no-loop", loop=False)
    loop = _write_run(
        tmp_path / "loop",
        loop=True,
        accuracy_delta=-0.001,
    )
    output = tmp_path / "comparison"

    report = compare_loop_run_directories(no_loop, loop, output)

    assert report["assemblies"] == {
        "no_loop": "laser-incremental-global-map-v1",
        "traditional_loop": "pipeline-loop-aggregate-v1",
    }
    assert report["interpretation_warning"].startswith(
        "Point-map assembly differs"
    )
    assert report["methods"][1]["delta_to_no_loop"][
        "accuracy_mean_m"
    ] == pytest.approx(-0.001)
    assert report["loop_diagnostics"] == {
        "candidate_count": 9,
        "constraint_count": 9,
        "rejected_candidate_count": 0,
        "joint_forward_count": 9,
        "used_no_loop_path_count": 0,
    }
    assert (output / "loop_comparison.json").is_file()
    assert (output / "loop_comparison.csv").is_file()
    with (output / "loop_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        assert [row["method"] for row in csv.DictReader(stream)] == [
            "no_loop",
            "traditional_loop",
        ]


def test_compare_loop_accepts_hashed_legacy_protocol_geometry(tmp_path):
    no_loop = _write_run(tmp_path / "no-loop", loop=False)
    loop = _write_run(tmp_path / "loop", loop=True)
    _mutate(
        no_loop,
        "protocol_manifest.json",
        lambda value: value.pop("geometry"),
    )
    _mutate(
        loop,
        "protocol_manifest.json",
        lambda value: value.pop("geometry"),
    )

    report = compare_loop_run_directories(
        no_loop,
        loop,
        tmp_path / "comparison",
    )

    assert report["sequence_count"] == 9


def test_compare_loop_rejects_tampered_legacy_protocol_geometry(tmp_path):
    no_loop = _write_run(tmp_path / "no-loop", loop=False)
    loop = _write_run(tmp_path / "loop", loop=True)
    _mutate(
        no_loop,
        "protocol_manifest.json",
        lambda value: value.pop("geometry"),
    )
    (no_loop / "resolved_protocol.yaml").write_text(
        "geometry:\n  center_crop_size: 112\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protocol file hash mismatch"):
        compare_loop_run_directories(
            no_loop,
            loop,
            tmp_path / "comparison",
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("checkpoint", "checkpoint.*mismatch"),
        ("sequence_map", "sequence map.*mismatch"),
        ("ordinary", "ordinary prediction.*mismatch"),
        ("input", "input manifest.*mismatch"),
        ("ground_truth", "ground truth.*mismatch"),
        ("runtime", "runtime.*mismatch"),
        ("coverage", "sequence coverage"),
        ("loop_method", "traditional loop method"),
        ("missing_constraint", "constraint_count"),
        ("bad_totals", "candidate.*constraint.*rejected"),
    ),
)
def test_compare_loop_runs_rejects_noncomparable_inputs(tmp_path, case, message):
    no_loop = _write_run(tmp_path / "no-loop", loop=False)
    loop = _write_run(tmp_path / "loop", loop=True)

    if case == "checkpoint":
        _mutate(
            loop,
            "results.json",
            lambda value: value["identity"].update(checkpoint_sha256="e" * 64),
        )
        _mutate(
            loop,
            "protocol_manifest.json",
            lambda value: value.update(checkpoint_sha256="e" * 64),
        )
    elif case == "sequence_map":
        _mutate(
            loop,
            "results.json",
            lambda value: value["identity"]["sequence_map_sha256"].update(
                {"NRGBD-dense": "e" * 64}
            ),
        )
        _mutate(
            loop,
            "protocol_manifest.json",
            lambda value: value["sequence_map_sha256"].update(
                {"NRGBD-dense": "e" * 64}
            ),
        )
    elif case in {"ordinary", "input", "ground_truth"}:
        field = {
            "ordinary": "ordinary_prediction_key",
            "input": "input_manifest_sha256",
            "ground_truth": "ground_truth_sha256",
        }[case]
        _mutate(
            loop,
            "results.json",
            lambda value: (
                value["sequences"][0].update({field: "e" * 64}),
                value["sequences"][0]["cache_diagnostics"].update(
                    ordinary_prediction_key="e" * 64
                )
                if case == "ordinary"
                else None,
            ),
        )
    elif case == "runtime":
        _mutate(
            loop,
            "protocol_manifest.json",
            lambda value: value["runtime"].update(torch="different"),
        )
    elif case == "coverage":
        _mutate(
            loop,
            "results.json",
            lambda value: value["sequences"].pop(),
        )
    elif case == "loop_method":
        def corrected(value):
            value["loop_method"] = "corrected"
            value["pipeline"]["loop"]["method"] = "corrected"

        _mutate(loop, "protocol_manifest.json", corrected)
    elif case == "missing_constraint":
        _mutate(
            loop,
            "results.json",
            lambda value: value["sequences"][0]["cache_diagnostics"].pop(
                "constraint_count"
            ),
        )
    elif case == "bad_totals":
        _mutate(
            loop,
            "results.json",
            lambda value: value["sequences"][0]["cache_diagnostics"].update(
                rejected_candidate_count=1
            ),
        )

    with pytest.raises(ValueError, match=message):
        compare_loop_run_directories(no_loop, loop, tmp_path / "output")


def test_compare_loop_cli_writes_both_outputs(tmp_path):
    no_loop = _write_run(tmp_path / "no-loop", loop=False)
    loop = _write_run(tmp_path / "loop", loop=True)
    output = tmp_path / "comparison"

    assert main(
        [
            "--no-loop-run",
            str(no_loop),
            "--traditional-loop-run",
            str(loop),
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert (output / "loop_comparison.json").is_file()
    assert (output / "loop_comparison.csv").is_file()
