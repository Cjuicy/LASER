from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY_ROOT / "tools" / "disposable_experiments"


def run_script(script: str, tmp_path: Path, extra_environment=None):
    environment = {
        **os.environ,
        "DRY_RUN": "1",
        "EXPERIMENT_ROOT": str(tmp_path / "experiment"),
        "LASER_PYTHON": "python",
        **(extra_environment or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT_ROOT / script)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def reconstruction_commands(output: str):
    return [
        line
        for line in output.splitlines()
        if line.startswith("DRY-RUN ")
        and (
            "run_reconstruction.py" in line
            or "run-reconstruction-loop-safe" in line
        )
    ]


def assert_method_overrides_are_set_options(commands):
    for command in commands:
        tokens = shlex.split(command.removeprefix("DRY-RUN "))
        override_tokens = [
            token
            for token in tokens
            if token.startswith(
                (
                    "segmentation.method=",
                    "segmentation.atomic.split_mode=",
                    "reconstruction.mode=",
                )
            )
        ]
        assert len(override_tokens) == 3
        for override in override_tokens:
            assert tokens[tokens.index(override) - 1] == "--set"


def test_pointcloud_dry_run_emits_five_exact_methods_per_scene(tmp_path):
    seven_map = tmp_path / "seven.json"
    nrgbd_map = tmp_path / "nrgbd.json"
    seven_map.write_text(json.dumps({"chess/seq-03": [0, 10]}), encoding="utf-8")
    nrgbd_map.write_text(json.dumps({"breakfast_room": [0, 10]}), encoding="utf-8")

    result = run_script(
        "run_pointcloud_all.sh",
        tmp_path,
        {
            "SEVEN_SCENES_MAP": str(seven_map),
            "NRGBD_MAP": str(nrgbd_map),
        },
    )

    assert result.returncode == 0, result.stderr
    commands = reconstruction_commands(result.stdout)
    assert len(commands) == 10
    assert_method_overrides_are_set_options(commands)
    assert all("window.size=20" in command for command in commands)
    assert all("window.overlap=5" in command for command in commands)
    assert all("registration.confidence_keep_ratio=0.5" in command for command in commands)
    per_scene = commands[:5]
    assert sum("segmentation.method=depth" in command for command in per_scene) == 1
    assert sum("segmentation.method=geometry" in command for command in per_scene) == 1
    assert sum("segmentation.method=atomic" in command for command in per_scene) == 3
    assert sum("segmentation.atomic.split_mode=none" in command for command in per_scene) == 3
    assert sum(
        "segmentation.atomic.split_mode=conservative" in command
        for command in per_scene
    ) == 1
    assert sum(
        "segmentation.atomic.split_mode=normal_only" in command
        for command in per_scene
    ) == 1
    assert all("reconstruction.mode=no_loop" in command for command in commands)


def test_pointcloud_dry_run_orders_compaction_before_artifact_cleanup(tmp_path):
    seven_map = tmp_path / "seven.json"
    nrgbd_map = tmp_path / "nrgbd.json"
    seven_map.write_text(json.dumps({"chess/seq-03": [0, 10]}), encoding="utf-8")
    nrgbd_map.write_text(json.dumps({}), encoding="utf-8")

    result = run_script(
        "run_pointcloud_all.sh",
        tmp_path,
        {
            "SEVEN_SCENES_MAP": str(seven_map),
            "NRGBD_MAP": str(nrgbd_map),
        },
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    compact = next(index for index, line in enumerate(lines) if "compact-pointcloud" in line)
    cleanup = next(index for index, line in enumerate(lines) if "cleanup-artifact" in line)
    assert compact < cleanup


def test_kitti_dry_run_emits_one_traditional_and_five_corrected_methods(tmp_path):
    result = run_script(
        "run_kitti_ate_all.sh",
        tmp_path,
        {"DRY_RUN_KITTI_SEQUENCE_LIST": "00 01"},
    )

    assert result.returncode == 0, result.stderr
    commands = reconstruction_commands(result.stdout)
    assert len(commands) == 12
    assert all("run-reconstruction-loop-safe" in command for command in commands)
    assert_method_overrides_are_set_options(commands)
    assert all("window.size=75" in command for command in commands)
    assert all("window.overlap=30" in command for command in commands)
    assert all("registration.confidence_keep_ratio=0.5" in command for command in commands)
    per_sequence = commands[:6]
    assert sum("reconstruction.mode=traditional" in command for command in per_sequence) == 1
    assert sum("reconstruction.mode=corrected" in command for command in per_sequence) == 5
    assert "segmentation.method=depth" in next(
        command for command in per_sequence if "reconstruction.mode=traditional" in command
    )
    assert sum("segmentation.method=atomic" in command for command in per_sequence) == 3
    assert sum(
        "segmentation.atomic.split_mode=conservative" in command
        for command in per_sequence
    ) == 1
    assert sum(
        "segmentation.atomic.split_mode=normal_only" in command
        for command in per_sequence
    ) == 1


def test_kitti_dry_run_evaluates_and_compacts_before_artifact_cleanup(tmp_path):
    result = run_script(
        "run_kitti_ate_all.sh",
        tmp_path,
        {"DRY_RUN_KITTI_SEQUENCE_LIST": "00"},
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    evaluate = next(index for index, line in enumerate(lines) if "evaluate_ate.py" in line)
    compact = next(index for index, line in enumerate(lines) if "compact-ate" in line)
    cleanup = next(index for index, line in enumerate(lines) if "cleanup-artifact" in line)
    assert evaluate < compact < cleanup


def test_run_all_dry_run_orders_preflight_campaign_and_summary(tmp_path):
    seven_map = tmp_path / "seven.json"
    nrgbd_map = tmp_path / "nrgbd.json"
    seven_map.write_text(json.dumps({"chess/seq-03": [0, 10]}), encoding="utf-8")
    nrgbd_map.write_text(
        json.dumps({"breakfast_room": [0, 10]}), encoding="utf-8"
    )

    result = run_script(
        "run_all.sh",
        tmp_path,
        {
            "SEVEN_SCENES_MAP": str(seven_map),
            "NRGBD_MAP": str(nrgbd_map),
            "DRY_RUN_KITTI_SEQUENCE_LIST": "00",
        },
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    phases = [line for line in lines if line.startswith("[all] phase=")]
    assert phases == [
        "[all] phase=preflight",
        "[all] phase=pointcloud",
        "[all] phase=ate",
        "[all] phase=summary",
        "[all] phase=complete",
    ]
    preflight = next(index for index, line in enumerate(lines) if " preflight " in line)
    reconstruction = next(
        index for index, line in enumerate(lines) if "run_reconstruction.py" in line
    )
    summary = next(
        index for index, line in enumerate(lines) if "summarize_results.py" in line
    )
    assert preflight < reconstruction < summary


def test_kitti_subset_hook_is_rejected_outside_dry_run(tmp_path):
    environment = {
        **os.environ,
        "DRY_RUN": "0",
        "DRY_RUN_KITTI_SEQUENCE_LIST": "00",
        "EXPERIMENT_ROOT": str(tmp_path / "experiment"),
        "LASER_PYTHON": "python",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT_ROOT / "run_kitti_ate_all.sh")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "DRY_RUN" in result.stderr
    assert "run_reconstruction.py" not in result.stdout


def test_completed_pointcloud_scene_is_not_prepared_again(tmp_path):
    seven_map = tmp_path / "seven.json"
    nrgbd_map = tmp_path / "nrgbd.json"
    seven_map.write_text(json.dumps({"chess/seq-03": [0, 10]}), encoding="utf-8")
    nrgbd_map.write_text(json.dumps({}), encoding="utf-8")
    metric_dir = (
        tmp_path
        / "experiment"
        / "metrics"
        / "pointcloud"
        / "7scenes"
        / "chess__seq-03"
    )
    metric_dir.mkdir(parents=True)
    method_table = {
        "depth": ("depth", "none"),
        "geometry": ("geometry", "none"),
        "atomic-original": ("atomic", "none"),
        "atomic-split-assisted": ("atomic", "conservative"),
        "atomic-split-no-assisted": ("atomic", "normal_only"),
    }
    for method, (segmentation, split_mode) in method_table.items():
        payload = {
            "schema_version": 1,
            "identity": {
                "evaluation": "pointcloud",
                "dataset": "7scenes",
                "scene": "chess/seq-03",
                "method": method,
                "segmentation_method": segmentation,
                "split_mode": split_mode,
                "reconstruction_mode": "no_loop",
                "sample_stride": 1,
                "window_size": 20,
                "overlap": 5,
                "registration_confidence_keep_ratio": 0.5,
            },
            "prediction_key": "a" * 64,
            "frame_count": 2,
            "metrics": {
                "accuracy_mean_m": 0.01,
                "accuracy_median_m": 0.01,
                "completion_mean_m": 0.01,
                "completion_median_m": 0.01,
                "normal_consistency_mean": 0.8,
                "normal_consistency_median": 0.8,
                "chamfer_l1_m": 0.01,
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
        (metric_dir / f"{method}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    environment = {
        **os.environ,
        "DRY_RUN": "0",
        "EXPERIMENT_ROOT": str(tmp_path / "experiment"),
        "LASER_PYTHON": "python",
        "SEVEN_SCENES_MAP": str(seven_map),
        "NRGBD_MAP": str(nrgbd_map),
        "SEVEN_SCENES_ROOT": str(tmp_path / "raw-does-not-need-to-exist"),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT_ROOT / "run_pointcloud_all.sh")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[skip-scene]" in result.stdout
    assert "prepare-pointcloud" not in result.stdout
