from __future__ import annotations

import importlib.util
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "disposable_experiments"
    / "preflight_experiments.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("disposable_preflight", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pointcloud_identity(**overrides):
    identity = {
        "evaluation": "pointcloud",
        "dataset": "nrgbd",
        "scene": "breakfast_room",
        "method": "atomic-original",
        "segmentation_method": "atomic",
        "split_mode": "none",
        "reconstruction_mode": "no_loop",
        "sample_stride": 1,
        "window_size": 20,
        "overlap": 5,
        "registration_confidence_keep_ratio": 0.5,
    }
    identity.update(overrides)
    return identity


def pointcloud_result(identity=None, accuracy_mean_m=0.01):
    return {
        "schema_version": 1,
        "identity": identity or pointcloud_identity(),
        "prediction_key": "a" * 64,
        "frame_count": 4,
        "metrics": {
            "accuracy_mean_m": accuracy_mean_m,
            "accuracy_median_m": 0.009,
            "completion_mean_m": 0.012,
            "completion_median_m": 0.011,
            "normal_consistency_mean": 0.8,
            "normal_consistency_median": 0.9,
            "chamfer_l1_m": 0.011,
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


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_requested_matrices_resolve_to_exact_runtime_overrides():
    module = load_module()

    assert list(module.POINTCLOUD_METHODS) == [
        "depth",
        "geometry",
        "atomic-original",
        "atomic-split-assisted",
        "atomic-split-no-assisted",
    ]
    assert list(module.ATE_METHODS) == [
        "depth-traditional",
        "depth-corrected",
        "geometry-corrected",
        "atomic-original-corrected",
        "atomic-split-assisted-corrected",
        "atomic-split-no-assisted-corrected",
    ]
    assert module.KITTI_SEQUENCES == tuple(f"{index:02d}" for index in range(11))
    assert module.build_method_overrides("atomic-original", "pointcloud") == (
        "segmentation.method=atomic",
        "segmentation.atomic.split_mode=none",
        "reconstruction.mode=no_loop",
    )
    assert module.build_method_overrides(
        "atomic-split-assisted-corrected", "ate"
    ) == (
        "segmentation.method=atomic",
        "segmentation.atomic.split_mode=conservative",
        "reconstruction.mode=corrected",
    )
    assert module.build_method_overrides(
        "atomic-split-no-assisted-corrected", "ate"
    ) == (
        "segmentation.method=atomic",
        "segmentation.atomic.split_mode=normal_only",
        "reconstruction.mode=corrected",
    )


def test_direct_script_load_adds_repository_root_to_import_path(tmp_path):
    environment = {**os.environ, "PYTHONPATH": ""}
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"runpy.run_path({str(MODULE_PATH)!r}, run_name='probe'); "
                "import evaluation, pipeline; "
                "print(evaluation.__file__); print(pipeline.__file__)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    assert str(REPOSITORY_ROOT) in probe.stdout


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_result_validation_rejects_non_finite_metrics(tmp_path, bad):
    module = load_module()
    path = write_json(tmp_path / "result.json", pointcloud_result(accuracy_mean_m=bad))

    with pytest.raises(ValueError, match="finite"):
        module.load_valid_result(path, pointcloud_identity())


def test_result_validation_rejects_wrong_split_identity(tmp_path):
    module = load_module()
    wrong_identity = pointcloud_identity(split_mode="conservative")
    path = write_json(tmp_path / "result.json", pointcloud_result(wrong_identity))

    with pytest.raises(ValueError, match="identity"):
        module.load_valid_result(path, pointcloud_identity())


def test_result_validation_accepts_complete_finite_result(tmp_path):
    module = load_module()
    path = write_json(tmp_path / "result.json", pointcloud_result())

    loaded = module.load_valid_result(path, pointcloud_identity())

    assert loaded["prediction_key"] == "a" * 64
    assert loaded["metrics"]["chamfer_l1_m"] == pytest.approx(0.011)


def test_guarded_remove_rejects_target_outside_owned_root(tmp_path):
    module = load_module()
    raw = tmp_path / "data" / "KITTI" / "00"
    raw.mkdir(parents=True)

    with pytest.raises(ValueError, match="outside allowed root"):
        module.guarded_remove(raw, tmp_path / "outputs" / "work")

    assert raw.is_dir()


def test_guarded_remove_deletes_only_descendant(tmp_path):
    module = load_module()
    owned = tmp_path / "outputs" / "work"
    target = owned / "kitti" / "00" / "depth-corrected"
    target.mkdir(parents=True)
    (target / "artifact.pt").write_bytes(b"artifact")

    module.guarded_remove(target, owned)

    assert not target.exists()
    assert owned.is_dir()


def test_prediction_cleanup_requires_explicit_sha256_keys(tmp_path):
    module = load_module()
    cache = tmp_path / "predictions"
    wanted = "1" * 64
    other = "2" * 64
    (cache / "v2" / wanted).mkdir(parents=True)
    (cache / "v2" / other).mkdir(parents=True)

    module.remove_prediction_keys(cache, (wanted,))

    assert not (cache / "v2" / wanted).exists()
    assert (cache / "v2" / other).is_dir()
    with pytest.raises(ValueError, match="SHA256"):
        module.remove_prediction_keys(cache, ("../data",))


def test_plan_counts_all_requested_runs(tmp_path, capsys):
    module = load_module()
    seven_map = {f"seven-{index}": [0] for index in range(18)}
    nrgbd_map = {f"nrgbd-{index}": [0] for index in range(9)}
    seven_path = write_json(tmp_path / "seven.json", seven_map)
    nrgbd_path = write_json(tmp_path / "nrgbd.json", nrgbd_map)

    exit_code = module.main(
        [
            "plan",
            "--seven-map",
            str(seven_path),
            "--nrgbd-map",
            str(nrgbd_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "ate_method_runs": 66,
        "kitti_sequences": 11,
        "pointcloud_method_runs": 135,
        "pointcloud_scenes": 27,
        "total_method_runs": 201,
    }


def test_pinned_sequence_map_rejects_same_size_different_content(tmp_path):
    module = load_module()
    expected = write_json(tmp_path / "expected.json", {"scene-a": [0, 10]})
    changed = write_json(tmp_path / "changed.json", {"scene-b": [0, 10]})
    expected_sha256 = hashlib.sha256(expected.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="checksum"):
        module.validate_pinned_sequence_map(changed, expected_sha256, "fixture")


def test_compact_ate_rejects_evaluator_from_another_artifact(tmp_path):
    module = load_module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_json(
        artifact / "manifest.json",
        {
            "schema_version": 1,
            "segmentation_method": "depth",
            "reconstruction_mode": "corrected",
            "prediction_key": "a" * 64,
        },
    )
    (artifact / "resolved_reconstruction.yaml").write_text(
        """
segmentation:
  atomic:
    split_mode: none
input:
  sample_stride: 1
window:
  size: 75
  overlap: 30
registration:
  confidence_keep_ratio: 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    evaluator = write_json(
        tmp_path / "trajectory_metrics.json",
        {
            "ate_rmse_m": 1.0,
            "rpe_translation_rmse_m": 0.1,
            "rpe_rotation_rmse_deg": 0.2,
            "matched_frame_count": 20,
            "artifact_manifest_sha256": "b" * 64,
        },
    )
    identity = module.method_identity(
        evaluation="ate",
        dataset="kitti",
        scene="00",
        method="depth-corrected",
    )

    with pytest.raises(ValueError, match="manifest"):
        module.compact_ate(
            artifact=artifact,
            evaluator_result=evaluator,
            output=tmp_path / "compact.json",
            identity=identity,
        )

    assert not (tmp_path / "compact.json").exists()


def test_artifact_identity_accepts_serialized_enum_name(tmp_path):
    module = load_module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_json(
        artifact / "manifest.json",
        {
            "schema_version": 1,
            "segmentation_method": "depth",
            "reconstruction_mode": "no_loop",
            "prediction_key": "a" * 64,
        },
    )
    (artifact / "resolved_reconstruction.yaml").write_text(
        """
segmentation:
  atomic:
    split_mode: NONE
input:
  sample_stride: 1
window:
  size: 20
  overlap: 5
registration:
  confidence_keep_ratio: 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    identity = module.method_identity(
        evaluation="pointcloud",
        dataset="7scenes",
        scene="chess/seq-03",
        method="depth",
    )

    manifest, _ = module._validate_artifact_identity(artifact, identity)

    assert manifest["prediction_key"] == "a" * 64


def test_nrgbd_preparation_reads_only_requested_scene(tmp_path):
    module = load_module()
    from PIL import Image
    import numpy as np

    root = tmp_path / "NeuralRGBD"
    selected = root / "selected"
    (selected / "images").mkdir(parents=True)
    (selected / "depth").mkdir()
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(
        selected / "images" / "img0.png"
    )
    Image.fromarray(np.full((480, 640), 1000, dtype=np.uint16)).save(
        selected / "depth" / "depth0.png"
    )
    np.savetxt(selected / "poses.txt", np.eye(4))
    (root / "unrelated-malformed-scene").mkdir()
    sequence_map = write_json(tmp_path / "map.json", {"selected": [0]})

    output = module.prepare_pointcloud(
        dataset="nrgbd",
        root=root,
        sequence_map=sequence_map,
        scene="selected",
        output_root=tmp_path / "prepared",
    )

    assert (output / "ground_truth.npz").is_file()
    assert (output / "images" / "000000.png").is_symlink()
