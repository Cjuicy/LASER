from __future__ import annotations

import subprocess
import sys
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from experiments.config import (
    CapabilityExperimentConfig,
    EvaluationInputConfig,
    EvaluationKind,
)
from pipeline.config import load_pipeline_config
from pipeline.artifacts import (
    ReconstructionArtifact,
    ReconstructionDiagnostics,
    write_reconstruction_artifact,
)
from tools.keyframe_ate_low_disk import (
    _copy_failure_metadata,
    _safe_remove_directory,
    run_low_disk_ate,
)


def _experiment(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(3):
        (image_dir / f"frame-{index}.png").write_bytes(b"image")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    reconstruction_path = tmp_path / "reconstruction.yaml"
    reconstruction = OmegaConf.load("configs/reconstruction/pi3_laser.yaml")
    reconstruction.input.image_dir = str(image_dir)
    reconstruction.input.sample_stride = 1
    reconstruction.model.checkpoint = str(checkpoint)
    reconstruction.model.inference_device = "cpu"
    reconstruction.model.process_device = "cpu"
    reconstruction.model.dtype = "float32"
    reconstruction.window.size = 3
    reconstruction.window.overlap = 1
    reconstruction.loop.optimizer.implementation = "python"
    OmegaConf.save(reconstruction, reconstruction_path)

    evaluation_config = tmp_path / "ate.yaml"
    evaluation_config.write_text(
        "version: 1\nalignment: sim3\nrpe_delta_frames: 1\n",
        encoding="utf-8",
    )
    ground_truth = tmp_path / "poses.txt"
    ground_truth.write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n" * 3,
        encoding="utf-8",
    )
    return CapabilityExperimentConfig(
        version=2,
        reconstruction_config=str(reconstruction_path),
        output_root=str(tmp_path / "unused-matrix-output"),
        dataset_name="KITTI",
        sequence="00",
        evaluator_inputs={
            EvaluationKind.ATE: EvaluationInputConfig(
                EvaluationKind.ATE,
                str(evaluation_config),
                str(ground_truth),
                "replica",
            )
        },
        segmentation_methods=("depth", "geometry", "atomic"),
        refinement_states=(False, True),
    )


def _fake_runtime(calls):
    class FakeRunner:
        def __init__(self, loaded, *, artifact_output_dir):
            self.loaded = loaded
            self.target = Path(artifact_output_dir)

        def run(self):
            config = self.loaded.config
            calls.append(
                (
                    config.segmentation.method.value,
                    config.reconstruction.mode.value,
                    config.segmentation.window_reference.enabled,
                    config.prediction_cache.root,
                )
            )
            self.target.mkdir(parents=True)
            (self.target / "manifest.json").write_text(
                '{"schema_version": 1}\n',
                encoding="utf-8",
            )

    def evaluator(artifact_dir, **values):
        manifest = Path(artifact_dir) / "manifest.json"
        assert manifest.is_file()
        output = Path(values["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        result = output / "trajectory_metrics.json"
        result.write_text(
            json.dumps(
                {
                    "ate_rmse_m": 1.0,
                    "rpe_translation_rmse_m": 0.1,
                    "rpe_rotation_rmse_deg": 0.2,
                    "matched_frame_count": 3,
                    "artifact_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    return FakeRunner, evaluator


def _write_real_artifact(loaded, target):
    poses = torch.eye(4).repeat(3, 1, 1)
    points = torch.zeros((3, 1, 1, 3))
    confidence = torch.ones((3, 1, 1))
    config = loaded.config
    return write_reconstruction_artifact(
        ReconstructionArtifact(
            schema_version=1,
            frame_ids=(0, 1, 2),
            local_points=points,
            global_points=points,
            camera_poses=poses,
            confidence=confidence,
            segmentation_method=config.segmentation.method,
            reconstruction_mode=config.reconstruction.mode,
            prediction_key="prediction-key",
            diagnostics=ReconstructionDiagnostics({}, (), 0, 0, {}),
        ),
        target,
        resolved_yaml=loaded.resolved_yaml,
        config_sha256=loaded.sha256,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )


def test_low_disk_keyframe_ate_cli_exposes_required_inputs():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/keyframe_ate_low_disk.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--work-root" in completed.stdout


def test_low_disk_runner_executes_canonical_variants_and_removes_artifacts(
    tmp_path,
):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    work_root = tmp_path / "low-disk"

    results = run_low_disk_ate(
        _experiment(tmp_path),
        work_root=work_root,
        runner_factory=runner,
        evaluator=evaluator,
        source_digest_provider=lambda: "a" * 64,
    )

    expected = [
        (segmentation, mode, refinement)
        for segmentation in ("depth", "geometry", "atomic")
        for mode, refinement in (
            ("no_loop", False),
            ("no_loop", True),
            ("traditional", False),
            ("corrected", False),
            ("corrected", True),
        )
    ]
    assert [call[:3] for call in calls] == expected
    assert len({call[3] for call in calls}) == 1
    assert all(result.status == "passed" for result in results)
    assert all(not result.artifact_dir.exists() for result in results)
    assert len(tuple((work_root / "metrics").glob("*/trajectory_metrics.json"))) == 15


def test_low_disk_runner_hands_real_artifacts_to_real_ate_evaluator(tmp_path):
    class ArtifactRunner:
        def __init__(self, loaded, *, artifact_output_dir):
            self.loaded = loaded
            self.target = Path(artifact_output_dir)

        def run(self):
            return _write_real_artifact(self.loaded, self.target)

    results = run_low_disk_ate(
        _experiment(tmp_path),
        work_root=tmp_path / "low-disk",
        runner_factory=ArtifactRunner,
        source_digest_provider=lambda: "a" * 64,
    )

    assert all(result.status == "passed" for result in results)
    assert all(
        json.loads(result.metrics_path.read_text())["matched_frame_count"] == 3
        for result in results
    )
    assert all(not result.artifact_dir.exists() for result in results)


def test_low_disk_runner_resumes_metrics_without_reconstruction(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    experiment = _experiment(tmp_path)
    work_root = tmp_path / "low-disk"
    values = {
        "work_root": work_root,
        "runner_factory": runner,
        "evaluator": evaluator,
        "source_digest_provider": lambda: "a" * 64,
    }

    run_low_disk_ate(experiment, **values)
    resumed = run_low_disk_ate(experiment, **values)

    assert len(calls) == 15
    assert all(result.status == "passed" for result in resumed)
    assert all(result.resumed for result in resumed)


def test_low_disk_runner_recomputes_tampered_metrics(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    experiment = _experiment(tmp_path)
    work_root = tmp_path / "low-disk"
    values = {
        "work_root": work_root,
        "runner_factory": runner,
        "evaluator": evaluator,
        "source_digest_provider": lambda: "a" * 64,
    }
    first = run_low_disk_ate(experiment, **values)
    first[0].metrics_path.write_text("{}\n", encoding="utf-8")

    resumed = run_low_disk_ate(experiment, **values)

    assert len(calls) == 16
    assert not resumed[0].resumed
    assert all(result.resumed for result in resumed[1:])


def test_low_disk_runner_keeps_failed_artifact_and_continues(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    should_fail = True

    def flaky_evaluator(*args, **kwargs):
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("synthetic evaluator failure")
        return evaluator(*args, **kwargs)

    experiment = _experiment(tmp_path)
    work_root = tmp_path / "low-disk"
    values = {
        "work_root": work_root,
        "runner_factory": runner,
        "evaluator": flaky_evaluator,
        "source_digest_provider": lambda: "a" * 64,
        "artifact_validator": lambda path: True,
    }

    first = run_low_disk_ate(experiment, **values)

    assert first[0].status == "failed"
    assert first[0].artifact_dir.is_dir()
    assert all(result.status == "passed" for result in first[1:])
    assert all(not result.artifact_dir.exists() for result in first[1:])

    second = run_low_disk_ate(experiment, **values)

    assert len(calls) == 15
    assert all(result.status == "passed" for result in second)
    assert sum(result.resumed for result in second) == 14
    assert all(not result.artifact_dir.exists() for result in second)


def test_changed_identity_does_not_delete_retained_failed_artifact(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    should_fail = True

    def flaky_evaluator(*args, **kwargs):
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("synthetic evaluator failure")
        return evaluator(*args, **kwargs)

    experiment = _experiment(tmp_path)
    work_root = tmp_path / "low-disk"
    first = run_low_disk_ate(
        experiment,
        work_root=work_root,
        runner_factory=runner,
        evaluator=flaky_evaluator,
        source_digest_provider=lambda: "a" * 64,
    )
    retained = first[0].artifact_dir

    second = run_low_disk_ate(
        experiment,
        work_root=work_root,
        runner_factory=runner,
        evaluator=evaluator,
        source_digest_provider=lambda: "b" * 64,
    )

    assert retained.is_dir()
    assert all(result.status == "passed" for result in second)
    assert len(calls) == 30


def test_per_entry_preparation_failure_is_recorded_and_matrix_continues(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    load_count = 0

    def flaky_config_loader(path, overrides):
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            raise ValueError("synthetic config failure")
        return load_pipeline_config(path, overrides)

    work_root = tmp_path / "low-disk"
    results = run_low_disk_ate(
        _experiment(tmp_path),
        work_root=work_root,
        runner_factory=runner,
        evaluator=evaluator,
        source_digest_provider=lambda: "a" * 64,
        config_loader=flaky_config_loader,
    )

    assert results[0].status == "failed"
    assert results[0].reconstruction_identity is None
    assert all(result.status == "passed" for result in results[1:])
    assert len(calls) == 14
    summary = json.loads((work_root / "summary.json").read_text())
    assert summary["failed"] == 1
    assert summary["passed"] == 14


def test_failed_artifact_retention_is_bounded(tmp_path):
    calls = []
    runner, _ = _fake_runtime(calls)

    def failing_evaluator(*args, **kwargs):
        raise RuntimeError("synthetic evaluator failure")

    work_root = tmp_path / "low-disk"
    results = run_low_disk_ate(
        _experiment(tmp_path),
        work_root=work_root,
        runner_factory=runner,
        evaluator=failing_evaluator,
        source_digest_provider=lambda: "a" * 64,
    )

    assert all(result.status == "failed" for result in results)
    assert sum(result.artifact_dir.is_dir() for result in results) == 1
    attempt_records = [
        json.loads(path.read_text())
        for path in (work_root / "scratch").rglob("attempt_record.json")
    ]
    assert sum(record["status"] == "failed" for record in attempt_records) == 1
    assert (
        sum(record["status"] == "artifact_evicted" for record in attempt_records) == 14
    )


def test_lower_retention_limit_is_enforced_without_a_new_failure(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    remaining_failures = 3

    def fail_three(*args, **kwargs):
        nonlocal remaining_failures
        if remaining_failures:
            remaining_failures -= 1
            raise RuntimeError("synthetic evaluator failure")
        return evaluator(*args, **kwargs)

    experiment = _experiment(tmp_path)
    work_root = tmp_path / "low-disk"
    first = run_low_disk_ate(
        experiment,
        work_root=work_root,
        runner_factory=runner,
        evaluator=fail_three,
        source_digest_provider=lambda: "a" * 64,
        max_retained_failures=3,
    )
    assert sum(result.artifact_dir.is_dir() for result in first) == 3

    run_low_disk_ate(
        experiment,
        work_root=work_root,
        runner_factory=runner,
        evaluator=evaluator,
        source_digest_provider=lambda: "b" * 64,
        max_retained_failures=1,
    )

    retained = tuple((work_root / "scratch").glob("*/*/attempt-*/artifact"))
    assert sum(path.is_dir() for path in retained) <= 1


def test_corrupted_record_cannot_reuse_another_entry_artifact(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    remaining_failures = 2

    def fail_two(*args, **kwargs):
        nonlocal remaining_failures
        if remaining_failures:
            remaining_failures -= 1
            raise RuntimeError("synthetic evaluator failure")
        return evaluator(*args, **kwargs)

    experiment = _experiment(tmp_path)
    work_root = tmp_path / "low-disk"
    first = run_low_disk_ate(
        experiment,
        work_root=work_root,
        runner_factory=runner,
        evaluator=fail_two,
        source_digest_provider=lambda: "a" * 64,
        max_retained_failures=2,
        artifact_validator=lambda path: True,
    )
    first_record = first[0].metrics_path.parent / "low_disk_record.json"
    payload = json.loads(first_record.read_text())
    payload["artifact_dir"] = str(first[1].artifact_dir)
    first_record.write_text(json.dumps(payload), encoding="utf-8")

    second = run_low_disk_ate(
        experiment,
        work_root=work_root,
        runner_factory=runner,
        evaluator=evaluator,
        source_digest_provider=lambda: "a" * 64,
        max_retained_failures=2,
        artifact_validator=lambda path: True,
    )

    assert len(calls) == 15
    assert all(result.status == "passed" for result in second)


def test_interrupted_complete_attempt_is_reused(tmp_path):
    calls = []
    interrupt = True

    class InterruptingRunner:
        def __init__(self, loaded, *, artifact_output_dir):
            self.loaded = loaded
            self.target = Path(artifact_output_dir)

        def run(self):
            nonlocal interrupt
            calls.append(self.target)
            result = _write_real_artifact(self.loaded, self.target)
            if interrupt:
                interrupt = False
                raise KeyboardInterrupt
            return result

    experiment = _experiment(tmp_path)
    work_root = tmp_path / "low-disk"
    values = {
        "work_root": work_root,
        "runner_factory": InterruptingRunner,
        "source_digest_provider": lambda: "a" * 64,
    }

    with pytest.raises(KeyboardInterrupt):
        run_low_disk_ate(experiment, **values)
    resumed = run_low_disk_ate(experiment, **values)

    assert len(calls) == 15
    assert all(result.status == "passed" for result in resumed)


def test_failure_metadata_copy_rejects_symlink_destination(tmp_path):
    scratch_root = tmp_path / "scratch"
    artifact = scratch_root / "entry" / ("a" * 64) / "attempt-0001" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text("source", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = artifact.parent / "evicted_artifact_metadata"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        _copy_failure_metadata(artifact, scratch_root=scratch_root)

    assert not (outside / "manifest.json").exists()


def test_quota_uses_guarded_fallback_before_evicting_failed_artifact(tmp_path):
    calls = []
    runner, evaluator = _fake_runtime(calls)
    failed_artifacts = []
    outside = tmp_path / "outside"
    outside.mkdir()

    def fail_two(artifact_dir, **kwargs):
        current = Path(artifact_dir)
        if len(failed_artifacts) == 0:
            failed_artifacts.append(current)
            raise RuntimeError("first synthetic evaluator failure")
        if len(failed_artifacts) == 1:
            failed_artifacts[0].parent.joinpath("evicted_artifact_metadata").symlink_to(
                outside, target_is_directory=True
            )
            failed_artifacts.append(current)
            raise RuntimeError("second synthetic evaluator failure")
        return evaluator(artifact_dir, **kwargs)

    work_root = tmp_path / "low-disk"
    run_low_disk_ate(
        _experiment(tmp_path),
        work_root=work_root,
        runner_factory=runner,
        evaluator=fail_two,
        source_digest_provider=lambda: "a" * 64,
        max_retained_failures=1,
    )

    assert not failed_artifacts[0].exists()
    assert failed_artifacts[1].is_dir()
    assert not (outside / "manifest.json").exists()
    fallback_manifests = tuple(
        (work_root / "scratch" / "evicted-failure-metadata").glob("*/manifest.json")
    )
    assert len(fallback_manifests) == 1
    assert fallback_manifests[0].read_text() == '{"schema_version": 1}\n'


def test_safe_cleanup_rejects_path_outside_scratch_root(tmp_path):
    scratch_root = tmp_path / "scratch"
    outside = tmp_path / "outside"
    scratch_root.mkdir()
    outside.mkdir()

    with pytest.raises(ValueError, match="outside allowed root"):
        _safe_remove_directory(outside, allowed_root=scratch_root)

    assert outside.is_dir()


def test_safe_cleanup_rejects_root_symlink_and_file(tmp_path):
    scratch_root = tmp_path / "scratch"
    directory = scratch_root / "directory"
    directory.mkdir(parents=True)
    symlink = scratch_root / "link"
    symlink.symlink_to(directory, target_is_directory=True)
    intermediate_symlink = scratch_root / "intermediate"
    intermediate_symlink.symlink_to(directory, target_is_directory=True)
    nested_via_symlink = intermediate_symlink / "nested"
    nested_via_symlink.mkdir()
    regular_file = scratch_root / "file"
    regular_file.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="root itself"):
        _safe_remove_directory(scratch_root, allowed_root=scratch_root)
    with pytest.raises(ValueError, match="symbolic link"):
        _safe_remove_directory(symlink, allowed_root=scratch_root)
    with pytest.raises(ValueError, match="symbolic link"):
        _safe_remove_directory(nested_via_symlink, allowed_root=scratch_root)
    with pytest.raises(ValueError, match="not a directory"):
        _safe_remove_directory(regular_file, allowed_root=scratch_root)

    assert scratch_root.is_dir()
    assert symlink.is_symlink()
    assert nested_via_symlink.is_dir()
    assert regular_file.is_file()
