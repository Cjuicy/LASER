from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import experiments.evaluation_bundle as bundle_module
from experiments.config import (
    CapabilityExperimentConfig,
    EvaluationInputConfig,
    EvaluationKind,
)
from experiments.evaluation_bundle import (
    EvaluatorError,
    EvaluatorRecord,
    EvaluatorStatus,
    read_evaluator_record,
    write_evaluator_record,
)
from experiments.matrix import CapabilityMatrixEntry
from pipeline.config import ReconstructionMode, SegmentationMethod


def _input(kind, config_path, ground_truth_path):
    return EvaluationInputConfig(
        kind=kind,
        config_path=str(config_path),
        ground_truth_path=str(ground_truth_path),
        ground_truth_format="tum" if kind is EvaluationKind.ATE else None,
    )


def _experiment(*inputs):
    return CapabilityExperimentConfig(
        version=2,
        reconstruction_config="configs/reconstruction/pi3_laser.yaml",
        output_root="outputs/keyframe-metrics",
        dataset_name="fixture",
        sequence="sequence-0",
        evaluator_inputs={item.kind: item for item in inputs},
        segmentation_methods=("depth", "geometry", "atomic"),
        refinement_states=(False, True),
    )


def _entry(*kinds, mode=ReconstructionMode.NO_LOOP):
    return CapabilityMatrixEntry(
        segmentation_method=SegmentationMethod.DEPTH,
        reconstruction_mode=mode,
        window_reference_enabled=False,
        evaluator_kinds=tuple(kinds),
    )


def _identity_files(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    files = {
        "ate_config": tmp_path / "ate.yaml",
        "ate_gt": tmp_path / "trajectory.txt",
        "pointcloud_config": tmp_path / "pointcloud.yaml",
        "pointcloud_gt": tmp_path / "pointcloud.npz",
    }
    for name, path in files.items():
        path.write_text(name + "\n", encoding="utf-8")
    return artifact_dir, files


def test_evaluator_records_enforce_terminal_status_contracts():
    bundle = importlib.import_module("experiments.evaluation_bundle")
    error = bundle.EvaluatorError("ValueError", "shape mismatch")

    passed = bundle.EvaluatorRecord(
        kind=EvaluationKind.ATE,
        status=bundle.EvaluatorStatus.PASSED,
        identity_digest="a" * 64,
        output_path="metrics.json",
        error=None,
    )
    assert passed.status is bundle.EvaluatorStatus.PASSED

    invalid_records = (
        {
            "kind": EvaluationKind.ATE,
            "status": bundle.EvaluatorStatus.PASSED,
            "identity_digest": None,
            "output_path": "metrics.json",
            "error": None,
        },
        {
            "kind": EvaluationKind.ATE,
            "status": bundle.EvaluatorStatus.FAILED,
            "identity_digest": None,
            "output_path": None,
            "error": None,
        },
        {
            "kind": EvaluationKind.ATE,
            "status": bundle.EvaluatorStatus.BLOCKED,
            "identity_digest": None,
            "output_path": None,
            "error": None,
        },
        {
            "kind": EvaluationKind.ATE,
            "status": bundle.EvaluatorStatus.SKIPPED,
            "identity_digest": "b" * 64,
            "output_path": None,
            "error": None,
        },
        {
            "kind": EvaluationKind.ATE,
            "status": bundle.EvaluatorStatus.SKIPPED,
            "identity_digest": None,
            "output_path": None,
            "error": error,
        },
    )
    for values in invalid_records:
        with pytest.raises(ValueError):
            bundle.EvaluatorRecord(**values)


def test_evaluator_record_round_trip_rejects_unknown_fields(tmp_path):
    record = EvaluatorRecord(
        kind=EvaluationKind.ATE,
        status=EvaluatorStatus.PASSED,
        identity_digest="a" * 64,
        output_path="metrics.json",
        error=None,
    )
    path = write_evaluator_record(record, tmp_path / "record.json")

    assert read_evaluator_record(path) == record

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields are invalid"):
        read_evaluator_record(path)


def test_dual_evaluators_isolate_failure_and_resume_passed_sibling(tmp_path):
    artifact_dir, files = _identity_files(tmp_path)
    experiment = _experiment(
        _input(EvaluationKind.ATE, files["ate_config"], files["ate_gt"]),
        _input(
            EvaluationKind.POINTCLOUD,
            files["pointcloud_config"],
            files["pointcloud_gt"],
        ),
    )
    calls = {EvaluationKind.ATE: 0, EvaluationKind.POINTCLOUD: 0}
    pointcloud_fails = [True]

    def evaluate_ate(artifact, **values):
        assert Path(artifact) == artifact_dir
        calls[EvaluationKind.ATE] += 1
        output = Path(values["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        result = output / "trajectory_metrics.json"
        result.write_text("{}\n", encoding="utf-8")
        return result

    def evaluate_pointcloud(artifact, **values):
        assert Path(artifact) == artifact_dir
        calls[EvaluationKind.POINTCLOUD] += 1
        if pointcloud_fails[0]:
            raise ValueError("shape mismatch")
        output = Path(values["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "pointcloud_metrics.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        return output

    runner = bundle_module.EvaluationBundleRunner(
        ate_evaluator=evaluate_ate,
        pointcloud_evaluator=evaluate_pointcloud,
    )
    output_root = tmp_path / "evaluation"
    first = runner.run(
        artifact_dir=artifact_dir,
        entry=_entry(EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
        experiment=experiment,
        output_root=output_root,
        source_revision="revision-1",
    )

    assert [record.status for record in first] == [
        EvaluatorStatus.PASSED,
        EvaluatorStatus.FAILED,
    ]
    assert Path(first[0].output_path).is_file()
    assert first[1].error == EvaluatorError("ValueError", "shape mismatch")
    assert calls == {EvaluationKind.ATE: 1, EvaluationKind.POINTCLOUD: 1}

    pointcloud_fails[0] = False
    second = runner.run(
        artifact_dir=artifact_dir,
        entry=_entry(EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
        experiment=experiment,
        output_root=output_root,
        source_revision="revision-1",
    )

    assert [record.status for record in second] == [
        EvaluatorStatus.PASSED,
        EvaluatorStatus.PASSED,
    ]
    assert calls == {EvaluationKind.ATE: 1, EvaluationKind.POINTCLOUD: 2}


def test_declared_missing_input_fails_while_omitted_capability_skips(tmp_path):
    artifact_dir, files = _identity_files(tmp_path)
    missing_ground_truth = tmp_path / "missing-trajectory.txt"
    experiment = _experiment(
        _input(
            EvaluationKind.ATE,
            files["ate_config"],
            missing_ground_truth,
        )
    )
    calls = []
    runner = bundle_module.EvaluationBundleRunner(
        ate_evaluator=lambda *args, **values: calls.append((args, values)),
        pointcloud_evaluator=lambda *args, **values: calls.append((args, values)),
    )

    records = runner.run(
        artifact_dir=artifact_dir,
        entry=_entry(EvaluationKind.ATE),
        experiment=experiment,
        output_root=tmp_path / "evaluation",
        source_revision="revision-1",
    )

    assert [record.status for record in records] == [
        EvaluatorStatus.FAILED,
        EvaluatorStatus.SKIPPED,
    ]
    assert records[0].error.exception_type == "FileNotFoundError"
    assert calls == []


def test_blocked_writes_only_scheduled_evaluators_as_blocked(tmp_path):
    _, files = _identity_files(tmp_path)
    experiment = _experiment(
        _input(EvaluationKind.ATE, files["ate_config"], files["ate_gt"]),
        _input(
            EvaluationKind.POINTCLOUD,
            files["pointcloud_config"],
            files["pointcloud_gt"],
        ),
    )
    runner = bundle_module.EvaluationBundleRunner()

    records = runner.blocked(
        entry=_entry(EvaluationKind.ATE, mode=ReconstructionMode.CORRECTED),
        experiment=experiment,
        output_root=tmp_path / "evaluation",
        error=RuntimeError("reconstruction failed"),
    )

    assert [record.status for record in records] == [
        EvaluatorStatus.BLOCKED,
        EvaluatorStatus.SKIPPED,
    ]
    assert records[0].error == EvaluatorError(
        "RuntimeError",
        "reconstruction failed",
    )


def test_evaluator_identity_changes_invalidate_only_affected_passes(tmp_path):
    artifact_dir, files = _identity_files(tmp_path)
    experiment = _experiment(
        _input(EvaluationKind.ATE, files["ate_config"], files["ate_gt"]),
        _input(
            EvaluationKind.POINTCLOUD,
            files["pointcloud_config"],
            files["pointcloud_gt"],
        ),
    )
    calls = {EvaluationKind.ATE: 0, EvaluationKind.POINTCLOUD: 0}

    def evaluator(kind):
        def run(artifact, **values):
            del artifact
            calls[kind] += 1
            output = Path(values["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            result = output / "metrics.json"
            result.write_text("{}\n", encoding="utf-8")
            return result

        return run

    runner = bundle_module.EvaluationBundleRunner(
        ate_evaluator=evaluator(EvaluationKind.ATE),
        pointcloud_evaluator=evaluator(EvaluationKind.POINTCLOUD),
    )

    def run(source_revision="revision-1"):
        return runner.run(
            artifact_dir=artifact_dir,
            entry=_entry(EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
            experiment=experiment,
            output_root=tmp_path / "evaluation",
            source_revision=source_revision,
        )

    run()
    assert calls == {EvaluationKind.ATE: 1, EvaluationKind.POINTCLOUD: 1}

    files["ate_config"].write_text("changed ate config\n", encoding="utf-8")
    run()
    assert calls == {EvaluationKind.ATE: 2, EvaluationKind.POINTCLOUD: 1}

    files["pointcloud_gt"].write_text("changed pointcloud gt\n", encoding="utf-8")
    run()
    assert calls == {EvaluationKind.ATE: 2, EvaluationKind.POINTCLOUD: 2}

    run(source_revision="revision-2")
    assert calls == {EvaluationKind.ATE: 3, EvaluationKind.POINTCLOUD: 3}


def test_evaluator_errors_redact_url_credentials_and_control_characters(tmp_path):
    artifact_dir, files = _identity_files(tmp_path)
    experiment = _experiment(
        _input(EvaluationKind.ATE, files["ate_config"], files["ate_gt"])
    )

    def fail(*args, **values):
        del args, values
        raise ValueError(
            "bad https://user:secret@example.com/path\nnext\x00line"
        )

    records = bundle_module.EvaluationBundleRunner(ate_evaluator=fail).run(
        artifact_dir=artifact_dir,
        entry=_entry(EvaluationKind.ATE),
        experiment=experiment,
        output_root=tmp_path / "evaluation",
        source_revision="revision-1",
    )

    message = records[0].error.message
    assert "user" not in message
    assert "secret" not in message
    assert "\n" not in message
    assert "\x00" not in message
    assert "https://<redacted>@example.com/path" in message
