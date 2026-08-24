from __future__ import annotations

from pathlib import Path

import pytest

import run_experiment_matrix as cli
from experiments.config import (
    CapabilityExperimentConfig,
    EvaluationInputConfig,
    EvaluationKind,
    ExperimentConfig,
    ExperimentDatasetConfig,
)
from experiments.evaluation_bundle import (
    EvaluatorError,
    EvaluatorRecord,
    EvaluatorStatus,
)
from experiments.matrix import CapabilityMatrixEntry, MatrixEntry
from experiments.runner import (
    CapabilityExperimentRunRecord,
    ExperimentRunRecord,
)
from pipeline.config import (
    PredictionCacheMode,
    ReconstructionMode,
    SegmentationMethod,
)


def _capability_config():
    return CapabilityExperimentConfig(
        version=2,
        reconstruction_config="reconstruction.yaml",
        output_root="output",
        dataset_name="fixture",
        sequence="sequence-0",
        evaluator_inputs={
            EvaluationKind.ATE: EvaluationInputConfig(
                EvaluationKind.ATE,
                "ate.yaml",
                "trajectory.txt",
                "tum",
            ),
            EvaluationKind.POINTCLOUD: EvaluationInputConfig(
                EvaluationKind.POINTCLOUD,
                "pointcloud.yaml",
                "pointcloud.npz",
            ),
        },
        segmentation_methods=("depth", "geometry", "atomic"),
        refinement_states=(False, True),
    )


def _evaluation(kind, status):
    error = (
        EvaluatorError("ValueError", "failed")
        if status in {EvaluatorStatus.FAILED, EvaluatorStatus.BLOCKED}
        else None
    )
    return EvaluatorRecord(
        kind=kind,
        status=status,
        identity_digest=("a" * 64 if status is EvaluatorStatus.PASSED else None),
        output_path=("metrics.json" if status is EvaluatorStatus.PASSED else None),
        error=error,
    )


def _capability_record(name_index, statuses):
    entry = CapabilityMatrixEntry(
        segmentation_method=(
            SegmentationMethod.DEPTH
            if name_index == 0
            else SegmentationMethod.GEOMETRY
        ),
        reconstruction_mode=ReconstructionMode.NO_LOOP,
        window_reference_enabled=False,
        evaluator_kinds=(EvaluationKind.ATE, EvaluationKind.POINTCLOUD),
    )
    return CapabilityExperimentRunRecord(
        entry=entry,
        reconstruction_identity=(
            None if EvaluatorStatus.BLOCKED in statuses else "b" * 64
        ),
        artifact_dir=(
            None if EvaluatorStatus.BLOCKED in statuses else Path("artifact")
        ),
        evaluations=tuple(
            _evaluation(kind, status)
            for kind, status in zip(EvaluationKind, statuses, strict=True)
        ),
        prediction_cache_mode=PredictionCacheMode.READONLY,
    )


@pytest.mark.parametrize(
    ("records", "expected_status"),
    [
        (
            (
                _capability_record(
                    0,
                    (EvaluatorStatus.PASSED, EvaluatorStatus.SKIPPED),
                ),
            ),
            0,
        ),
        (
            (
                _capability_record(
                    0,
                    (EvaluatorStatus.PASSED, EvaluatorStatus.SKIPPED),
                ),
                _capability_record(
                    1,
                    (EvaluatorStatus.FAILED, EvaluatorStatus.BLOCKED),
                ),
            ),
            1,
        ),
    ],
    ids=["success", "failed-or-blocked"],
)
def test_version_two_cli_reports_evaluator_status_counts(
    monkeypatch,
    capsys,
    records,
    expected_status,
):
    monkeypatch.setattr(cli, "load_experiment_config", lambda path: _capability_config())
    monkeypatch.setattr(cli, "run_matrix", lambda *args, **values: records)

    status = cli.main(["--config", "experiment.yaml"])

    counts = {status.value: 0 for status in EvaluatorStatus}
    for record in records:
        for evaluation in record.evaluations:
            counts[evaluation.status.value] += 1
    assert capsys.readouterr().out.strip() == (
        f"entries={len(records)} evaluators={2 * len(records)} "
        f"passed={counts['passed']} skipped={counts['skipped']} "
        f"failed={counts['failed']} blocked={counts['blocked']}"
    )
    assert status == expected_status


def test_version_one_cli_output_is_unchanged(monkeypatch, capsys):
    experiment = ExperimentConfig(
        version=1,
        evaluation=EvaluationKind.ATE,
        reconstruction_config="reconstruction.yaml",
        evaluation_config="ate.yaml",
        output_root="output",
        dataset=ExperimentDatasetConfig(
            "fixture",
            "sequence-0",
            "trajectory.txt",
            "tum",
        ),
        segmentation_methods=("depth", "geometry", "atomic"),
        reconstruction_modes=("no_loop", "traditional", "corrected"),
    )
    record = ExperimentRunRecord(
        entry=MatrixEntry(
            SegmentationMethod.DEPTH,
            ReconstructionMode.NO_LOOP,
        ),
        reconstruction_identity="a" * 64,
        artifact_dir=Path("artifact"),
        evaluation_output=Path("evaluation"),
        prediction_cache_mode=PredictionCacheMode.READONLY,
    )
    monkeypatch.setattr(cli, "load_experiment_config", lambda path: experiment)
    monkeypatch.setattr(cli, "run_matrix", lambda *args, **values: (record,))

    assert cli.main(["--config", "experiment.yaml"]) == 0
    assert capsys.readouterr().out.strip() == "entries=1 evaluation=ate"
