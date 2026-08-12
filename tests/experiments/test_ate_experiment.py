from __future__ import annotations

from dataclasses import replace

from experiments.config import EvaluationKind, ExperimentConfig, ExperimentDatasetConfig
from experiments.matrix import (
    ArtifactRepository,
    build_matrix,
    reconstruction_identity,
)
from pipeline.artifacts import write_reconstruction_artifact
from tests.test_pipeline_runner import _artifact
from pipeline.config import load_pipeline_config
from pipeline.config import ReconstructionMode


def _config(tmp_path):
    return ExperimentConfig(
        version=1,
        evaluation=EvaluationKind.ATE,
        reconstruction_config="configs/reconstruction/pi3_laser.yaml",
        evaluation_config="configs/evaluation/ate.yaml",
        output_root=str(tmp_path / "output"),
        dataset=ExperimentDatasetConfig(
            name="fixture",
            sequence="sequence-0",
            ground_truth=str(tmp_path / "groundtruth.txt"),
            ground_truth_format="tum",
        ),
        segmentation_methods=("depth", "geometry", "atomic"),
        reconstruction_modes=("no_loop", "traditional", "corrected"),
    )


def test_reconstruction_identity_excludes_evaluation_kind_and_config(tmp_path):
    loaded = load_pipeline_config("configs/reconstruction/pi3_laser.yaml")
    ate = _config(tmp_path)
    pointcloud = replace(
        ate,
        evaluation=EvaluationKind.POINTCLOUD,
        evaluation_config="configs/evaluation/pointcloud.yaml",
        reconstruction_modes=("no_loop",),
        dataset=replace(ate.dataset, ground_truth_format=None),
    )

    first = reconstruction_identity(
        loaded,
        input_manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )
    second = reconstruction_identity(
        loaded,
        input_manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )

    assert ate.evaluation is not pointcloud.evaluation
    assert first == second


def test_ate_config_expands_to_exact_matrix(tmp_path):
    config = _config(tmp_path)
    assert tuple(build_matrix(config.evaluation)) == config.entries


def test_identical_reconstruction_identity_reuses_complete_artifact(tmp_path):
    repository = ArtifactRepository(tmp_path / "artifacts")
    identity = "a" * 64
    calls = []

    def reconstruct(target):
        calls.append(target)
        return write_reconstruction_artifact(
            _artifact(ReconstructionMode.NO_LOOP),
            target,
            resolved_yaml="version: 2\n",
            config_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            git_commit="d" * 40,
        )

    first = repository.get_or_create(identity, reconstruct)
    second = repository.get_or_create(identity, reconstruct)

    assert first == second
    assert len(calls) == 1


def test_incomplete_artifact_directory_fails_instead_of_overwrite(tmp_path):
    repository = ArtifactRepository(tmp_path / "artifacts")
    identity = "b" * 64
    target = repository.path_for(identity)
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("{}", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="incomplete artifact"):
        repository.get_or_create(identity, lambda path: path)
