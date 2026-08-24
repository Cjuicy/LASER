from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import experiments.pointcloud as pointcloud_experiment
from evaluation.pointcloud import PrimaryMetrics
from experiments.config import (
    EvaluationKind,
    ExperimentConfig,
    ExperimentDatasetConfig,
    load_experiment_config,
)
from pipeline.artifacts import write_reconstruction_artifact
from pipeline.config import ReconstructionMode
from tests.test_pipeline_runner import _artifact


def test_pointcloud_experiment_accepts_only_canonical_no_loop_product(tmp_path):
    config = ExperimentConfig(
        version=1,
        evaluation=EvaluationKind.POINTCLOUD,
        reconstruction_config="configs/reconstruction/pi3_laser.yaml",
        evaluation_config="configs/evaluation/pointcloud.yaml",
        output_root=str(tmp_path),
        dataset=ExperimentDatasetConfig("fixture", "sequence", "gt.npz"),
        segmentation_methods=("depth", "geometry", "atomic"),
        reconstruction_modes=("no_loop",),
    )
    assert [entry.name for entry in config.entries] == [
        "depth-no_loop",
        "geometry-no_loop",
        "atomic-no_loop",
    ]


def test_experiment_config_rejects_unknown_and_pipeline_fields(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "version: 1\nevaluation: pointcloud\n"
        "reconstruction_config: configs/reconstruction/pi3_laser.yaml\n"
        "evaluation_config: configs/evaluation/pointcloud.yaml\n"
        "output_root: output\n"
        "dataset: {name: fixture, sequence: one, ground_truth: gt.npz}\n"
        "matrix:\n  segmentation: [depth, geometry, atomic]\n"
        "  reconstruction_modes: [no_loop]\n"
        "window: {size: 20}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown experiment field"):
        load_experiment_config(path)


def test_pointcloud_input_function_uses_explicit_ground_truth_and_config(
    tmp_path,
    monkeypatch,
):
    artifact_dir = write_reconstruction_artifact(
        _artifact(ReconstructionMode.NO_LOOP),
        tmp_path / "artifact",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    ground_truth = tmp_path / "groundtruth.npz"
    np.savez(
        ground_truth,
        point_maps=np.zeros((3, 2, 2, 3), dtype=np.float32),
        valid_mask=np.ones((3, 2, 2), dtype=bool),
    )
    observed = []

    @dataclass(frozen=True)
    class Diagnostics:
        frame_count: int

    def evaluate(estimate, truth, config):
        observed.append(
            (
                estimate.reconstruction_mode,
                truth.point_maps.shape,
                truth.valid_mask.shape,
                config.version,
            )
        )
        return type(
            "Evaluation",
            (),
            {
                "primary": PrimaryMetrics(1.0, 2.0, 3.0, 4.0, 0.5, 0.25),
                "diagnostics": Diagnostics(3),
            },
        )()

    monkeypatch.setattr(pointcloud_experiment, "evaluate_point_maps", evaluate)

    output = pointcloud_experiment.evaluate_pointcloud_inputs(
        artifact_dir,
        dataset_name="fixture",
        sequence="sequence-0",
        ground_truth=ground_truth,
        evaluation_config="configs/evaluation/pointcloud.yaml",
        output_dir=tmp_path / "evaluation",
    )

    assert observed == [
        (ReconstructionMode.NO_LOOP, (3, 2, 2, 3), (3, 2, 2), 1)
    ]
    assert output == tmp_path / "evaluation"
    assert (output / "pointcloud_metrics.json").is_file()
    assert (output / "pointcloud_sequences.csv").is_file()
