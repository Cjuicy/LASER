from __future__ import annotations

import pytest

from experiments.config import (
    EvaluationKind,
    ExperimentConfig,
    ExperimentDatasetConfig,
    load_experiment_config,
)


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
