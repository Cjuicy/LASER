from __future__ import annotations

from pathlib import Path

from evaluation.trajectory import (
    evaluate_trajectory,
    load_ground_truth_trajectory,
    load_trajectory_evaluation_config,
    write_trajectory_metrics,
)
from pipeline.artifacts import load_trajectory_estimate

from .config import ExperimentConfig


def evaluate_ate_artifact(
    artifact_dir: str | Path,
    experiment: ExperimentConfig,
    output_dir: str | Path,
) -> Path:
    if experiment.dataset.ground_truth_format is None:
        raise ValueError("ATE dataset requires ground_truth_format")
    estimate = load_trajectory_estimate(artifact_dir)
    ground_truth = load_ground_truth_trajectory(
        experiment.dataset.ground_truth,
        experiment.dataset.ground_truth_format,
    )
    config = load_trajectory_evaluation_config(experiment.evaluation_config)
    metrics = evaluate_trajectory(estimate, ground_truth, config)
    return write_trajectory_metrics(
        metrics,
        output_dir,
        artifact_manifest_sha256="managed-by-experiment-artifact-repository",
    )
