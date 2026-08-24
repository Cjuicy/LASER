from __future__ import annotations

import hashlib
from pathlib import Path

from evaluation.trajectory import (
    evaluate_trajectory,
    load_ground_truth_trajectory,
    load_trajectory_evaluation_config,
    write_trajectory_metrics,
)
from pipeline.artifacts import load_trajectory_estimate

from .config import ExperimentConfig


def _artifact_manifest_sha256(artifact_dir: str | Path) -> str:
    manifest = Path(artifact_dir) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"artifact manifest does not exist: {manifest}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def evaluate_ate_artifact(
    artifact_dir: str | Path,
    experiment: ExperimentConfig,
    output_dir: str | Path,
) -> Path:
    if experiment.dataset.ground_truth_format is None:
        raise ValueError("ATE dataset requires ground_truth_format")
    return evaluate_ate_inputs(
        artifact_dir,
        ground_truth=experiment.dataset.ground_truth,
        ground_truth_format=experiment.dataset.ground_truth_format,
        evaluation_config=experiment.evaluation_config,
        output_dir=output_dir,
    )


def evaluate_ate_inputs(
    artifact_dir: str | Path,
    *,
    ground_truth: str | Path,
    ground_truth_format: str,
    evaluation_config: str | Path,
    output_dir: str | Path,
) -> Path:
    estimate = load_trajectory_estimate(artifact_dir)
    truth = load_ground_truth_trajectory(
        ground_truth,
        ground_truth_format,
    )
    config = load_trajectory_evaluation_config(evaluation_config)
    metrics = evaluate_trajectory(estimate, truth, config)
    return write_trajectory_metrics(
        metrics,
        output_dir,
        artifact_manifest_sha256=_artifact_manifest_sha256(artifact_dir),
    )
