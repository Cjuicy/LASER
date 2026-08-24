from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluation.pointcloud import (
    PointCloudGroundTruth,
    SequencePointCloudResult,
    aggregate_pointcloud_results,
    evaluate_point_maps,
    load_pointcloud_evaluation_config,
    write_pointcloud_results,
)
from pipeline.artifacts import load_pointmap_estimate

from .config import ExperimentConfig


def evaluate_pointcloud_artifact(
    artifact_dir: str | Path,
    experiment: ExperimentConfig,
    output_dir: str | Path,
) -> Path:
    return evaluate_pointcloud_inputs(
        artifact_dir,
        dataset_name=experiment.dataset.name,
        sequence=experiment.dataset.sequence,
        ground_truth=experiment.dataset.ground_truth,
        evaluation_config=experiment.evaluation_config,
        output_dir=output_dir,
    )


def evaluate_pointcloud_inputs(
    artifact_dir: str | Path,
    *,
    dataset_name: str,
    sequence: str,
    ground_truth: str | Path,
    evaluation_config: str | Path,
    output_dir: str | Path,
) -> Path:
    estimate = load_pointmap_estimate(artifact_dir)
    config = load_pointcloud_evaluation_config(evaluation_config)
    with np.load(ground_truth, allow_pickle=False) as data:
        truth = PointCloudGroundTruth(
            data["point_maps"],
            data["valid_mask"],
        )
    evaluation = evaluate_point_maps(estimate, truth, config)
    result = SequencePointCloudResult(
        dataset_name,
        sequence,
        len(estimate.frame_ids),
        evaluation.primary,
        asdict(evaluation.diagnostics),
    )
    summary = aggregate_pointcloud_results(
        dataset_name,
        (result,),
        expected_sequences=(sequence,),
    )
    return write_pointcloud_results(summary, (result,), output_dir)
