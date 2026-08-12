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
    estimate = load_pointmap_estimate(artifact_dir)
    config = load_pointcloud_evaluation_config(experiment.evaluation_config)
    with np.load(experiment.dataset.ground_truth, allow_pickle=False) as data:
        ground_truth = PointCloudGroundTruth(
            data["point_maps"],
            data["valid_mask"],
        )
    evaluation = evaluate_point_maps(estimate, ground_truth, config)
    result = SequencePointCloudResult(
        experiment.dataset.name,
        experiment.dataset.sequence,
        len(estimate.frame_ids),
        evaluation.primary,
        asdict(evaluation.diagnostics),
    )
    summary = aggregate_pointcloud_results(
        experiment.dataset.name,
        (result,),
        expected_sequences=(experiment.dataset.sequence,),
    )
    return write_pointcloud_results(summary, (result,), output_dir)
