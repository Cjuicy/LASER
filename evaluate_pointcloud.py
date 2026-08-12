from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np

from evaluation.pointcloud import (
    PointCloudGroundTruth,
    SequencePointCloudResult,
    aggregate_pointcloud_results,
    evaluate_point_maps,
    load_pointcloud_evaluation_config,
    write_pointcloud_results,
)
from evaluation.pointcloud.evaluator import build_open3d_backend
from pipeline.artifacts import load_pointmap_estimate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate no-loop point-map artifact")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    estimate = load_pointmap_estimate(arguments.artifact)
    config = load_pointcloud_evaluation_config(arguments.config)
    try:
        data = np.load(arguments.ground_truth, allow_pickle=False)
        ground_truth = PointCloudGroundTruth(
            point_maps=data["point_maps"],
            valid_mask=data["valid_mask"],
        )
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError("ground-truth NPZ must contain point_maps and valid_mask") from exc
    evaluation = evaluate_point_maps(
        estimate,
        ground_truth,
        config,
        backend_factory=build_open3d_backend,
    )
    result = SequencePointCloudResult(
        arguments.dataset,
        arguments.sequence,
        len(estimate.frame_ids),
        evaluation.primary,
        asdict(evaluation.diagnostics),
    )
    summary = aggregate_pointcloud_results(
        arguments.dataset,
        (result,),
        expected_sequences=(arguments.sequence,),
    )
    output = write_pointcloud_results(summary, (result,), arguments.output)
    print(
        f"accuracy={evaluation.primary.accuracy_mean_m:.9f} "
        f"completion={evaluation.primary.completion_mean_m:.9f} "
        f"normal_consistency={evaluation.primary.normal_consistency_mean:.9f} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
