from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from evaluation.trajectory import (
    evaluate_trajectory,
    load_ground_truth_trajectory,
    load_trajectory_evaluation_config,
    write_trajectory_metrics,
)
from pipeline.artifacts import load_trajectory_estimate


def _manifest_sha256(artifact_dir: str | Path) -> str:
    path = Path(artifact_dir) / "manifest.json"
    if not path.is_file():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trajectory artifact")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--ground-truth-format",
        required=True,
        choices=("sintel", "replica", "tum", "tartanair"),
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    estimate = load_trajectory_estimate(arguments.artifact)
    ground_truth = load_ground_truth_trajectory(
        arguments.ground_truth,
        arguments.ground_truth_format,
    )
    config = load_trajectory_evaluation_config(arguments.config)
    metrics = evaluate_trajectory(estimate, ground_truth, config)
    output = write_trajectory_metrics(
        metrics,
        arguments.output,
        artifact_manifest_sha256=_manifest_sha256(arguments.artifact),
    )
    print(
        f"ATE={metrics.ate_rmse_m:.9f} "
        f"RPE_t={metrics.rpe_translation_rmse_m:.9f} "
        f"RPE_r={metrics.rpe_rotation_rmse_deg:.9f} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
