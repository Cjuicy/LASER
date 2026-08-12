from __future__ import annotations

import json

import torch

import evaluate_ate
from pipeline.artifacts import TrajectoryEstimate


def test_ate_cli_loads_trajectory_view_only(monkeypatch, tmp_path):
    loaded = []
    poses = torch.eye(4).repeat(3, 1, 1)

    def trajectory_loader(path):
        loaded.append(path)
        return TrajectoryEstimate((0, 1, 2), poses)

    monkeypatch.setattr(evaluate_ate, "load_trajectory_estimate", trajectory_loader)
    ground_truth = tmp_path / "groundtruth.txt"
    ground_truth.write_text(
        "0 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    output = tmp_path / "results"

    exit_code = evaluate_ate.main(
        [
            "--artifact",
            str(tmp_path / "artifact"),
            "--config",
            "configs/evaluation/ate.yaml",
            "--ground-truth",
            str(ground_truth),
            "--ground-truth-format",
            "tum",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert loaded == [str(tmp_path / "artifact")]
    payload = json.loads((output / "trajectory_metrics.json").read_text())
    assert payload["matched_frame_count"] == 3
