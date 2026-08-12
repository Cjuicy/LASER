from __future__ import annotations

import json

import numpy as np
import torch

import evaluate_pointcloud
from pipeline.artifacts import PointMapEstimate
from pipeline.config import ReconstructionMode


def test_pointcloud_cli_uses_narrow_pointmap_loader(monkeypatch, tmp_path):
    points = np.zeros((1, 4, 4, 3), dtype=np.float32)
    points[..., 0] = np.arange(4)[None, None, :]
    points[..., 1] = np.arange(4)[None, :, None]
    points[..., 2] = 1.0
    loaded = []

    def loader(path):
        loaded.append(path)
        return PointMapEstimate(
            (0,),
            torch.from_numpy(points),
            torch.ones((1, 4, 4)),
            ReconstructionMode.NO_LOOP,
        )

    monkeypatch.setattr(evaluate_pointcloud, "load_pointmap_estimate", loader)
    monkeypatch.setattr(
        evaluate_pointcloud,
        "build_open3d_backend",
        lambda: __import__(
            "tests.evaluation.test_pointcloud_evaluator",
            fromlist=["IdentityBackend"],
        ).IdentityBackend(),
    )
    ground_truth = tmp_path / "gt.npz"
    np.savez(
        ground_truth,
        point_maps=points,
        valid_mask=np.ones(points.shape[:-1], dtype=bool),
    )
    output = tmp_path / "output"
    config = tmp_path / "pointcloud.yaml"
    config.write_text(
        "version: 1\ncenter_crop_size: 4\n"
        "alignment: umeyama_sim3_then_icp\n"
        "icp_type: point_to_point\nicp_threshold_m: 0.1\n"
        "normal_estimation: open3d_default\n"
        "fscore_thresholds_m: [0.01, 0.02, 0.05]\n",
        encoding="utf-8",
    )

    assert evaluate_pointcloud.main(
        [
            "--artifact",
            str(tmp_path / "artifact"),
            "--config",
            str(config),
            "--ground-truth",
            str(ground_truth),
            "--dataset",
            "fixture",
            "--sequence",
            "sequence-0",
            "--output",
            str(output),
        ]
    ) == 0
    assert loaded == [str(tmp_path / "artifact")]
    payload = json.loads((output / "pointcloud_metrics.json").read_text())
    assert payload["dataset"] == "fixture"
