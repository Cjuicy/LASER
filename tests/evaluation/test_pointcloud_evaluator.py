from __future__ import annotations

import numpy as np
import pytest
import torch
from dataclasses import asdict

from evaluation.pointcloud.config import PointCloudEvaluationConfig
from evaluation.pointcloud.evaluator import (
    PointCloudGroundTruth,
    evaluate_point_maps,
)
from evaluation.pointcloud.geometry_metrics import BackendResult
from pipeline.artifacts import PointMapEstimate
from pipeline.config import ReconstructionMode


def _grid(height=4, width=4):
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.5, 1.5, width),
        indexing="ij",
    )
    return np.stack((x, y, 0.2 * x + 0.1 * y + 1.0), axis=-1)[None]


class IdentityBackend:
    def refine_and_estimate_normals(self, predicted, ground_truth, threshold_m):
        del threshold_m
        return BackendResult(
            predicted_points=np.asarray(predicted),
            ground_truth_points=np.asarray(ground_truth),
            predicted_normals=np.tile([1.0, 0.0, 0.0], (len(predicted), 1)),
            ground_truth_normals=np.tile(
                [1.0, 0.0, 0.0], (len(ground_truth), 1)
            ),
            transformation=np.eye(4),
            fitness=1.0,
            inlier_rmse=0.0,
        )


def _estimate(mode=ReconstructionMode.NO_LOOP):
    points = torch.as_tensor(_grid(), dtype=torch.float32)
    return PointMapEstimate(
        frame_ids=(0,),
        global_points=points,
        confidence=torch.ones(points.shape[:-1]),
        reconstruction_mode=mode,
    )


def _ground_truth():
    points = _grid()
    return PointCloudGroundTruth(
        point_maps=points,
        valid_mask=np.ones(points.shape[:-1], dtype=bool),
    )


def test_pointcloud_rejects_loop_artifact_before_backend_construction():
    def forbidden_backend():
        raise AssertionError("loop artifact must fail before backend construction")

    with pytest.raises(ValueError, match="requires no_loop artifact"):
        evaluate_point_maps(
            _estimate(ReconstructionMode.TRADITIONAL),
            _ground_truth(),
            PointCloudEvaluationConfig(center_crop_size=4),
            backend_factory=forbidden_backend,
        )


def test_identical_point_maps_keep_existing_primary_metrics():
    result = evaluate_point_maps(
        _estimate(),
        _ground_truth(),
        PointCloudEvaluationConfig(center_crop_size=4),
        backend=IdentityBackend(),
    )

    assert result.primary.accuracy_mean_m == pytest.approx(0.0, abs=1e-7)
    assert result.primary.completion_mean_m == pytest.approx(0.0, abs=1e-7)
    assert result.primary.normal_consistency_mean == pytest.approx(1.0)


def test_pointcloud_config_rejects_pipeline_fields(tmp_path):
    path = tmp_path / "pointcloud.yaml"
    path.write_text(
        "version: 1\ncenter_crop_size: 4\nalignment: umeyama_sim3_then_icp\n"
        "icp_type: point_to_point\nicp_threshold_m: 0.1\n"
        "normal_estimation: open3d_default\nfscore_thresholds_m: [0.01]\n"
        "window: {size: 20}\n",
        encoding="utf-8",
    )

    from evaluation.pointcloud.config import load_pointcloud_evaluation_config

    with pytest.raises(ValueError, match="unknown point-cloud evaluation field"):
        load_pointcloud_evaluation_config(path)


def test_new_pointcloud_metrics_match_existing_implementation():
    from mv_recon.geometry_metrics import (
        BackendResult as OldBackendResult,
        evaluate_point_maps as old_evaluate,
    )
    from mv_recon.protocol import GeometryProtocol

    points = _grid(5, 6)
    predicted = points * 2.0 + np.array([3.0, -1.0, 0.5])
    mask = np.ones(points.shape[:-1], dtype=bool)

    class OldIdentityBackend:
        def refine_and_estimate_normals(
            self, predicted_points, ground_truth_points, threshold_m
        ):
            del threshold_m
            return OldBackendResult(
                predicted_points=np.asarray(predicted_points),
                ground_truth_points=np.asarray(ground_truth_points),
                predicted_normals=np.tile(
                    [1.0, 0.0, 0.0], (len(predicted_points), 1)
                ),
                ground_truth_normals=np.tile(
                    [1.0, 0.0, 0.0], (len(ground_truth_points), 1)
                ),
                transformation=np.eye(4),
                fitness=1.0,
                inlier_rmse=0.0,
            )

    old = old_evaluate(
        predicted,
        points,
        mask,
        GeometryProtocol(
            center_crop_size=4,
            alignment="umeyama_sim3_then_icp",
            icp_type="point_to_point",
            icp_threshold_m=0.1,
            normal_estimation="open3d_default",
            fscore_thresholds_m=(0.01, 0.02, 0.05),
        ),
        backend=OldIdentityBackend(),
    )
    new = evaluate_point_maps(
        PointMapEstimate(
            (0,),
            torch.as_tensor(predicted, dtype=torch.float32),
            torch.ones(predicted.shape[:-1]),
            ReconstructionMode.NO_LOOP,
        ),
        PointCloudGroundTruth(points, mask),
        PointCloudEvaluationConfig(center_crop_size=4),
        backend=IdentityBackend(),
    )

    assert asdict(new.primary) == pytest.approx(
        asdict(old.primary),
        abs=1e-7,
    )
    assert asdict(new.diagnostics.directional_normals) == pytest.approx(
        asdict(old.diagnostics.directional_normals),
        abs=1e-12,
    )
    assert new.diagnostics.chamfer_l1_m == pytest.approx(
        old.diagnostics.chamfer_l1_m,
        abs=1e-7,
    )
    assert [asdict(item) for item in new.diagnostics.thresholds] == [
        pytest.approx(asdict(item), abs=1e-12)
        for item in old.diagnostics.thresholds
    ]
