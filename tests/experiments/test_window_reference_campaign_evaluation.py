from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.pointcloud.config import PointCloudEvaluationConfig
from evaluation.pointcloud.evaluator import evaluate_point_maps
from evaluation.pointcloud.geometry_metrics import (
    BackendResult,
    DirectionalNormalMetrics,
    GeometryDiagnostics,
    GeometryEvaluation,
    PrimaryMetrics,
    ThresholdMetrics,
)
from evaluation.trajectory.config import TrajectoryEvaluationConfig
from evaluation.trajectory.evaluator import GroundTruthTrajectory
from evaluation.trajectory.results import TrajectoryMetrics
from pipeline.artifacts import PointMapEstimate, TrajectoryEstimate
from pipeline.config import ReconstructionMode

from experiments.window_reference_campaign.config import (
    DatasetKind,
    EvaluationKind,
    LoadedCampaignConfig,
)
from experiments.window_reference_campaign.results import (
    POINTCLOUD_METRICS,
    TRAJECTORY_METRICS,
)
from experiments.window_reference_campaign.runner import EvaluationOutput
from experiments.window_reference_campaign.scenes import ResolvedFrameSelection
from experiments.window_reference_campaign.staging import StagedScene

from experiments.window_reference_campaign.evaluation import (
    EvaluationDependencies,
    evaluate_artifact,
    evaluate_kitti_artifact,
    evaluate_pointcloud_artifact,
)


class LiteralBackend:
    def refine_and_estimate_normals(self, predicted, ground_truth, threshold_m):
        normals_pred = np.tile([0.0, 0.0, 1.0], (len(predicted), 1))
        normals_gt = np.tile([0.0, 0.0, 1.0], (len(ground_truth), 1))
        return BackendResult(
            predicted_points=predicted,
            ground_truth_points=ground_truth,
            predicted_normals=normals_pred,
            ground_truth_normals=normals_gt,
            transformation=np.eye(4),
            fitness=1.0,
            inlier_rmse=0.0,
        )


def _staged(
    tmp_path: Path,
    *,
    evaluation_kind: EvaluationKind,
    source_frame_ids: tuple[int, ...] = (0,),
    pointcloud_gt_path: Path | None = None,
    poses_path: Path | None = None,
) -> StagedScene:
    image_dir = tmp_path / "staged" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for ordinal in range(len(source_frame_ids)):
        (image_dir / f"{ordinal:06d}.png").write_bytes(b"fixture")
    selection = ResolvedFrameSelection(
        0,
        max(2, len(source_frame_ids)),
        1,
        source_frame_ids,
    )
    staged = StagedScene(
        scene_id="fixture",
        dataset=(
            DatasetKind.KITTI
            if evaluation_kind is EvaluationKind.INTERNAL_TRAJECTORY
            else DatasetKind.SEVEN_SCENES
        ),
        scene="fixture",
        slice_id="f000000-000004-s1",
        image_dir=image_dir,
        source_frame_ids=source_frame_ids,
        selection=selection,
        poses_path=poses_path,
        pointcloud_gt_path=pointcloud_gt_path,
        manifest_path=tmp_path / "staged" / "staging.json",
        manifest_sha256="a" * 64,
    )
    # Task 2's current StagedScene predates the explicit field in the Task 5
    # brief.  Keep the fixture aligned with the brief without changing the
    # production staging dataclass in this task.
    object.__setattr__(staged, "evaluation_kind", evaluation_kind)
    return staged


def _dependencies(**overrides) -> EvaluationDependencies:
    values = {
        "load_pointmap_estimate": lambda _: (_ for _ in ()).throw(
            AssertionError("point-map loader was not expected")
        ),
        "load_trajectory_estimate": lambda _: (_ for _ in ()).throw(
            AssertionError("trajectory loader was not expected")
        ),
        "load_ground_truth_trajectory": lambda *_: (_ for _ in ()).throw(
            AssertionError("ground-truth trajectory loader was not expected")
        ),
        "load_pointcloud_config": lambda _: PointCloudEvaluationConfig(),
        "load_trajectory_config": lambda _: TrajectoryEvaluationConfig(),
        "pointcloud_backend_factory": LiteralBackend,
        "evaluate_point_maps": lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("point-map evaluator was not expected")
        ),
        "evaluate_trajectory": lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trajectory evaluator was not expected")
        ),
    }
    values.update(overrides)
    return EvaluationDependencies(**values)


def _pointcloud_fixture(tmp_path: Path):
    points = np.array(
        [[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
          [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]],
        dtype=np.float32,
    )
    estimate = PointMapEstimate(
        frame_ids=(0,),
        global_points=torch.from_numpy(points),
        confidence=torch.ones((1, 2, 2)),
        reconstruction_mode=ReconstructionMode.NO_LOOP,
    )
    gt = tmp_path / "ground_truth.npz"
    np.savez_compressed(
        gt,
        point_maps=points,
        valid_mask=np.ones((1, 2, 2), dtype=bool),
        frame_ids=np.array([0]),
    )
    staged = _staged(
        tmp_path,
        evaluation_kind=EvaluationKind.POINTCLOUD,
        pointcloud_gt_path=gt,
    )
    dependencies = _dependencies(
        load_pointmap_estimate=lambda _: estimate,
        load_pointcloud_config=lambda _: PointCloudEvaluationConfig(center_crop_size=2),
        pointcloud_backend_factory=LiteralBackend,
        evaluate_point_maps=evaluate_point_maps,
    )
    return staged, dependencies


def test_pointcloud_adapter_retains_primary_chamfer_and_1_2_5cm(tmp_path):
    staged, dependencies = _pointcloud_fixture(tmp_path)

    result = evaluate_pointcloud_artifact(
        tmp_path / "artifact",
        staged,
        tmp_path / "ignored.yaml",
        tmp_path / "evaluation",
        dependencies=dependencies,
    )

    assert result.evaluation_kind is EvaluationKind.POINTCLOUD
    assert set(result.metrics) == set(POINTCLOUD_METRICS)
    assert result.metrics["accuracy_mean_m"] == pytest.approx(0.0)
    assert result.metrics["completion_median_m"] == pytest.approx(0.0)
    assert result.metrics["normal_consistency_mean"] == pytest.approx(1.0)
    assert result.metrics["chamfer_l1_m"] == pytest.approx(0.0)
    assert result.metrics["fscore_1cm"] == pytest.approx(1.0)
    assert result.metrics["fscore_2cm"] == pytest.approx(1.0)
    assert result.metrics["fscore_5cm"] == pytest.approx(1.0)
    assert (tmp_path / "evaluation" / "pointcloud_metrics.json").is_file()


def _poses(count: int) -> torch.Tensor:
    poses = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    poses[:, 0, 3] = torch.arange(count, dtype=torch.float64)
    return poses


def test_kitti_adapter_uses_12_value_loader_and_internal_metric_names(tmp_path):
    poses = _poses(4)
    poses_path = tmp_path / "poses.txt"
    np.savetxt(poses_path, poses[:, :3].reshape(4, 12).numpy())
    staged = _staged(
        tmp_path,
        evaluation_kind=EvaluationKind.INTERNAL_TRAJECTORY,
        source_frame_ids=(0, 1, 2, 3),
        poses_path=poses_path,
    )
    seen = {}
    dependencies = _dependencies(
        load_trajectory_estimate=lambda _: TrajectoryEstimate(
            (0, 1, 2, 3), poses
        ),
        load_ground_truth_trajectory=lambda path, format_name: (
            seen.update(path=path, format_name=format_name)
            or GroundTruthTrajectory((0, 1, 2, 3), poses)
        ),
        load_trajectory_config=lambda _: TrajectoryEvaluationConfig(),
        evaluate_trajectory=lambda estimate, truth, config: TrajectoryMetrics(
            ate_rmse_m=0.0,
            rpe_translation_rmse_m=0.0,
            rpe_rotation_rmse_deg=0.0,
            matched_frame_count=4,
        ),
    )

    result = evaluate_kitti_artifact(
        tmp_path / "artifact",
        staged,
        tmp_path / "ignored.yaml",
        tmp_path / "evaluation",
        dependencies=dependencies,
    )

    assert seen == {"path": poses_path, "format_name": "replica"}
    assert set(result.metrics) == set(TRAJECTORY_METRICS)
    assert result.metrics["internal_ate_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["internal_rpe_translation_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["internal_rpe_rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-12)
    assert result.metrics["internal_matched_frame_count"] == 4
    assert all(not name.startswith(("ate_", "rpe_")) for name in result.metrics)


def test_kitti_adapter_rejects_partial_match_count(tmp_path):
    poses = _poses(4)
    poses_path = tmp_path / "poses.txt"
    np.savetxt(poses_path, poses[:, :3].reshape(4, 12).numpy())
    staged = _staged(
        tmp_path,
        evaluation_kind=EvaluationKind.INTERNAL_TRAJECTORY,
        source_frame_ids=(0, 1, 2, 3),
        poses_path=poses_path,
    )
    dependencies = _dependencies(
        load_trajectory_estimate=lambda _: TrajectoryEstimate((0, 1, 2, 3), poses),
        load_ground_truth_trajectory=lambda *_: GroundTruthTrajectory(
            (0, 1, 2, 3), poses
        ),
        evaluate_trajectory=lambda *_: TrajectoryMetrics(
            ate_rmse_m=0.0,
            rpe_translation_rmse_m=0.0,
            rpe_rotation_rmse_deg=0.0,
            matched_frame_count=3,
        ),
    )

    with pytest.raises(ValueError, match="matched frame count"):
        evaluate_kitti_artifact(
            tmp_path / "artifact", staged, tmp_path / "config.yaml",
            tmp_path / "evaluation", dependencies=dependencies,
        )


def _none_request(tmp_path: Path, staged: StagedScene):
    # Dispatch only needs the staged scene and output/artifact boundaries.  A
    # test-local namespace keeps this test focused on the adapter contract;
    # the real RunRequest is exercised by the runner's own test suite.
    return SimpleNamespace(
        staged=staged,
        attempt_dir=tmp_path / "run" / "attempts" / "000001",
        artifact_dir=tmp_path / "run" / "artifact",
    )


def _loaded_none() -> LoadedCampaignConfig:
    return SimpleNamespace(config=SimpleNamespace(evaluation=SimpleNamespace(
        pointcloud_config=Path("pointcloud.yaml"),
        trajectory_config=Path("trajectory.yaml"),
    )))


def test_no_gt_dispatch_emits_no_fabricated_quality_metrics(tmp_path):
    staged = _staged(tmp_path, evaluation_kind=EvaluationKind.NONE)
    request = _none_request(tmp_path, staged)
    execution = SimpleNamespace(artifact_dir=request.artifact_dir)

    result = evaluate_artifact(request, execution, _loaded_none())

    assert result == EvaluationOutput(EvaluationKind.NONE, None, ())


def test_pointcloud_adapter_rejects_gt_frame_ids_different_from_staging(tmp_path):
    staged, dependencies = _pointcloud_fixture(tmp_path)
    with np.load(staged.pointcloud_gt_path, allow_pickle=False) as original:
        np.savez_compressed(
            staged.pointcloud_gt_path,
            point_maps=original["point_maps"],
            valid_mask=original["valid_mask"],
            frame_ids=np.array([99]),
        )
    with pytest.raises(ValueError, match="frame IDs"):
        evaluate_pointcloud_artifact(
            tmp_path / "artifact", staged, tmp_path / "config.yaml",
            tmp_path / "evaluation", dependencies=dependencies,
        )


def test_adapter_rejects_non_finite_evaluator_result(tmp_path):
    staged, dependencies = _pointcloud_fixture(tmp_path)
    dependencies = replace(
        dependencies,
        evaluate_point_maps=lambda *args, **kwargs: _geometry_evaluation(
            accuracy_mean_m=float("nan")
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        evaluate_pointcloud_artifact(
            tmp_path / "artifact", staged, tmp_path / "config.yaml",
            tmp_path / "evaluation", dependencies=dependencies,
        )


def test_pointcloud_adapter_maps_thresholds_by_value_not_position(tmp_path):
    staged, dependencies = _pointcloud_fixture(tmp_path)
    dependencies = replace(
        dependencies,
        evaluate_point_maps=lambda *args, **kwargs: _geometry_evaluation(
            threshold_values=(0.05, 0.01, 0.02),
        ),
    )

    result = evaluate_pointcloud_artifact(
        tmp_path / "artifact", staged, tmp_path / "config.yaml",
        tmp_path / "evaluation", dependencies=dependencies,
    )

    assert result.metrics["precision_1cm"] == pytest.approx(0.1)
    assert result.metrics["precision_2cm"] == pytest.approx(0.2)
    assert result.metrics["precision_5cm"] == pytest.approx(0.5)


def test_pointcloud_adapter_rejects_dense_artifact_frame_misalignment(tmp_path):
    staged, dependencies = _pointcloud_fixture(tmp_path)
    points = np.zeros((1, 2, 2, 3), dtype=np.float32)
    estimate = PointMapEstimate(
        frame_ids=(9,),
        global_points=torch.from_numpy(points),
        confidence=torch.ones((1, 2, 2)),
        reconstruction_mode=ReconstructionMode.NO_LOOP,
    )
    dependencies = replace(dependencies, load_pointmap_estimate=lambda _: estimate)

    with pytest.raises(ValueError, match="frame IDs"):
        evaluate_pointcloud_artifact(
            tmp_path / "artifact", staged, tmp_path / "config.yaml",
            tmp_path / "evaluation", dependencies=dependencies,
        )


def _geometry_evaluation(
    *,
    accuracy_mean_m: float = 0.0,
    threshold_values: tuple[float, ...] = (0.01, 0.02, 0.05),
) -> GeometryEvaluation:
    primary = PrimaryMetrics(
        accuracy_mean_m=accuracy_mean_m,
        accuracy_median_m=0.0,
        completion_mean_m=0.0,
        completion_median_m=0.0,
        normal_consistency_mean=1.0,
        normal_consistency_median=1.0,
    )
    directional = DirectionalNormalMetrics(1.0, 1.0, 1.0, 1.0)
    thresholds = tuple(
        ThresholdMetrics(value, value * 10.0, value * 10.0, value * 10.0)
        for value in threshold_values
    )
    diagnostics = GeometryDiagnostics(
        umeyama_scale=1.0,
        icp_transformation=tuple(
            tuple(float(item == col) for col in range(4)) for item in range(4)
        ),
        icp_fitness=1.0,
        icp_inlier_rmse=0.0,
        predicted_point_count=4,
        ground_truth_point_count=4,
        directional_normals=directional,
        chamfer_l1_m=0.0,
        thresholds=thresholds,
    )
    return GeometryEvaluation(primary, diagnostics)
