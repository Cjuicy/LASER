from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.pointcloud.config import PointCloudEvaluationConfig
from evaluation.pointcloud.evaluator import evaluate_point_maps
from evaluation.pointcloud.geometry_metrics import BackendResult
from pipeline.artifacts import PointMapEstimate
from pipeline.config import ReconstructionMode, SegmentationMethod

import experiments.window_reference_campaign.evaluation as campaign_evaluation
from experiments.window_reference_campaign.config import DatasetKind, EvaluationKind
from experiments.window_reference_campaign.evaluation import EvaluationDependencies
from experiments.window_reference_campaign.runner import RunnerDependencies
from experiments.window_reference_campaign.scenes import (
    ResolvedFrameSelection,
    ResolvedScene,
)
from experiments.window_reference_campaign.staging import (
    StagedScene,
    load_valid_staging,
    preview_staging_manifest,
    stage_scene,
)
from experiments.window_reference_campaign.synthetic import build_synthetic_fixture


class _LiteralBackend:
    """Deterministic test backend; geometry math remains the production evaluator."""

    def refine_and_estimate_normals(self, predicted, ground_truth, threshold_m):
        return BackendResult(
            predicted_points=predicted,
            ground_truth_points=ground_truth,
            predicted_normals=np.tile([0.0, 0.0, 1.0], (len(predicted), 1)),
            ground_truth_normals=np.tile([0.0, 0.0, 1.0], (len(ground_truth), 1)),
            transformation=np.eye(4),
            fitness=1.0,
            inlier_rmse=0.0,
        )


def _staged(
    tmp_path: Path,
    *,
    kind: EvaluationKind,
    dataset: DatasetKind = DatasetKind.SYNTHETIC,
    poses_path: Path | None = None,
    pointcloud_gt_path: Path | None = None,
) -> StagedScene:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "000000.png").write_bytes(b"fixture")
    staged = StagedScene(
        scene_id="fixture",
        dataset=dataset,
        scene="fixture",
        slice_id="f000000-000001-s1",
        image_dir=image_dir,
        source_frame_ids=(0,),
        selection=ResolvedFrameSelection(0, 1, 1, (0,)),
        evaluation_kind=kind,
        poses_path=poses_path,
        pointcloud_gt_path=pointcloud_gt_path,
        manifest_path=tmp_path / "staging.json",
        identity_manifest_sha256="a" * 64,
        manifest_sha256="a" * 64,
    )
    return staged


def test_staged_scene_has_typed_evaluation_kind(tmp_path):
    staged = _staged(tmp_path, kind=EvaluationKind.NONE)
    assert staged.evaluation_kind is EvaluationKind.NONE


def test_staged_scene_rejects_mixed_evaluation_payload(tmp_path):
    poses = tmp_path / "poses.txt"
    poses.write_text("0 " * 12, encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation_kind|poses|GT|synthetic"):
        _staged(
            tmp_path,
            kind=EvaluationKind.NONE,
            dataset=DatasetKind.KITTI,
            poses_path=poses,
        )


@pytest.mark.parametrize(
    "method",
    [SegmentationMethod.DEPTH, SegmentationMethod.GEOMETRY, SegmentationMethod.ATOMIC],
)
def test_synthetic_rejection_pose_is_se3_and_diagnostics_are_deterministic(method):
    first = build_synthetic_fixture(method)
    second = build_synthetic_fixture(method)

    rejection_poses = first.rejection_poses
    rotation = rejection_poses[:, :3, :3]
    identity = torch.eye(3, dtype=rotation.dtype).expand_as(rotation)
    torch.testing.assert_close(
        rotation @ rotation.transpose(-1, -2), identity, rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones(rotation.shape[0], dtype=rotation.dtype),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        rejection_poses[:, 3, :],
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=rejection_poses.dtype).expand(
            rejection_poses.shape[0], -1
        ),
        rtol=0.0,
        atol=1e-6,
    )
    assert rejection_poses[1, 2, 3].item() == pytest.approx(4.0)
    assert [item.diagnostics for item in first.merge_results] == [
        item.diagnostics for item in second.merge_results
    ]
    assert [item.diagnostics for item in first.rejection_results] == [
        item.diagnostics for item in second.rejection_results
    ]


def test_runner_default_dependency_dispatches_real_none_evaluator(tmp_path):
    staged = _staged(tmp_path, kind=EvaluationKind.NONE)
    request = SimpleNamespace(staged=staged, attempt_dir=tmp_path / "attempt")
    execution = SimpleNamespace(artifact_dir=tmp_path / "artifact")
    loaded = SimpleNamespace(config=SimpleNamespace(evaluation=SimpleNamespace()))

    result = RunnerDependencies().evaluate_artifact(request, execution, loaded)

    assert result.evaluation_kind is EvaluationKind.NONE
    assert result.metrics is None


def test_runner_default_dependency_dispatches_real_pointcloud_evaluator(tmp_path, monkeypatch):
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
    ground_truth = tmp_path / "ground_truth.npz"
    np.savez_compressed(
        ground_truth,
        point_maps=points,
        valid_mask=np.ones((1, 2, 2), dtype=bool),
        frame_ids=np.array([0]),
    )
    staged = _staged(
        tmp_path,
        kind=EvaluationKind.POINTCLOUD,
        dataset=DatasetKind.NRGBD,
        pointcloud_gt_path=ground_truth,
    )
    dependencies = EvaluationDependencies(
        load_pointmap_estimate=lambda _: estimate,
        load_trajectory_estimate=lambda _: (_ for _ in ()).throw(
            AssertionError("trajectory loader was not expected")
        ),
        load_ground_truth_trajectory=lambda *_: (_ for _ in ()).throw(
            AssertionError("trajectory GT loader was not expected")
        ),
        load_pointcloud_config=lambda _: PointCloudEvaluationConfig(center_crop_size=2),
        load_trajectory_config=lambda _: (_ for _ in ()).throw(
            AssertionError("trajectory config loader was not expected")
        ),
        pointcloud_backend_factory=_LiteralBackend,
        evaluate_point_maps=evaluate_point_maps,
        evaluate_trajectory=lambda *_: (_ for _ in ()).throw(
            AssertionError("trajectory evaluator was not expected")
        ),
    )
    monkeypatch.setattr(
        campaign_evaluation,
        "default_evaluation_dependencies",
        lambda: dependencies,
    )
    request = SimpleNamespace(
        staged=staged,
        attempt_dir=tmp_path / "attempt",
        artifact_dir=tmp_path / "artifact",
    )
    execution = SimpleNamespace(artifact_dir=request.artifact_dir)
    loaded = SimpleNamespace(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(pointcloud_config=tmp_path / "pointcloud.yaml")
        )
    )

    result = RunnerDependencies().evaluate_artifact(request, execution, loaded)

    assert result.evaluation_kind is EvaluationKind.POINTCLOUD
    assert result.metrics["fscore_1cm"] == pytest.approx(1.0)


def test_standalone_runner_import_does_not_load_evaluator_heavy_modules():
    script = (
        "import sys; "
        "import experiments.window_reference_campaign.runner; "
        "assert not any(name.startswith(('evaluation.pointcloud', 'evaluation.trajectory', 'open3d')) "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_evaluator_does_not_infer_kind_from_paths_or_dataset(tmp_path):
    staged = SimpleNamespace(
        dataset=DatasetKind.SEVEN_SCENES,
        pointcloud_gt_path=tmp_path / "ground_truth.npz",
        poses_path=None,
    )

    with pytest.raises(ValueError, match="evaluation_kind"):
        campaign_evaluation._staged_evaluation_kind(staged)


def test_staged_scene_rejects_pointcloud_missing_or_mixed_fields(tmp_path):
    with pytest.raises(ValueError, match="requires GT"):
        _staged(
            tmp_path / "missing",
            kind=EvaluationKind.POINTCLOUD,
            dataset=DatasetKind.NRGBD,
        )
    with pytest.raises(ValueError, match="must not contain poses"):
        _staged(
            tmp_path / "mixed",
            kind=EvaluationKind.POINTCLOUD,
            dataset=DatasetKind.NRGBD,
            poses_path=tmp_path / "poses.txt",
            pointcloud_gt_path=tmp_path / "ground_truth.npz",
        )


def test_staged_scene_rejects_internal_trajectory_missing_or_mixed_fields(tmp_path):
    with pytest.raises(ValueError, match="requires poses"):
        _staged(
            tmp_path / "missing",
            kind=EvaluationKind.INTERNAL_TRAJECTORY,
            dataset=DatasetKind.KITTI,
        )
    with pytest.raises(ValueError, match="must not contain GT"):
        _staged(
            tmp_path / "mixed",
            kind=EvaluationKind.INTERNAL_TRAJECTORY,
            dataset=DatasetKind.KITTI,
            poses_path=tmp_path / "poses.txt",
            pointcloud_gt_path=tmp_path / "ground_truth.npz",
        )


def test_staging_manifest_round_trip_preserves_kind_and_detects_tampering(tmp_path):
    # This is intentionally a real staged directory round trip.  A tiny
    # synthetic scene keeps it independent of external datasets.
    image = tmp_path / "data" / "synthetic" / "000.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fixture")
    scene = ResolvedScene(
        scene_id="fixture",
        dataset=DatasetKind.SYNTHETIC,
        scene="fixture",
        slice_id="f000000-000001-s1",
        approved_data_root=tmp_path / "data",
        source_images=(image,),
        selection=ResolvedFrameSelection(0, 1, 1, (0,)),
        evaluation_kind=EvaluationKind.NONE,
        poses_path=None,
        prepared_gt_path=None,
        frame_index_map=None,
        expected_gt_shape=None,
    )
    staged = stage_scene(scene, tmp_path / "campaign")
    payload, _ = preview_staging_manifest(scene)
    loaded = load_valid_staging(staged.manifest_path, payload)
    assert loaded.evaluation_kind is EvaluationKind.NONE
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_kind"] = "pointcloud"
    staged.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="staging manifest mismatch"):
        load_valid_staging(staged.manifest_path, payload)
    with pytest.raises(ValueError, match="staging manifest mismatch"):
        stage_scene(scene, tmp_path / "campaign")
