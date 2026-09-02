from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from pipeline.artifacts import (
    PointMapEstimate,
    ReconstructionArtifact,
    ReconstructionDiagnostics,
    StagedReconstructionArtifacts,
    TrajectoryEstimate,
    load_pointmap_estimate,
    load_reconstruction_artifact,
    load_trajectory_estimate,
    write_reconstruction_artifact,
    write_staged_reconstruction_artifacts,
)
from pipeline.config import ReconstructionMode, SegmentationMethod


def make_artifact(**overrides) -> ReconstructionArtifact:
    local_points = torch.arange(27, dtype=torch.float32).reshape(3, 3, 1, 3)
    values = {
        "schema_version": 1,
        "frame_ids": (0, 1, 2),
        "local_points": local_points,
        "global_points": local_points + 1.0,
        "camera_poses": torch.eye(4).repeat(3, 1, 1),
        "confidence": torch.ones((3, 3, 1)),
        "segmentation_method": SegmentationMethod.DEPTH,
        "reconstruction_mode": ReconstructionMode.NO_LOOP,
        "prediction_key": "prediction-key",
        "diagnostics": ReconstructionDiagnostics(
            stage_timings_ms={"prediction": 1.25},
            segmentation_summaries=({"region_count": 2},),
            candidate_count=0,
            constraint_count=0,
            mode_scalars={"window_count": 2},
        ),
    }
    values.update(overrides)
    return ReconstructionArtifact(**values)


def assert_artifacts_equal(actual, expected):
    assert actual.schema_version == expected.schema_version
    assert actual.frame_ids == expected.frame_ids
    assert actual.segmentation_method is expected.segmentation_method
    assert actual.reconstruction_mode is expected.reconstruction_mode
    assert actual.prediction_key == expected.prediction_key
    assert actual.diagnostics == expected.diagnostics
    for name in (
        "local_points",
        "global_points",
        "camera_poses",
        "confidence",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name))


def make_staged_artifacts() -> StagedReconstructionArtifacts:
    return StagedReconstructionArtifacts(
        stage1=make_artifact(
            reconstruction_mode=ReconstructionMode.TRADITIONAL,
        ),
        stage2=make_artifact(
            reconstruction_mode=(
                ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
            ),
        ),
    )


def test_artifact_requires_one_consistent_finite_frame_axis():
    with pytest.raises(ValueError, match="frame count"):
        make_artifact(camera_poses=torch.eye(4).repeat(2, 1, 1))
    nonfinite = make_artifact().global_points.clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="global_points.*finite"):
        make_artifact(global_points=nonfinite)


def test_staged_result_requires_exact_stage_modes():
    result = StagedReconstructionArtifacts(
        stage1=make_artifact(
            reconstruction_mode=ReconstructionMode.TRADITIONAL,
        ),
        stage2=make_artifact(
            reconstruction_mode=(
                ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
            ),
        ),
    )

    assert result.primary is result.stage2

    with pytest.raises(ValueError, match="stage1.*traditional"):
        StagedReconstructionArtifacts(
            stage1=make_artifact(
                reconstruction_mode=ReconstructionMode.NO_LOOP,
            ),
            stage2=result.stage2,
        )


def test_staged_writer_atomically_writes_two_standard_artifacts(tmp_path):
    staged = make_staged_artifacts()
    output = tmp_path / "run"

    paths = write_staged_reconstruction_artifacts(
        staged,
        output,
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )

    assert dict(paths) == {
        "stage1": output / "stage1",
        "stage2": output,
    }
    assert_artifacts_equal(
        load_reconstruction_artifact(paths["stage1"]),
        staged.stage1,
    )
    assert_artifacts_equal(
        load_reconstruction_artifact(paths["stage2"]),
        staged.stage2,
    )


def test_staged_writer_leaves_no_destination_when_nested_write_fails(
    tmp_path,
    monkeypatch,
):
    import pipeline.artifacts as artifacts_module

    output = tmp_path / "run"
    real_writer = artifacts_module.write_reconstruction_artifact
    call_count = 0

    def fail_stage1(*arguments, **keywords):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected Stage 1 write failure")
        return real_writer(*arguments, **keywords)

    monkeypatch.setattr(
        artifacts_module,
        "write_reconstruction_artifact",
        fail_stage1,
    )

    with pytest.raises(RuntimeError, match="Stage 1 write failure"):
        write_staged_reconstruction_artifacts(
            make_staged_artifacts(),
            output,
            resolved_yaml="version: 2\n",
            config_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            git_commit="c" * 40,
        )

    assert not output.exists()


def test_artifact_views_expose_only_evaluator_fields():
    artifact = make_artifact()
    trajectory = artifact.trajectory
    pointmap = artifact.pointmap

    assert isinstance(trajectory, TrajectoryEstimate)
    assert trajectory.frame_ids == (0, 1, 2)
    assert torch.equal(trajectory.camera_poses, artifact.camera_poses)
    assert isinstance(pointmap, PointMapEstimate)
    assert pointmap.reconstruction_mode is ReconstructionMode.NO_LOOP
    assert torch.equal(pointmap.global_points, artifact.global_points)
    assert not hasattr(trajectory, "global_points")
    assert not hasattr(pointmap, "camera_poses")


def test_artifact_round_trip_validates_tensor_digest(tmp_path):
    artifact = make_artifact()
    output = write_reconstruction_artifact(
        artifact,
        tmp_path / "run",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    assert_artifacts_equal(load_reconstruction_artifact(output), artifact)

    (output / "pointmap.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="pointmap.pt digest mismatch"):
        load_pointmap_estimate(output)


def test_enabled_segmentation_diagnostics_round_trip_preserves_artifact_contract(
    tmp_path,
):
    summary = {
        "window_reference_applied": True,
        "window_reference_keyframes": "0,2",
        "window_reference_keyframe_count": 2,
        "window_reference_is_keyframe": False,
        "window_reference_coverage_ratio": 0.75,
        "window_reference_regions_before": 5,
        "window_reference_regions_after": 3,
        "window_reference_candidate_edges": 4,
        "window_reference_accepted_edges": 2,
        "window_reference_conflict_edges": 1,
        "window_reference_projected_samples": 32,
        "window_reference_occluded_samples": 3,
        "window_reference_depth_rejected_samples": 1,
        "window_reference_fallback": "none",
    }
    diagnostics = ReconstructionDiagnostics(
        stage_timings_ms={"segmentation": 4.5},
        segmentation_summaries=(summary,),
        candidate_count=2,
        constraint_count=1,
        mode_scalars={"window_count": 3, "coverage_ratio": 0.75},
    )
    artifact = make_artifact(diagnostics=diagnostics)
    output = write_reconstruction_artifact(
        artifact,
        tmp_path / "enabled-window-reference",
        resolved_yaml="segmentation:\n  window_reference:\n    enabled: true\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )

    assert_artifacts_equal(load_reconstruction_artifact(output), artifact)
    assert artifact.schema_version == 1
    assert load_reconstruction_artifact(output).schema_version == 1
    assert all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in artifact.diagnostics.mode_scalars.values()
    )
    assert {
        path.name
        for path in Path(output).iterdir()
    } == {
        "trajectory.pt",
        "pointmap.pt",
        "confidence.pt",
        "resolved_reconstruction.yaml",
        "diagnostics.json",
        "manifest.json",
    }
    assert not any("segmentation" in path.name for path in Path(output).iterdir())


def test_narrow_loaders_do_not_read_unrelated_tensors(tmp_path):
    output = write_reconstruction_artifact(
        make_artifact(),
        tmp_path / "run",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    (output / "pointmap.pt").write_bytes(b"trajectory loader must ignore")
    trajectory = load_trajectory_estimate(output)
    assert trajectory.frame_ids == (0, 1, 2)

    output = write_reconstruction_artifact(
        make_artifact(),
        tmp_path / "run-two",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    (output / "trajectory.pt").write_bytes(b"pointmap loader must ignore")
    pointmap = load_pointmap_estimate(output)
    assert pointmap.frame_ids == (0, 1, 2)


@pytest.mark.parametrize(
    "frame_ids",
    ((0, 0, 2), (-1, 0, 1), (0, 2)),
)
def test_artifact_rejects_invalid_frame_ids(frame_ids):
    with pytest.raises(ValueError, match="frame_ids|frame count"):
        make_artifact(frame_ids=frame_ids)


def test_writer_refuses_to_overwrite_completed_artifact(tmp_path):
    output = tmp_path / "run"
    arguments = {
        "resolved_yaml": "version: 2\n",
        "config_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }
    write_reconstruction_artifact(make_artifact(), output, **arguments)
    with pytest.raises(FileExistsError, match="already exists"):
        write_reconstruction_artifact(make_artifact(), output, **arguments)


def test_artifact_manifest_records_provenance_and_tensor_shapes(tmp_path):
    output = write_reconstruction_artifact(
        make_artifact(),
        tmp_path / "run",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    manifest = json.loads((Path(output) / "manifest.json").read_text())
    assert manifest["config_sha256"] == "a" * 64
    assert manifest["checkpoint_sha256"] == "b" * 64
    assert manifest["git_commit"] == "c" * 40
    assert manifest["tensors"]["pointmap.pt"]["global_points"]["shape"] == [
        3,
        3,
        1,
        3,
    ]


def test_diagnostics_reject_live_tensors_and_runtime_objects():
    with pytest.raises(ValueError, match="JSON scalar"):
        ReconstructionDiagnostics(
            stage_timings_ms={},
            segmentation_summaries=({"labels": torch.ones(1)},),
            candidate_count=0,
            constraint_count=0,
            mode_scalars={},
        )


def test_loader_rejects_manifest_tensor_metadata_mismatch(tmp_path):
    output = write_reconstruction_artifact(
        make_artifact(),
        tmp_path / "run",
        resolved_yaml="version: 2\n",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        git_commit="c" * 40,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensors"]["trajectory.pt"]["camera_poses"]["shape"] = [2, 4, 4]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="camera_poses metadata mismatch"):
        load_trajectory_estimate(output)
