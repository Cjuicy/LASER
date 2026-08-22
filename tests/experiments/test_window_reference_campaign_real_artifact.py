from __future__ import annotations

from pathlib import Path

import torch

from experiments.window_reference_campaign.config import (
    CampaignOverrides,
    load_campaign_config,
)
from experiments.window_reference_campaign.matrix import build_identity_seed, build_plan
from experiments.window_reference_campaign.runner import validate_artifact_for_seed
from inference_engine.prediction_cache.fingerprint import sha256_file
from pipeline.artifacts import ReconstructionArtifact, ReconstructionDiagnostics, write_reconstruction_artifact
from pipeline.config import ReconstructionMode, SegmentationMethod, load_pipeline_config


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_CONFIG = ROOT / "configs/experiments/window_reference_campaign.yaml"


def test_real_artifact_validator_accepts_canonical_pipeline_enum_names(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"real-artifact-validator-checkpoint")
    loaded_campaign = load_campaign_config(
        CAMPAIGN_CONFIG,
        CampaignOverrides(
            preset="kitti-smoke",
            data_root=tmp_path / "data",
            output_root=tmp_path / "outputs",
            checkpoint=checkpoint,
        ),
    )
    planned = build_plan(loaded_campaign).runs[0]
    seed = build_identity_seed(
        loaded=loaded_campaign,
        planned=planned,
        frame_start=0,
        frame_stop=2,
        frame_stride=1,
        staged_manifest_sha256="a" * 64,
        source_commit="b" * 40,
        source_dirty=False,
        checkpoint_sha256=sha256_file(checkpoint),
    )

    loaded_pipeline = load_pipeline_config(
        loaded_campaign.config.pipeline_config,
        (
            "input.sample_stride=1",
            f"model.checkpoint={checkpoint}",
            "window.size=75",
            "window.overlap=30",
            "segmentation.method=depth",
            "segmentation.atomic.split_mode=conservative",
            "segmentation.window_reference.enabled=false",
            "reconstruction.mode=no_loop",
        ),
    )
    assert "\n  name: PI3\n" in loaded_pipeline.resolved_yaml
    assert "\n  method: DEPTH\n" in loaded_pipeline.resolved_yaml
    assert "\n    split_mode: CONSERVATIVE\n" in loaded_pipeline.resolved_yaml
    assert "\n  mode: NO_LOOP\n" in loaded_pipeline.resolved_yaml

    frame_ids = (0, 1)
    points = torch.zeros((2, 1, 1, 3), dtype=torch.float32)
    artifact = ReconstructionArtifact(
        schema_version=1,
        frame_ids=frame_ids,
        local_points=points,
        global_points=points,
        camera_poses=torch.eye(4, dtype=torch.float32).repeat(2, 1, 1),
        confidence=torch.ones((2, 1, 1), dtype=torch.float32),
        segmentation_method=SegmentationMethod.DEPTH,
        reconstruction_mode=ReconstructionMode.NO_LOOP,
        prediction_key="c" * 64,
        diagnostics=ReconstructionDiagnostics(
            stage_timings_ms={"reconstruction": 1.0},
            segmentation_summaries=({}, {}),
            candidate_count=0,
            constraint_count=0,
            mode_scalars={"window_count": 1},
        ),
    )
    artifact_dir = write_reconstruction_artifact(
        artifact,
        tmp_path / "artifact",
        resolved_yaml=loaded_pipeline.resolved_yaml,
        config_sha256=loaded_pipeline.sha256,
        checkpoint_sha256=seed.checkpoint_sha256,
        git_commit=seed.source_commit,
    )

    assert validate_artifact_for_seed(artifact_dir, seed) == (
        artifact.prediction_key,
        sha256_file(artifact_dir / "manifest.json"),
    )
