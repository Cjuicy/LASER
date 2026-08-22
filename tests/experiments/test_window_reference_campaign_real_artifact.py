from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from experiments.window_reference_campaign.config import (
    CampaignOverrides,
    load_campaign_config,
)
from experiments.window_reference_campaign.matrix import build_identity_seed, build_plan
from experiments.window_reference_campaign.runner import validate_artifact_for_seed
from inference_engine.prediction_cache.fingerprint import sha256_file
from pipeline.artifacts import (
    ReconstructionArtifact,
    ReconstructionDiagnostics,
    write_reconstruction_artifact,
)
from pipeline.config import ReconstructionMode, SegmentationMethod, load_pipeline_config


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_CONFIG = ROOT / "configs/experiments/window_reference_campaign.yaml"


def _build_real_artifact(tmp_path):
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
    return artifact_dir, seed, artifact, loaded_pipeline.resolved_yaml


def _rewrite_resolved_config(artifact_dir, updates):
    resolved_path = artifact_dir / "resolved_reconstruction.yaml"
    config = OmegaConf.load(resolved_path)
    for path, value in updates.items():
        OmegaConf.update(config, path, value, merge=False)
    resolved_yaml = OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
    resolved_path.write_text(resolved_yaml, encoding="utf-8")
    resolved_digest = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = resolved_digest
    manifest["resolved_yaml_sha256"] = resolved_digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_real_artifact_validator_accepts_canonical_pipeline_enum_names(tmp_path):
    artifact_dir, seed, artifact, resolved_yaml = _build_real_artifact(tmp_path)
    assert "\n  name: PI3\n" in resolved_yaml
    assert "\n  method: DEPTH\n" in resolved_yaml
    assert "\n    split_mode: CONSERVATIVE\n" in resolved_yaml
    assert "\n  mode: NO_LOOP\n" in resolved_yaml

    assert validate_artifact_for_seed(artifact_dir, seed) == (
        artifact.prediction_key,
        sha256_file(artifact_dir / "manifest.json"),
    )


def test_real_artifact_validator_accepts_lowercase_pipeline_enum_values(tmp_path):
    artifact_dir, seed, artifact, _ = _build_real_artifact(tmp_path)
    _rewrite_resolved_config(
        artifact_dir,
        {
            "model.name": "pi3",
            "segmentation.method": "depth",
            "segmentation.atomic.split_mode": "conservative",
            "reconstruction.mode": "no_loop",
        },
    )

    assert validate_artifact_for_seed(artifact_dir, seed) == (
        artifact.prediction_key,
        sha256_file(artifact_dir / "manifest.json"),
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("model.name", "not-a-model"),
        ("model.name", "DEPTH"),
        ("segmentation.method", "not-a-method"),
        ("segmentation.method", "GEOMETRY"),
        ("segmentation.atomic.split_mode", "not-a-split-mode"),
        ("segmentation.atomic.split_mode", "NONE"),
        ("reconstruction.mode", "not-a-mode"),
        ("reconstruction.mode", "TRADITIONAL"),
    ),
)
def test_real_artifact_validator_rejects_malformed_or_wrong_enum_values(
    tmp_path, path, value
):
    artifact_dir, seed, _, _ = _build_real_artifact(tmp_path)
    _rewrite_resolved_config(artifact_dir, {path: value})

    with pytest.raises(ValueError) as error:
        validate_artifact_for_seed(artifact_dir, seed)
    assert str(error.value) == f"resolved config {path} does not match identity"
