import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from inference_engine.diagnostics.orchestrator import (
    DIAGNOSTIC_PROFILES,
    _artifact_inventory,
    _git_commit,
    _validate_sequence_checkpoint,
    checkpoint_sha256,
    dataset_fingerprint,
    build_cases,
    preflight,
    run_master,
)
from inference_engine.diagnostics.schema import SelectedInterval
from inference_engine.diagnostics.storage import StorageLimitExceeded
import inference_engine.diagnostics.orchestrator as orchestrator_module


def _dataset(root):
    for seq in ("00", "02"):
        images = root / "sequences" / seq / "image_2"
        images.mkdir(parents=True)
        for index in range(3):
            cv2.imwrite(
                str(images / f"{index:06d}.png"),
                np.full((4, 6, 3), index, dtype=np.uint8),
            )
        poses = root / "poses"; poses.mkdir(exist_ok=True)
        (poses / f"{seq}.txt").write_text((" ".join(["1"] * 12) + "\n") * 3)


def _args(tmp_path, **overrides):
    dataset = tmp_path / "dataset"; _dataset(dataset)
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"weights")
    values = dict(
        dataset_root=str(dataset), model_ckpt=str(checkpoint), output_dir=str(tmp_path / "output"),
        temp_root=str(tmp_path / "temp"), sequences=["00", "02"], window_size=3, overlap=1,
        top_conf_percentile=.3, seed=0, max_temp_gib=10.0, warn_temp_gib=9.0,
        min_free_gib=0, max_selected=8, resume=False, report_only=False, dry_run=False,
        worker=False, pass_id=0, config_id=None, selected_intervals=None, device="cpu",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_profiles_are_fixed_and_legacy_is_not_official():
    assert list(DIAGNOSTIC_PROFILES) == ["depth", "geometry_baseline", "layer_atomic", "geometry_legacy_reference"]
    assert DIAGNOSTIC_PROFILES["geometry_legacy_reference"]["official"] is False
    assert all(DIAGNOSTIC_PROFILES[name]["official"] for name in ("depth", "geometry_baseline", "layer_atomic"))


def test_preflight_validates_layout_hashes_checkpoint_and_fingerprints_data(tmp_path):
    args = _args(tmp_path)
    result = preflight(args)
    assert result["checkpoint_sha256"] == checkpoint_sha256(args.model_ckpt)
    assert result["dataset_fingerprint"] == dataset_fingerprint(Path(args.dataset_root), args.sequences)
    assert result["frame_counts"] == {"00": 3, "02": 3}
    assert result["profiles"] == list(DIAGNOSTIC_PROFILES)
    assert result["loop_closure"] is False


def test_git_fingerprint_is_repository_scoped_not_caller_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _git_commit() != "unknown-local-commit"


def test_preflight_rejects_missing_pose_or_sequence(tmp_path):
    args = _args(tmp_path)
    (Path(args.dataset_root) / "poses" / "02.txt").unlink()
    with pytest.raises(FileNotFoundError, match="02"):
        preflight(args)


def test_master_runs_profiles_sequentially_and_stops_on_failure(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calls = []
    def fake_run(command, check):
        config = command[command.index("--config-id") + 1]
        pass_id = command[command.index("--pass-id") + 1]
        calls.append((pass_id, config))
        # Create minimal trajectory output expected after Pass 1.
        if pass_id == "1":
            for seq in args.sequences:
                path = Path(args.output_dir) / "trajectory" / config
                path.mkdir(parents=True, exist_ok=True)
                (path / f"{seq}.json").write_text(json.dumps({"valid": True, "ate_rmse": 1, "per_frame_translation_error": [0, 0, 0]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(orchestrator_module, "build_cases", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator_module, "build_report", lambda root: Path(root) / "report/index.html")
    monkeypatch.setattr(orchestrator_module, "_validate_sequence_checkpoint", lambda *a, **k: (True, "complete"))
    monkeypatch.setattr(orchestrator_module, "select_intervals", lambda *a, **k: [])
    assert run_master(args, runner=fake_run) == 0
    assert calls[:4] == [("1", name) for name in DIAGNOSTIC_PROFILES]
    assert all(item[0] == "2" for item in calls[4:])


def test_resume_refuses_changed_checkpoint(tmp_path):
    args = _args(tmp_path, dry_run=True)
    assert run_master(args) == 0
    manifest = Path(args.output_dir) / "manifest.json"
    data = json.loads(manifest.read_text()); data["checkpoint_sha256"] = "different"
    manifest.write_text(json.dumps(data))
    args.resume = True
    with pytest.raises(ValueError, match="checkpoint"):
        run_master(args)


def test_dry_run_rejects_projected_two_pass_storage_before_manifest(tmp_path):
    args = _args(
        tmp_path, dry_run=True, max_temp_gib=.001, warn_temp_gib=.0009,
        max_selected=48,
    )
    with pytest.raises(StorageLimitExceeded):
        run_master(args)
    assert not (Path(args.output_dir) / "manifest.json").exists()


def test_report_only_does_not_replace_manifest_or_invoke_preflight(tmp_path, monkeypatch):
    args = _args(tmp_path, report_only=True)
    output = Path(args.output_dir); output.mkdir(parents=True)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"run_id": "existing", "status": "complete"}))
    before = manifest.read_bytes()
    monkeypatch.setattr(orchestrator_module, "build_report", lambda root: Path(root) / "report/index.html")
    Path(args.model_ckpt).unlink()
    assert run_master(args) == 0
    assert manifest.read_bytes() == before


def test_sequence_checkpoint_detects_missing_or_tampered_artifact(tmp_path):
    output = tmp_path / "output"
    trajectory = output / "trajectory" / "depth"
    shard = output / "artifacts" / "depth" / "00" / "pass1"
    trajectory.mkdir(parents=True); shard.mkdir(parents=True)
    (trajectory / "00.json").write_text("{}")
    np.savez_compressed(trajectory / "00.npz", predicted=np.zeros((2, 4, 4)))
    (shard / "segmentation.jsonl").write_text("{}\n")
    (shard / "temporal.jsonl").write_text("{}\n")
    inventory = _artifact_inventory(output, [trajectory / "00.json", trajectory / "00.npz", shard])
    checkpoint = output / "checkpoints" / "pass1" / "depth" / "00.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({
        "run_id": "run", "config_id": "depth", "sequence_id": "00",
        "pass_id": 1, "checkpoint_sha256": "weights", "frame_count": 2,
        "pose_count": 2, "implementation_fingerprint": "commit",
        "config_hash": "config", "dataset_fingerprint": "dataset",
        "artifacts": inventory,
    }))
    kwargs = dict(
        run_id="run", config_id="depth", sequence_id="00", pass_id=1,
        checkpoint_hash="weights", expected_frames=2, window_size=3,
        implementation_fingerprint="commit", config_hash="config", dataset_hash="dataset",
    )
    assert _validate_sequence_checkpoint(output, checkpoint, **kwargs) == (True, "complete")
    payload = json.loads(checkpoint.read_text())
    payload["frame_count"] = payload["pose_count"] = 4
    checkpoint.write_text(json.dumps(payload))
    large_kwargs = {**kwargs, "expected_frames": 4}
    valid, reason = _validate_sequence_checkpoint(output, checkpoint, **large_kwargs)
    assert valid is False and "scale.jsonl" in reason
    payload["frame_count"] = payload["pose_count"] = 2
    checkpoint.write_text(json.dumps(payload))
    (shard / "segmentation.jsonl").write_text("tampered\n")
    valid, reason = _validate_sequence_checkpoint(output, checkpoint, **kwargs)
    assert valid is False
    assert "artifact_" in reason


def test_build_cases_requires_all_methods_and_namespaces_complete_trace(tmp_path):
    interval = SelectedInterval("02", 0, 2, ("trajectory",), 3.0)
    height, width = 3, 4
    y, x = np.mgrid[:height, :width]
    labels = (x > 1).astype(np.int32)
    for config in DIAGNOSTIC_PROFILES:
        trace = tmp_path / "artifacts" / config / "02" / "pass2" / "traces"
        trace.mkdir(parents=True)
        np.savez_compressed(
            trace / "inputs-frame-000001.npz",
            rgb=np.zeros((height, width, 3)),
            point_map=np.stack([x, y, np.ones_like(x)], axis=-1),
            confidence=np.ones((height, width)),
        )
        np.savez_compressed(
            trace / "segmentation-frame-000001.npz",
            initial_labels=labels, coarse_labels=labels, final_labels=labels,
            normalized_gap_map=np.ones((height, width)),
        )
        np.savez_compressed(
            trace / "scale-window-000000.npz",
            scale_map=np.ones((3, height, width)),
            source_map=np.ones((3, height, width), np.uint8),
            dispersion_map=np.zeros((3, height, width)),
        )
        np.savez_compressed(
            trace / "temporal-window-000000.npz",
            temporal_best_iou_map=np.ones((3, height, width)),
        )
    records = [{
        "config_id": config, "sequence_id": "02", "frame_start": 0,
        "frame_end": 2, "trajectory_regret": 1.25,
    } for config in DIAGNOSTIC_PROFILES]
    args = argparse.Namespace(window_size=3, overlap=1)
    build_cases(tmp_path, [interval], records, args)
    root = tmp_path / "cases" / "02" / "000000-000002"
    with np.load(root / "trace.npz") as data:
        assert "layer_atomic__segmentation__normalized_gap_map" in data
        assert "geometry_baseline__temporal__temporal_best_iou_map" in data
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["selection_score"] == 3.0
    assert metrics["trajectory_regret"] == 1.25
    assert (root / "artifact-manifest.json").is_file()
