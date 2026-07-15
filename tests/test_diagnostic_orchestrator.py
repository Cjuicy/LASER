import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_engine.diagnostics.orchestrator import (
    DIAGNOSTIC_PROFILES,
    checkpoint_sha256,
    dataset_fingerprint,
    preflight,
    run_master,
)
import inference_engine.diagnostics.orchestrator as orchestrator_module


def _dataset(root):
    for seq in ("00", "02"):
        images = root / "sequences" / seq / "image_2"
        images.mkdir(parents=True)
        for index in range(3):
            (images / f"{index:06d}.png").write_bytes(b"png" + bytes([index]))
        poses = root / "poses"; poses.mkdir(exist_ok=True)
        (poses / f"{seq}.txt").write_text(" ".join(["1"] * 12) + "\n")


def _args(tmp_path, **overrides):
    dataset = tmp_path / "dataset"; _dataset(dataset)
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"weights")
    values = dict(
        dataset_root=str(dataset), model_ckpt=str(checkpoint), output_dir=str(tmp_path / "output"),
        temp_root=str(tmp_path / "temp"), sequences=["00", "02"], window_size=3, overlap=1,
        top_conf_percentile=.3, seed=0, max_temp_gib=.001, warn_temp_gib=.0009,
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
