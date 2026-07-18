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
    _experiment_contract,
    _build_split_structural_summary,
    _manifest,
    _persist_regret_artifacts,
    _validate_case_artifacts,
    _validate_sequence_checkpoint,
    checkpoint_sha256,
    dataset_fingerprint,
    build_cases,
    build_selection_records,
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


def test_profiles_are_the_strict_three_method_contract():
    assert list(DIAGNOSTIC_PROFILES) == [
        "depth",
        "geometry_baseline",
        "layer_atomic_split",
    ]
    assert DIAGNOSTIC_PROFILES["layer_atomic_split"] == {
        "segment_mode": "layer_atomic_split",
        "geometry_seg_profile": "baseline_params",
        "normal_method": "cross",
        "split_score_thresh": 0.10,
        "split_aux_confirmation": True,
        "official": True,
    }


def test_preflight_validates_layout_hashes_checkpoint_and_fingerprints_data(tmp_path):
    args = _args(tmp_path)
    result = preflight(args)
    assert result["checkpoint_sha256"] == checkpoint_sha256(args.model_ckpt)
    assert result["dataset_fingerprint"] == dataset_fingerprint(Path(args.dataset_root), args.sequences)
    assert result["frame_counts"] == {"00": 3, "02": 3}
    assert result["profiles"] == list(DIAGNOSTIC_PROFILES)
    assert result["loop_closure"] is False


def test_manifest_stores_exact_ordered_effective_experiment_contract(tmp_path):
    args = _args(tmp_path)
    info = preflight(args)
    contract = _experiment_contract(args)

    assert contract == {
        "profiles": [
            {
                "config_id": "depth",
                "effective_parameters": {
                    "segment_mode": "depth",
                    "geometry_seg_profile": "baseline_params",
                    "seg_scale": 300,
                    "seg_sigma": 1.1,
                    "seg_min_size": 500,
                    "top_conf_percentile": 0.3,
                    "official": True,
                },
            },
            {
                "config_id": "geometry_baseline",
                "effective_parameters": {
                    "segment_mode": "geometry",
                    "geometry_seg_profile": "baseline_params",
                    "normal_method": "cross",
                    "seg_scale": 300,
                    "seg_sigma": 1.1,
                    "seg_min_size": 500,
                    "top_conf_percentile": 0.3,
                    "official": True,
                },
            },
            {
                "config_id": "layer_atomic_split",
                "effective_parameters": {
                    "segment_mode": "layer_atomic_split",
                    "geometry_seg_profile": "baseline_params",
                    "normal_method": "cross",
                    "split_score_thresh": 0.10,
                    "split_aux_confirmation": True,
                    "seg_scale": 300,
                    "seg_sigma": 1.1,
                    "seg_min_size": 500,
                    "top_conf_percentile": 0.3,
                    "official": True,
                },
            },
        ],
        "sequences": ["00", "02"],
        "window_size": 3,
        "overlap": 1,
        "evaluation_signature": {
            "ape_relation": "translation_part",
            "align": True,
            "correct_scale": True,
            "rpe_translation_relation": "translation_part",
            "rpe_rotation_relation": "rotation_angle_deg",
            "rpe_delta": 1,
            "delta_unit": "frames",
            "all_pairs": True,
        },
    }
    manifest = _manifest(args, info, "run")
    assert manifest.experiment_contract == contract
    assert manifest.to_dict()["experiment_contract"] == contract


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
    monkeypatch.setattr(
        orchestrator_module,
        "select_intervals_with_coverage",
        lambda *a, **k: ([], {
            reason: {
                "available": False,
                "selected": False,
                "reason": "no_qualifying_window",
                "qualifying_window_count": 0,
            }
            for reason in (
                "trajectory_degradation", "trajectory_improvement",
                "trajectory_change", "split_anomaly",
                "split_no_trajectory_effect", "matched_control",
            )
        }),
    )
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


def test_dry_run_reports_the_registered_profile_count(tmp_path, capsys):
    args = _args(tmp_path, dry_run=True)

    assert run_master(args) == 0

    assert (
        f"[phase preflight] {len(DIAGNOSTIC_PROFILES)} sequential profiles"
        in capsys.readouterr().out
    )


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


def _pass2_checkpoint_fixture(tmp_path, *, omit_frame=None, split_overrides=None):
    output = tmp_path / "output"
    config = "layer_atomic_split"
    trajectory = output / "trajectory" / "pass2" / config
    shard = output / "artifacts" / config / "00" / "pass2"
    traces = shard / "traces"
    trajectory.mkdir(parents=True)
    traces.mkdir(parents=True)
    (trajectory / "00.json").write_text("{}")
    np.savez_compressed(trajectory / "00.npz", predicted=np.zeros((3, 4, 4)))
    (shard / "segmentation.jsonl").write_text("{}\n")
    (shard / "temporal.jsonl").write_text("{}\n")
    labels = np.zeros((3, 4), dtype=np.int32)
    for frame in range(3):
        if frame == omit_frame:
            continue
        np.savez_compressed(
            traces / f"inputs-frame-{frame:06d}.npz",
            rgb=np.zeros((3, 4, 3)),
            point_map=np.zeros((3, 4, 3)),
            confidence=np.ones((3, 4)),
        )
        arrays = {
            "pre_split_labels": labels,
            "final_labels": labels,
            "atom_labels": labels,
            "atom_scales": np.ones(1),
            "changed_mask__packed": np.packbits(np.zeros(labels.size, dtype=bool)),
            "changed_mask__shape": np.asarray(labels.shape),
            "split_parent_map": labels,
            "split_child_map": labels,
            "split_score_map": np.zeros(labels.shape),
            "split_decision_map": np.zeros(labels.shape, dtype=np.uint8),
        }
        if split_overrides:
            arrays.update(split_overrides)
        np.savez_compressed(
            traces / f"segmentation-frame-{frame:06d}.npz", **arrays
        )
    inventory = _artifact_inventory(
        output, [trajectory / "00.json", trajectory / "00.npz", shard]
    )
    checkpoint = output / "checkpoints" / "pass2" / config / "00.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({
        "run_id": "run", "config_id": config, "sequence_id": "00",
        "pass_id": 2, "checkpoint_sha256": "weights", "frame_count": 3,
        "pose_count": 3, "implementation_fingerprint": "commit",
        "config_hash": "config", "dataset_fingerprint": "dataset",
        "artifacts": inventory,
    }))
    return output, checkpoint


def _validate_pass2_fixture(output, checkpoint):
    return _validate_sequence_checkpoint(
        output,
        checkpoint,
        run_id="run",
        config_id="layer_atomic_split",
        sequence_id="00",
        pass_id=2,
        checkpoint_hash="weights",
        expected_frames=3,
        window_size=3,
        selected_intervals=[
            SelectedInterval("00", 0, 2, ("split_anomaly",), 1.0)
        ],
        implementation_fingerprint="commit",
        config_hash="config",
        dataset_hash="dataset",
    )


def test_pass2_checkpoint_validates_every_selected_context_frame(tmp_path):
    output, checkpoint = _pass2_checkpoint_fixture(tmp_path, omit_frame=0)

    valid, reason = _validate_pass2_fixture(output, checkpoint)

    assert valid is False
    assert reason.endswith("inputs-frame-000000.npz")


def test_pass2_split_checkpoint_requires_all_dense_split_keys(tmp_path):
    output, checkpoint = _pass2_checkpoint_fixture(tmp_path)
    # np.savez represents None as an object scalar; remove the key explicitly.
    trace = (
        output / "artifacts" / "layer_atomic_split" / "00" / "pass2"
        / "traces" / "segmentation-frame-000001.npz"
    )
    with np.load(trace) as data:
        arrays = {name: data[name] for name in data.files if name != "split_parent_map"}
    np.savez_compressed(trace, **arrays)
    payload = json.loads(checkpoint.read_text())
    payload["artifacts"] = _artifact_inventory(
        output,
        [
            output / "trajectory" / "pass2" / "layer_atomic_split" / "00.json",
            output / "trajectory" / "pass2" / "layer_atomic_split" / "00.npz",
            output / "artifacts" / "layer_atomic_split" / "00" / "pass2",
        ],
    )
    checkpoint.write_text(json.dumps(payload))

    valid, reason = _validate_pass2_fixture(output, checkpoint)

    assert valid is False
    assert reason == "missing_split_evidence:000001:split_parent_map"


def test_pass2_split_checkpoint_rejects_dense_shape_mismatch(tmp_path):
    output, checkpoint = _pass2_checkpoint_fixture(
        tmp_path, split_overrides={"split_score_map": np.zeros((2, 4))}
    )

    valid, reason = _validate_pass2_fixture(output, checkpoint)

    assert valid is False
    assert reason == "split_evidence_shape_mismatch:000000:split_score_map"


def test_selection_records_expose_two_regrets_and_only_split_profile_activity(tmp_path):
    args = SimpleNamespace(
        sequences=["01"], window_size=2, overlap=0,
    )
    trajectory_results = {
        "depth": {"01": {"per_frame_translation_error": [1.0, 1.0, 2.0, 2.0]}},
        "geometry_baseline": {"01": {"per_frame_translation_error": [2.0, 2.0, 4.0, 4.0]}},
        "layer_atomic_split": {"01": {"per_frame_translation_error": [3.0, 5.0, 8.0, 10.0]}},
    }
    trajectory = tmp_path / "trajectory" / "layer_atomic_split"
    trajectory.mkdir(parents=True)
    ground_truth = np.repeat(np.eye(4)[None], 4, axis=0)
    np.savez_compressed(trajectory / "01.npz", ground_truth=ground_truth)

    for config in DIAGNOSTIC_PROFILES:
        artifact = tmp_path / "artifacts" / config / "01" / "pass1"
        artifact.mkdir(parents=True)
        split_values = (
            [(1, 0.1), (0, 0.3), (2, 0.4), (1, 0.6)]
            if config == "layer_atomic_split"
            else [(99, 0.99)] * 4
        )
        rows = []
        for frame, (accepted, changed) in enumerate(split_values):
            rows.append(json.dumps({
                "context": {"window_id": frame // 2},
                "metrics": {
                    "confidence_mean": 0.8,
                    "initial": {},
                    "final": {},
                    "merge": {},
                    "split": {
                        "split_accepted_count": accepted,
                        "split_changed_pixel_ratio": changed,
                    },
                },
            }))
        (artifact / "segmentation.jsonl").write_text("\n".join(rows) + "\n")

    records = build_selection_records(tmp_path, trajectory_results, args)
    record = next(
        row for row in records
        if row["config_id"] == "layer_atomic_split" and row["window_id"] == 0
    )
    assert record["split_minus_depth_regret"] == pytest.approx(3.0)
    assert record["split_minus_geometry_regret"] == pytest.approx(2.0)
    assert record["split_accepted_count"] == 1
    assert record["split_changed_pixel_ratio"] == pytest.approx(0.2)
    assert record["split_minus_depth_regret_max"] == 4.0
    assert record["split_minus_depth_regret_positive_area"] == 6.0
    assert record["split_minus_depth_regret_positive_duration"] == 2
    assert record["split_minus_depth_regret_positive_persistence"] == 2
    assert record["split_minus_depth_regret_change_points"] == [
        {"frame_index": 1, "delta": 2.0, "magnitude": 2.0},
    ]
    assert "trajectory_regret" not in record
    assert "layer_atomic_split_minus_depth_regret" not in record


def test_regret_artifacts_persist_both_arrays_and_truthful_metadata(tmp_path):
    trajectory_results = {
        "depth": {"01": {"per_frame_translation_error": [1.0, 2.0, 3.0]}},
        "geometry_baseline": {
            "01": {"per_frame_translation_error": [0.5, np.nan, 2.0]},
        },
        "layer_atomic_split": {
            "01": {"per_frame_translation_error": [2.0, 4.0, 1.0]},
        },
    }

    metadata = _persist_regret_artifacts(tmp_path, trajectory_results, ["01"])

    assert metadata["01"]["split_minus_depth_regret"]["values"] == [1.0, 2.0, -2.0]
    assert metadata["01"]["split_minus_geometry_regret"]["values"] == [1.5, None, -1.0]
    assert metadata["01"]["split_minus_geometry_regret"]["invalid_reason"] == "partial_missing_comparison"
    with np.load(tmp_path / "trajectory" / "regret" / "01.npz") as arrays:
        np.testing.assert_allclose(
            arrays["split_minus_depth_regret"], [1.0, 2.0, -2.0]
        )
        np.testing.assert_allclose(
            arrays["split_minus_geometry_regret"], [1.5, np.nan, -1.0],
            equal_nan=True,
        )
    persisted = json.loads(
        (tmp_path / "trajectory" / "regret" / "01.json").read_text()
    )
    assert persisted == metadata["01"]


def test_split_structural_summary_preserves_reasoned_nulls_but_rejects_silent_missing():
    record = {
        "split_accepted_count": 0,
        "split_score_mean": None,
        "split_score_invalid_reason": "no_scored_candidates",
        "split_changed_pixel_ratio": 0.0,
        "split_pre_segment_count": 2.0,
        "split_post_segment_count": 2.0,
        "split_segment_count_delta": 0.0,
        "split_child_count_mean": None,
        "split_child_summary_invalid_reason": "no_valid_child_proposals",
        "split_min_child_fraction_min": None,
        "split_parent_normal_dispersion_mean": None,
        "split_child_normal_dispersion_mean": None,
        "split_normal_dispersion_gain_mean": None,
        "split_normal_dispersion_invalid_reason": "no_valid_child_proposals",
        "split_largest_segment_ratio_delta": 0.0,
        "split_area_entropy_delta": 0.0,
        "split_effective_segment_count_delta": 0.0,
        "split_tiny_child_area_ratio": 0.0,
        "split_boundary_ratio_delta": 0.0,
        "split_fragmentation_signal": False,
    }

    summary = _build_split_structural_summary(record)

    assert summary["score_mean"] is None
    assert summary["score_invalid_reason"] == "no_scored_candidates"
    assert summary["normal_dispersion_gain_mean"] is None
    assert summary["normal_dispersion_invalid_reason"] == "no_valid_child_proposals"
    assert summary["fragmentation_signal"] is False

    record["split_score_invalid_reason"] = None
    with pytest.raises(RuntimeError, match="score_mean"):
        _build_split_structural_summary(record)


def test_selection_records_preserve_missing_split_evidence(tmp_path):
    args = SimpleNamespace(sequences=["01"], window_size=1, overlap=0)
    trajectory_results = {
        "depth": {"01": {"per_frame_translation_error": []}},
        "geometry_baseline": {"01": {"per_frame_translation_error": []}},
        "layer_atomic_split": {"01": {"per_frame_translation_error": [1.0]}},
    }
    artifact = tmp_path / "artifacts" / "layer_atomic_split" / "01" / "pass1"
    artifact.mkdir(parents=True)
    (artifact / "segmentation.jsonl").write_text(json.dumps({
        "context": {"window_id": 0},
        "metrics": {"confidence_mean": 0.8, "split": {}},
    }) + "\n")

    records = build_selection_records(tmp_path, trajectory_results, args)
    record = next(
        row for row in records if row["config_id"] == "layer_atomic_split"
    )

    assert record["split_accepted_count"] is None
    assert record["split_changed_pixel_ratio"] is None
    assert record["split_minus_depth_regret"] is None
    assert record["split_minus_geometry_regret"] is None


def test_build_cases_requires_all_methods_and_namespaces_complete_trace(tmp_path):
    interval = SelectedInterval("02", 0, 2, ("trajectory_degradation",), 3.0)
    height, width = 3, 4
    y, x = np.mgrid[:height, :width]
    labels = (x > 1).astype(np.int32)
    trajectory_errors = {
        "depth": [1.0, 2.0, 3.0],
        "geometry_baseline": [0.5, 1.5, 2.5],
        "layer_atomic_split": [1.5, 1.0, 4.0],
    }
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
            **({
                "pre_split_labels": labels,
                "atom_labels": labels,
                "atom_scales": np.ones(2),
                "changed_mask": np.zeros((height, width), dtype=bool),
                "split_parent_map": labels,
                "split_child_map": labels,
                "split_score_map": np.full((height, width), np.nan, dtype=np.float32),
                "split_decision_map": np.zeros((height, width), dtype=np.uint8),
            } if config == "layer_atomic_split" else {}),
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
        trajectory = tmp_path / "trajectory" / config
        trajectory.mkdir(parents=True, exist_ok=True)
        (trajectory / "02.json").write_text(json.dumps({
            "per_frame_translation_error": trajectory_errors[config],
        }))
    records = [{
        "config_id": config, "sequence_id": "02", "frame_start": 0,
        "frame_end": 2, "split_minus_depth_regret": 1.5,
        "split_minus_geometry_regret": 1.25,
        "split_score_mean": .2,
        "split_score_invalid_reason": None,
        "split_accepted_count": 1,
        "split_changed_pixel_ratio": .1,
        "split_pre_segment_count": 2,
        "split_post_segment_count": 3,
        "split_segment_count_delta": 1,
        "split_child_count_mean": 2,
        "split_child_summary_invalid_reason": None,
        "split_min_child_fraction_min": .2,
        "split_parent_normal_dispersion_mean": .4,
        "split_child_normal_dispersion_mean": .1,
        "split_normal_dispersion_gain_mean": .3,
        "split_normal_dispersion_invalid_reason": None,
        "split_largest_segment_ratio_delta": -.1,
        "split_area_entropy_delta": .2,
        "split_effective_segment_count_delta": 1,
        "split_tiny_child_area_ratio": 0,
        "split_boundary_ratio_delta": .1,
        "split_fragmentation_signal": False,
    } for config in DIAGNOSTIC_PROFILES]
    args = argparse.Namespace(window_size=3, overlap=1)
    build_cases(tmp_path, [interval], records, args)
    root = tmp_path / "cases" / "02" / "000000-000002"
    with np.load(root / "trace.npz") as data:
        assert "layer_atomic_split__segmentation__normalized_gap_map" in data
        assert "geometry_baseline__temporal__temporal_best_iou_map" in data
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["selection_score"] == 3.0
    assert metrics["split_minus_depth_regret"] == 1.5
    assert metrics["split_minus_geometry_regret"] == 1.25
    assert metrics["selection_reasons"] == ["trajectory_degradation"]
    assert metrics["geometry_split_comparison"]["boundary_disagreement_pre"] is not None
    assert metrics["geometry_split_comparison"]["boundary_disagreement_post"] is not None
    assert metrics["split_structural_summary"]["accepted_count"] == 1
    timeline = json.loads((root / "trajectory-timeline.json").read_text())
    assert timeline["errors"]["depth"] == [1.0, 2.0, 3.0]
    assert timeline["regrets"]["split_minus_depth_regret"] == [0.5, -1.0, 1.0]
    comparison = json.loads((root / "comparison-rendering.json").read_text())
    assert comparison["methods"] == [
        "depth", "geometry_baseline", "layer_atomic_split",
    ]
    for config in DIAGNOSTIC_PROFILES:
        rendering = json.loads((root / config / "rendering.json").read_text())
        assert len(rendering["artifacts"]) == 21
        assert rendering["availability"]["pre_split_labels"] is (
            config == "layer_atomic_split"
        )
    split_rendering = json.loads(
        (root / "layer_atomic_split" / "rendering.json").read_text()
    )
    assert split_rendering["availability"]["split_parent_map"] is True
    assert split_rendering["availability"]["split_child_map"] is True
    assert (root / "layer_atomic_split" / "split_parent_map.png").is_file()
    assert (root / "layer_atomic_split" / "split_child_map.png").is_file()
    assert (root / "artifact-manifest.json").is_file()
    assert _validate_case_artifacts(root) == (True, "complete")
    (root / "layer_atomic_split" / "scale_map.png").write_bytes(b"corrupt")
    valid, reason = _validate_case_artifacts(root)
    assert valid is False
    assert "mismatch" in reason or "unreadable" in reason


def test_build_cases_checks_all_method_directories_before_writing(tmp_path):
    interval = SelectedInterval("02", 0, 2, ("matched_control",), 1.0)
    for config in ("depth", "layer_atomic_split"):
        (tmp_path / "artifacts" / config / "02" / "pass2" / "traces").mkdir(
            parents=True
        )

    with pytest.raises(FileNotFoundError, match="geometry_baseline"):
        build_cases(
            tmp_path,
            [interval],
            [],
            argparse.Namespace(window_size=3, overlap=1),
        )

    assert not (tmp_path / "cases").exists()
