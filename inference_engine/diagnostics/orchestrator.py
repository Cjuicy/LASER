"""One-command, sequential, resumable two-pass KITTI diagnostics."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import signal
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .metrics import build_sequence_summary, evaluate_stability_guard
from .rendering import render_case, render_method_comparison
from .report import build_report
from .schema import RunManifest, SelectedInterval
from .selection import GUARD, RECOVERY, select_intervals
from .segmentation import compare_labelings
from .storage import (
    OWNER_FILE,
    RunLock,
    StorageBudget,
    atomic_write_json,
    atomic_write_npz,
    cleanup_owned_directory,
    directory_size,
)


DIAGNOSTIC_PROFILES = {
    "depth": {"segment_mode": "depth", "geometry_seg_profile": "baseline_params", "official": True},
    "geometry_baseline": {"segment_mode": "geometry", "geometry_seg_profile": "baseline_params", "official": True},
    "layer_atomic_split": {
        "segment_mode": "layer_atomic_split",
        "geometry_seg_profile": "baseline_params",
        "normal_method": "cross",
        "split_score_thresh": 0.10,
        "split_aux_confirmation": True,
        "official": True,
    },
}
DEFAULT_SEQUENCES = tuple(f"{index:02d}" for index in range(11))


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(root: Path, sequences: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for sequence in sorted(sequences):
        image_dir = root / "sequences" / sequence / "image_2"
        for path in sorted(image_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        pose = root / "poses" / f"{sequence}.txt"
        digest.update(str(pose.relative_to(root)).encode() + b"\0")
        digest.update(pose.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        repository = Path(__file__).resolve().parents[2]
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            cwd=repository,
        ).strip()
        tracked_diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary", "--", "."],
            stderr=subprocess.DEVNULL,
            cwd=repository,
        )
        if tracked_diff:
            return f"{head}+dirty-{hashlib.sha256(tracked_diff).hexdigest()[:16]}"
        return head
    except Exception:
        return "unknown-local-commit"


@contextmanager
def _termination_as_interrupt():
    """Turn SIGTERM into a normal unwind so manifest and lock are persisted."""
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _config_hash(args) -> str:
    payload = {
        "profiles": DIAGNOSTIC_PROFILES,
        "sequences": list(args.sequences), "window_size": args.window_size,
        "overlap": args.overlap, "top_conf_percentile": args.top_conf_percentile,
        "seed": args.seed, "max_selected": args.max_selected, "device": args.device,
        "max_temp_gib": args.max_temp_gib, "warn_temp_gib": args.warn_temp_gib,
        "min_free_gib": args.min_free_gib,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preflight(args) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    checkpoint = Path(args.model_ckpt)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    if args.overlap >= args.window_size or args.overlap < 0:
        raise ValueError("overlap must be non-negative and smaller than window_size")
    focus_count = len(set(args.sequences) & (RECOVERY | GUARD))
    minimum_intervals = 2 * focus_count
    if args.max_selected <= 0 or args.max_selected < minimum_intervals:
        raise ValueError(
            f"max_selected must be at least {minimum_intervals} for the requested "
            "Recovery/Guard event+control coverage"
        )
    frame_counts = {}
    pose_counts = {}
    image_shapes = {}
    for sequence in args.sequences:
        image_dir = dataset_root / "sequences" / sequence / "image_2"
        pose_file = dataset_root / "poses" / f"{sequence}.txt"
        if not image_dir.is_dir() or not pose_file.is_file():
            raise FileNotFoundError(f"KITTI sequence {sequence} requires {image_dir} and {pose_file}")
        frames = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if len(frames) < 2:
            raise ValueError(f"KITTI sequence {sequence} has fewer than two images")
        frame_counts[sequence] = len(frames)
        import cv2
        sample = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
        if sample is None or sample.ndim < 2:
            raise ValueError(f"Cannot read KITTI image dimensions: {frames[0]}")
        image_shapes[sequence] = [int(sample.shape[0]), int(sample.shape[1])]
        pose_count = sum(bool(line.strip()) for line in pose_file.read_text(encoding="utf-8").splitlines())
        if pose_count < len(frames):
            raise ValueError(
                f"KITTI sequence {sequence} has {len(frames)} images but only {pose_count} poses"
            )
        pose_counts[sequence] = pose_count
    temp = Path(args.temp_root); temp.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(temp, max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib, min_free_gib=args.min_free_gib)
    state = budget.enforce()
    return {
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "dataset_fingerprint": dataset_fingerprint(dataset_root, args.sequences),
        "frame_counts": frame_counts,
        "pose_counts": pose_counts,
        "image_shapes": image_shapes,
        "profiles": list(DIAGNOSTIC_PROFILES),
        "loop_closure": False,
        "git_commit": _git_commit(),
        "config_hash": _config_hash(args),
        "budget": state.__dict__,
    }


def estimate_storage_bytes(info: dict[str, Any], args) -> dict[str, int]:
    """Conservative compressed-storage projection for both passes and cases."""
    profiles = len(DIAGNOSTIC_PROFILES)
    total_frames = sum(info["frame_counts"].values())
    stride = args.window_size - args.overlap
    context_span = args.window_size + 4 * stride
    max_pixels = max(height * width for height, width in info["image_shapes"].values())
    # The selector proves one native-window control for every requested
    # Recovery/Guard sequence. Treat every remaining slot as a fully expanded
    # anomaly; unlike a 50/50 heuristic, this is a conservative upper bound.
    guaranteed_controls = min(
        len(set(args.sequences) & (RECOVERY | GUARD)),
        max(0, args.max_selected),
    )
    anomaly_cases = max(0, args.max_selected) - guaranteed_controls
    control_cases = guaranteed_controls
    selected_frames = min(
        total_frames,
        anomaly_cases * context_span + control_cases * args.window_size,
    )
    # Inputs, segmentation/merge, scale and temporal maps compress to roughly
    # 24 B/pixel on KITTI; 32 B/pixel leaves headroom for noisy point maps.
    dense = selected_frames * max_pixels * 32 * profiles
    scalar = total_frames * profiles * 16 * 1024
    trajectories = total_frames * profiles * 1024
    # One center-frame PLY, 15 PNGs, combined trace and metadata per case/config.
    cases = args.max_selected * profiles * (160 * 1024 * 1024 + max_pixels * 20)
    report = max(64 * 1024 * 1024, args.max_selected * 2 * 1024 * 1024)
    total = int(dense + scalar + trajectories + cases + report)
    return {
        "pass1_scalar": int(scalar + trajectories),
        "pass2_dense": int(dense),
        "cases_and_report": int(cases + report),
        "total": total,
        "selected_frame_upper_bound": int(selected_frames),
    }


def _environment() -> dict:
    result = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
    try:
        import torch
        result.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    except Exception:
        result["torch"] = None
    return result


def _manifest(args, info, run_id: str) -> RunManifest:
    return RunManifest(
        run_id=run_id, git_commit=info["git_commit"], checkpoint_sha256=info["checkpoint_sha256"],
        config_hash=info["config_hash"], dataset_fingerprint=info["dataset_fingerprint"], seed=args.seed,
        environment=_environment(), budget={
            "max_gib": args.max_temp_gib, "warn_gib": args.warn_temp_gib,
            "min_free_gib": args.min_free_gib, "frame_counts": info["frame_counts"],
        },
    )


def _resume_manifest(path: Path, info: dict) -> RunManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    comparisons = {
        "checkpoint": (data.get("checkpoint_sha256"), info["checkpoint_sha256"]),
        "dataset": (data.get("dataset_fingerprint"), info["dataset_fingerprint"]),
        "config": (data.get("config_hash"), info["config_hash"]),
        "commit": (data.get("git_commit"), info["git_commit"]),
    }
    changed = [name for name, (old, new) in comparisons.items() if old != new]
    if changed:
        raise ValueError(f"Cannot resume: {', '.join(changed)} fingerprint changed")
    return RunManifest.from_dict(data)


def _phase_done(manifest: RunManifest, phase: str, config: str) -> bool:
    return manifest.state.get(phase, {}).get(config, {}).get("*") == "complete"


def _artifact_inventory(output_root: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.add(path)
    return {
        str(path.relative_to(output_root)): {
            "size": path.stat().st_size,
            "sha256": checkpoint_sha256(path),
        }
        for path in sorted(files)
    }


def _validate_sequence_checkpoint(
    output_root: Path,
    checkpoint_path: Path,
    *,
    run_id: str,
    config_id: str,
    sequence_id: str,
    pass_id: int,
    checkpoint_hash: str,
    expected_frames: int,
    window_size: int,
    selected_intervals: list[SelectedInterval] | tuple[SelectedInterval, ...] = (),
    implementation_fingerprint: str | None = None,
    config_hash: str | None = None,
    dataset_hash: str | None = None,
) -> tuple[bool, str]:
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False, "missing_or_invalid_checkpoint"
    expected = {
        "run_id": run_id,
        "config_id": config_id,
        "sequence_id": sequence_id,
        "pass_id": pass_id,
        "checkpoint_sha256": checkpoint_hash,
        "frame_count": expected_frames,
        "pose_count": expected_frames,
    }
    if implementation_fingerprint is not None:
        expected["implementation_fingerprint"] = implementation_fingerprint
    if config_hash is not None:
        expected["config_hash"] = config_hash
    if dataset_hash is not None:
        expected["dataset_fingerprint"] = dataset_hash
    for field, value in expected.items():
        if payload.get(field) != value:
            return False, f"checkpoint_{field}_mismatch"
    inventory = payload.get("artifacts")
    if not isinstance(inventory, dict) or not inventory:
        return False, "missing_artifact_inventory"
    for relative, metadata in inventory.items():
        path = output_root / relative
        if not path.is_file():
            return False, f"missing_artifact:{relative}"
        if path.stat().st_size != metadata.get("size"):
            return False, f"artifact_size_mismatch:{relative}"
        if checkpoint_sha256(path) != metadata.get("sha256"):
            return False, f"artifact_checksum_mismatch:{relative}"
    trajectory_root = output_root / "trajectory"
    if pass_id != 1:
        trajectory_root = trajectory_root / f"pass{pass_id}"
    shard_root = output_root / "artifacts" / config_id / sequence_id / f"pass{pass_id}"
    required = {
        str((trajectory_root / config_id / f"{sequence_id}.json").relative_to(output_root)),
        str((trajectory_root / config_id / f"{sequence_id}.npz").relative_to(output_root)),
        str((shard_root / "segmentation.jsonl").relative_to(output_root)),
        str((shard_root / "temporal.jsonl").relative_to(output_root)),
    }
    if expected_frames > window_size:
        required.add(str((shard_root / "scale.jsonl").relative_to(output_root)))
    if not required <= set(inventory):
        missing_required = sorted(required - set(inventory))
        return False, "required_artifact_not_in_inventory:" + ",".join(missing_required)
    if any(int(inventory[relative].get("size", 0)) <= 0 for relative in required):
        return False, "empty_required_artifact"
    if pass_id == 2:
        trace_root = output_root / "artifacts" / config_id / sequence_id / "pass2" / "traces"
        for interval in selected_intervals:
            if interval.sequence_id != sequence_id:
                continue
            center = min((interval.start_frame + interval.end_frame) // 2, expected_frames - 1)
            for prefix in ("inputs", "segmentation"):
                relative = str((trace_root / f"{prefix}-frame-{center:06d}.npz").relative_to(output_root))
                if relative not in inventory:
                    return False, f"missing_selected_trace:{relative}"
    return True, "complete"


def _worker_command(args, *, pass_id: int, config_id: str, run_id: str, selected: Path | None = None) -> list[str]:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_segmentation_diagnostics.py"
    command = [
        sys.executable, str(script), "--worker", "--pass-id", str(pass_id), "--config-id", config_id,
        "--run-id", run_id, "--dataset-root", str(args.dataset_root), "--model-ckpt", str(args.model_ckpt),
        "--output-dir", str(args.output_dir), "--temp-root", str(args.temp_root),
        "--window-size", str(args.window_size), "--overlap", str(args.overlap),
        "--top-conf-percentile", str(args.top_conf_percentile), "--seed", str(args.seed),
        "--max-temp-gib", str(args.max_temp_gib), "--warn-temp-gib", str(args.warn_temp_gib),
        "--min-free-gib", str(args.min_free_gib), "--device", str(args.device),
        "--sequences", *args.sequences,
    ]
    if selected is not None:
        command.extend(["--selected-intervals", str(selected)])
    return command


def _read_trajectory_results(run_dir: Path, *, pass_id: int = 1) -> dict:
    root = run_dir / "trajectory" if pass_id == 1 else run_dir / "trajectory" / f"pass{pass_id}"
    results: dict[str, dict[str, dict]] = {}
    for config in DIAGNOSTIC_PROFILES:
        results[config] = {}
        for path in sorted((root / config).glob("*.json")) if (root / config).exists() else []:
            results[config][path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return results


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_selection_records(run_dir: Path, trajectory_results: dict, args) -> list[dict]:
    records = []
    stride = args.window_size - args.overlap
    for config in DIAGNOSTIC_PROFILES:
        for sequence in args.sequences:
            evaluation = trajectory_results.get(config, {}).get(sequence, {})
            errors = np.asarray(evaluation.get("per_frame_translation_error") or [], dtype=float)
            split_errors = np.asarray(trajectory_results.get("layer_atomic_split", {}).get(sequence, {}).get("per_frame_translation_error") or [], dtype=float)
            geometry_errors = np.asarray(trajectory_results.get("geometry_baseline", {}).get(sequence, {}).get("per_frame_translation_error") or [], dtype=float)
            depth_errors = np.asarray(trajectory_results.get("depth", {}).get(sequence, {}).get("per_frame_translation_error") or [], dtype=float)
            gt_positions = np.empty((0, 3), dtype=float)
            gt_rotations = np.empty((0, 3, 3), dtype=float)
            trajectory_npz = Path(run_dir) / "trajectory" / "layer_atomic_split" / f"{sequence}.npz"
            if trajectory_npz.exists():
                with np.load(trajectory_npz) as data:
                    ground_truth = data["ground_truth"]
                gt_positions = ground_truth[:, :3, 3]
                gt_rotations = ground_truth[:, :3, :3]
            base = Path(run_dir) / "artifacts" / config / sequence / "pass1"
            segmentation = _read_jsonl(base / "segmentation.jsonl")
            split_segmentation = _read_jsonl(
                Path(run_dir) / "artifacts" / "layer_atomic_split" / sequence
                / "pass1" / "segmentation.jsonl"
            )
            scale = {item["context"]["window_id"]: item for item in _read_jsonl(base / "scale.jsonl")}
            temporal = {item["context"]["window_id"]: item for item in _read_jsonl(base / "temporal.jsonl")}
            by_window: dict[int, list[dict]] = {}
            for item in segmentation:
                by_window.setdefault(int(item["context"]["window_id"]), []).append(item)
            split_by_window: dict[int, list[dict]] = {}
            for item in split_segmentation:
                split_by_window.setdefault(
                    int(item["context"]["window_id"]), [],
                ).append(item)
            window_count = max(1, int(np.ceil(max(len(errors) - args.overlap, 1) / stride)))
            for window_id in range(window_count):
                start = window_id * stride; end = min(start + args.window_size - 1, max(len(errors) - 1, start))
                split_slice = split_errors[start:end + 1]
                frame_metrics = by_window.get(window_id, [])
                split_frame_metrics = split_by_window.get(window_id, [])
                final_metrics = [item["metrics"].get("final", {}) for item in frame_metrics]
                merge_metrics = [item["metrics"].get("merge", {}) for item in frame_metrics]
                lsr = max((float(item.get("largest_segment_ratio") or 0) for item in final_metrics), default=0.0)
                growth = max((float(item.get("largest_growth_ratio") or 0) for item in final_metrics), default=0.0)
                cross = max((float(item.get("cross_coarse_merge_ratio") or 0) for item in merge_metrics), default=0.0)
                scale_item = scale.get(window_id, {}).get("metrics", {})
                scale_p90 = ((scale_item.get("scale_log_mad_quantiles") or {}).get("p90") or 0.0)
                temporal_item = temporal.get(window_id, {}).get("metrics", {})
                confidence_values = [
                    item["metrics"].get("confidence_mean")
                    for item in frame_metrics
                    if item["metrics"].get("confidence_mean") is not None
                ]
                position_slice = gt_positions[start:end + 1]
                speed = float(np.mean(np.linalg.norm(np.diff(position_slice, axis=0), axis=1))) if len(position_slice) > 1 else 0.0
                rotation_slice = gt_rotations[start:end + 1]
                turn_angles = []
                for rotation_index in range(max(0, len(rotation_slice) - 1)):
                    delta_rotation = rotation_slice[rotation_index].T @ rotation_slice[rotation_index + 1]
                    cosine = np.clip((np.trace(delta_rotation) - 1) / 2, -1, 1)
                    turn_angles.append(float(np.arccos(cosine)))
                def mean_regret(reference):
                    reference_values = reference[start:end + 1]
                    return float(np.nanmean(split_slice - reference_values)) if split_slice.size and len(reference_values) == len(split_slice) else None
                split_metrics = [
                    item.get("metrics", {}).get("split", {})
                    for item in split_frame_metrics
                ]
                accepted_values = [
                    int(metrics["split_accepted_count"])
                    for metrics in split_metrics
                    if metrics.get("split_accepted_count") is not None
                ]
                changed_values = [
                    float(metrics["split_changed_pixel_ratio"])
                    for metrics in split_metrics
                    if metrics.get("split_changed_pixel_ratio") is not None
                ]
                depth_regret = mean_regret(depth_errors)
                geometry_regret = mean_regret(geometry_errors)
                records.append({
                    "config_id": config, "sequence_id": sequence, "window_id": window_id,
                    "frame_start": start, "frame_end": end,
                    "split_minus_depth_regret": depth_regret,
                    "split_minus_geometry_regret": geometry_regret,
                    "split_accepted_count": (
                        int(sum(accepted_values)) if accepted_values else None
                    ),
                    "split_changed_pixel_ratio": (
                        float(np.mean(changed_values)) if changed_values else None
                    ),
                    "merge_anomaly": cross,
                    "atom_anomaly": lsr + np.log1p(growth),
                    "scale_dispersion": float(scale_p90),
                    "temporal_churn": float(temporal_item.get("segment_churn_ratio") or temporal_item.get("unmatched_area_ratio") or 0),
                    "gt_speed": speed,
                    "gt_turn": float(np.mean(turn_angles)) if turn_angles else 0.0,
                    "confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
                })
    return records


def build_cases(
    run_dir: Path,
    intervals: list[SelectedInterval],
    records: list[dict],
    args,
    *,
    budget: StorageBudget | None = None,
) -> None:
    stride = args.window_size - args.overlap
    for interval in intervals:
        center = (interval.start_frame + interval.end_frame) // 2
        comparison_data: dict[str, dict[str, np.ndarray]] = {}
        interval_root = run_dir / "cases" / interval.sequence_id / f"{interval.start_frame:06d}-{interval.end_frame:06d}"
        combined_arrays: dict[str, np.ndarray] = {}
        artifact_manifest: dict[str, Any] = {
            "interval": interval.to_dict(),
            "selection_score": interval.score,
            "configs": {},
        }
        for config in DIAGNOSTIC_PROFILES:
            trace_dir = run_dir / "artifacts" / config / interval.sequence_id / "pass2" / "traces"
            if not trace_dir.is_dir():
                raise FileNotFoundError(f"Missing selected trace directory: {trace_dir}")
            required = [
                trace_dir / f"inputs-frame-{center:06d}.npz",
                trace_dir / f"segmentation-frame-{center:06d}.npz",
            ]
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Missing required selected trace(s): " + ", ".join(map(str, missing))
                )
            window_id = center // stride
            optional = [
                trace_dir / f"scale-window-{window_id:06d}.npz",
                trace_dir / f"temporal-window-{window_id:06d}.npz",
            ]
            paths = required + [path for path in optional if path.is_file()]
            case_dir = interval_root / config
            if budget is not None:
                budget.enforce(estimated_bytes=100 * 1024 * 1024)
            local_index = min(max(center - window_id * stride, 0), args.window_size - 1)
            rendered = render_case(paths, case_dir, frame_index=local_index)
            if budget is not None:
                budget.enforce()
            values: dict[str, np.ndarray] = {}
            config_artifacts = {
                "available": [], "unavailable": [],
                "rendered": sorted(path.name for path in rendered.values()),
            }
            for path in paths:
                kind = path.name.split("-", 1)[0]
                config_artifacts["available"].append({
                    "kind": kind,
                    "path": str(path.relative_to(run_dir)),
                    "sha256": checkpoint_sha256(path),
                })
                with np.load(path) as data:
                    for name in data.files:
                        value = data[name]
                        combined_arrays[f"{config}__{kind}__{name}"] = value
                        if name in ("final_labels", "scale_map"):
                            selected_value = value
                            if selected_value.ndim == 3:
                                selected_value = selected_value[
                                    min(max(local_index, 0), len(selected_value) - 1)
                                ]
                            values[name] = selected_value
            for kind, path in zip(("scale", "temporal"), optional, strict=True):
                if not path.is_file():
                    config_artifacts["unavailable"].append({
                        "kind": kind,
                        "reason": "not_emitted_for_this_window",
                    })
            artifact_manifest["configs"][config] = config_artifacts
            comparison_data[config] = values
            matching = [row for row in records if row["config_id"] == config and row["sequence_id"] == interval.sequence_id and row["frame_start"] <= center <= row["frame_end"]]
            metrics = {"interval": interval.to_dict(), "config_id": config, **(matching[0] if matching else {})}
            atomic_write_json(case_dir / "metrics.json", metrics)
        atomic = comparison_data.get("layer_atomic_split", {})
        geometry = comparison_data.get("geometry_baseline", {})
        if len(comparison_data) != len(DIAGNOSTIC_PROFILES):
            raise RuntimeError(f"Incomplete method comparison for {interval.sequence_id}/{center}")
        if budget is not None:
            budget.enforce(estimated_bytes=sum(value.nbytes for value in combined_arrays.values()))
        atomic_write_npz(interval_root / "trace.npz", **combined_arrays)
        if budget is not None:
            budget.enforce()
        if "final_labels" in atomic and "final_labels" in geometry:
            comparison_metrics = compare_labelings(atomic["final_labels"], geometry["final_labels"])
            render_method_comparison(
                atomic["final_labels"], geometry["final_labels"], interval_root,
                left_scale=atomic.get("scale_map"), right_scale=geometry.get("scale_map"),
            )
            center_records = [
                row for row in records
                if row["config_id"] == "layer_atomic_split"
                and row["sequence_id"] == interval.sequence_id
                and row["frame_start"] <= center <= row["frame_end"]
            ]
            center_record = min(
                center_records,
                key=lambda row: abs((row["frame_start"] + row["frame_end"]) / 2 - center),
            ) if center_records else None
            depth_regret = (
                center_record.get("split_minus_depth_regret") if center_record else None
            )
            geometry_regret = (
                center_record.get("split_minus_geometry_regret") if center_record else None
            )
            atomic_write_json(interval_root / "metrics.json", {
                "comparison": "layer_atomic_split_vs_geometry_baseline",
                "interval": interval.to_dict(),
                "selection_score": interval.score,
                "split_minus_depth_regret": depth_regret,
                "split_minus_geometry_regret": geometry_regret,
                **comparison_metrics,
            })
        else:
            raise RuntimeError(
                f"Missing comparable final labels for {interval.sequence_id}/{center}"
            )
        generated = _artifact_inventory(interval_root, [interval_root])
        generated.pop("artifact-manifest.json", None)
        artifact_manifest["generated_artifacts"] = generated
        atomic_write_json(interval_root / "artifact-manifest.json", artifact_manifest)
        valid, reason = _validate_case_artifacts(interval_root)
        if not valid:
            raise RuntimeError(f"Case artifact verification failed for {interval_root}: {reason}")


def _validate_case_artifacts(interval_root: Path) -> tuple[bool, str]:
    manifest_path = interval_root / "artifact-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False, "missing_or_invalid_artifact_manifest"
    inventory = manifest.get("generated_artifacts")
    if not isinstance(inventory, dict) or not inventory:
        return False, "missing_generated_artifact_inventory"
    actual = {
        str(path.relative_to(interval_root))
        for path in interval_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(inventory):
        return False, "generated_artifact_set_mismatch"
    required = {"trace.npz", "metrics.json", "segmentation_disagreement.png"}
    for config in DIAGNOSTIC_PROFILES:
        required.update({
            f"{config}/rendering.json",
            f"{config}/metrics.json",
            f"{config}/segments.ply",
        })
        try:
            rendering = json.loads(
                (interval_root / config / "rendering.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return False, f"invalid_rendering_manifest:{config}"
        artifacts = rendering.get("artifacts", {})
        if not isinstance(artifacts, dict) or len(artifacts) != 15:
            return False, f"incomplete_rendering_manifest:{config}"
        required.update(f"{config}/{filename}" for filename in artifacts.values())
    if not required <= set(inventory):
        return False, "missing_required_case_artifact:" + ",".join(sorted(required - set(inventory)))
    import cv2
    for relative, metadata in inventory.items():
        path = interval_root / relative
        if not path.is_file() or path.stat().st_size != metadata.get("size"):
            return False, f"case_artifact_size_mismatch:{relative}"
        if checkpoint_sha256(path) != metadata.get("sha256"):
            return False, f"case_artifact_checksum_mismatch:{relative}"
        try:
            if path.suffix == ".png" and cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
                return False, f"unreadable_png:{relative}"
            if path.suffix == ".npz":
                with np.load(path) as data:
                    if not data.files:
                        return False, f"empty_npz:{relative}"
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".ply" and not path.read_bytes().startswith(b"ply\n"):
                return False, f"unreadable_ply:{relative}"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return False, f"unreadable_case_artifact:{relative}:{exc}"
    return True, "complete"


def run_master(args, *, runner=subprocess.run) -> int:
    output = Path(args.output_dir)
    manifest_path = output / "manifest.json"
    if args.report_only:
        if not manifest_path.is_file():
            raise FileNotFoundError("--report-only requires an existing manifest.json")
        print("[phase report] rebuilding report without inference or manifest mutation")
        build_report(output)
        return 0
    if manifest_path.exists() and not (args.resume or args.report_only):
        raise FileExistsError(
            f"Output already contains a diagnostic manifest: {manifest_path}; "
            "use --resume, --report-only, or a new output directory."
        )
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output}; use --resume or a new directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    info = preflight(args)
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError("--resume requested but manifest.json is missing")
        manifest = _resume_manifest(manifest_path, info)
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{info['checkpoint_sha256'][:8]}"
        manifest = _manifest(args, info, run_id)
    projection = estimate_storage_bytes(info, args)
    manifest.budget["projection_bytes"] = projection
    output_budget = StorageBudget(
        output, max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib,
        min_free_gib=args.min_free_gib,
    )
    projected_remaining = max(0, projection["total"] - directory_size(output))
    projected_state = output_budget.enforce(estimated_bytes=projected_remaining)
    temp_peak = max(info["frame_counts"].values()) * 8 * 1024
    StorageBudget(
        args.temp_root, max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib,
        min_free_gib=args.min_free_gib,
    ).enforce(estimated_bytes=temp_peak)
    if not args.resume:
        atomic_write_json(manifest_path, manifest.to_dict())
    print(
        f"[phase preflight] {len(DIAGNOSTIC_PROFILES)} sequential profiles, {sum(info['frame_counts'].values())} frames; "
        f"projected total {projection['total'] / 2**30:.2f} GiB "
        f"(pass2 {projection['pass2_dense'] / 2**30:.2f} GiB, "
        f"cases/report {projection['cases_and_report'] / 2**30:.2f} GiB); "
        f"budget state={projected_state.level}"
    )
    if args.dry_run:
        manifest.status = "dry_run"
        atomic_write_json(manifest_path, manifest.to_dict())
        print("[phase dry-run] no model inference executed")
        return 0
    lock = RunLock(output / ".diagnostic.lock", manifest.run_id)
    with _termination_as_interrupt(), lock:
        budget = StorageBudget(args.temp_root, max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib, min_free_gib=args.min_free_gib)
        run_temp = Path(args.temp_root) / manifest.run_id
        if run_temp.exists():
            marker = run_temp / OWNER_FILE
            try:
                owner = json.loads(marker.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise PermissionError(f"Cannot verify diagnostic temp ownership: {run_temp}") from exc
            if owner.get("run_id") != manifest.run_id:
                raise PermissionError(f"Cannot verify diagnostic temp ownership: {run_temp}")
        else:
            run_temp.mkdir(parents=True)
            atomic_write_json(run_temp / OWNER_FILE, {"run_id": manifest.run_id})

        def enforce_budget():
            state = budget.enforce()
            if state.level == "warning":
                print(
                    f"[storage warning] projected temporary use "
                    f"{state.projected_bytes / 2**30:.2f} GiB; hard limit "
                    f"{state.max_bytes / 2**30:.2f} GiB"
                )
            return state

        def config_complete(pass_id, config, sequences, selected=()):
            failures = []
            for sequence in sequences:
                valid, reason = _validate_sequence_checkpoint(
                    output,
                    output / "checkpoints" / f"pass{pass_id}" / config / f"{sequence}.json",
                    run_id=manifest.run_id,
                    config_id=config,
                    sequence_id=sequence,
                    pass_id=pass_id,
                    checkpoint_hash=info["checkpoint_sha256"],
                    expected_frames=info["frame_counts"][sequence],
                    window_size=args.window_size,
                    selected_intervals=selected,
                    implementation_fingerprint=manifest.git_commit,
                    config_hash=manifest.config_hash,
                    dataset_hash=manifest.dataset_fingerprint,
                )
                if not valid:
                    failures.append(f"{sequence}:{reason}")
            return not failures, failures
        try:
            for index, config in enumerate(DIAGNOSTIC_PROFILES, 1):
                pass1_valid, pass1_failures = config_complete(1, config, args.sequences)
                if _phase_done(manifest, "pass1", config) and pass1_valid:
                    print(f"[phase pass1 {index}/{len(DIAGNOSTIC_PROFILES)}] {config}: resume skip")
                    continue
                if _phase_done(manifest, "pass1", config) and not pass1_valid:
                    print(f"[phase pass1 {index}/{len(DIAGNOSTIC_PROFILES)}] {config}: invalid resume state ({'; '.join(pass1_failures)}); rerunning")
                print(f"[phase pass1 {index}/{len(DIAGNOSTIC_PROFILES)}] {config}: starting")
                enforce_budget()
                runner(_worker_command(args, pass_id=1, config_id=config, run_id=manifest.run_id), check=True)
                pass1_valid, pass1_failures = config_complete(1, config, args.sequences)
                if not pass1_valid:
                    raise RuntimeError(
                        f"Pass 1 artifact verification failed for {config}: "
                        + "; ".join(pass1_failures)
                    )
                for sequence in args.sequences:
                    manifest.mark("pass1", config, sequence, "complete")
                manifest.mark("pass1", config, "*", "complete")
                atomic_write_json(manifest_path, manifest.to_dict())
            trajectory = _read_trajectory_results(output)
            invalid_results = [
                f"{config}/{sequence}"
                for config in DIAGNOSTIC_PROFILES
                for sequence in args.sequences
                if not trajectory.get(config, {}).get(sequence, {}).get("valid", False)
            ]
            if invalid_results:
                raise RuntimeError(
                    "Pass 1 trajectory verification failed for: "
                    + ", ".join(invalid_results)
                )
            summary = build_sequence_summary(trajectory)
            split_ates = {seq: value["ate_rmse"] for seq, value in trajectory.get("layer_atomic_split", {}).items() if value.get("valid")}
            summary["stability_guard"] = evaluate_stability_guard(
                split_ates,
                {seq: value["ate_rmse"] for seq, value in trajectory.get("depth", {}).items() if value.get("valid")},
                expected_sequences=args.sequences,
            ) if split_ates else {"passed": False, "baseline_config": "depth", "failure_reasons": ["missing_layer_atomic_split"]}
            summary["sequence_metrics"] = trajectory
            atomic_write_json(output / "summary.json", summary)
            records = build_selection_records(output, trajectory, args)
            atomic_write_json(output / "selection_records.json", records)
            intervals = select_intervals(records, limit=args.max_selected)
            intervals = [
                SelectedInterval(
                    item.sequence_id,
                    min(item.start_frame, info["frame_counts"][item.sequence_id] - 1),
                    min(item.end_frame, info["frame_counts"][item.sequence_id] - 1),
                    item.reasons,
                    item.score,
                )
                for item in intervals
            ]
            selected_path = output / "selected_intervals.json"
            atomic_write_json(selected_path, [item.to_dict() for item in intervals])
            selected_sequences = sorted({item.sequence_id for item in intervals})
            pass2_args = argparse.Namespace(**vars(args)); pass2_args.sequences = selected_sequences
            for index, config in enumerate(DIAGNOSTIC_PROFILES, 1):
                pass2_valid, pass2_failures = config_complete(
                    2, config, selected_sequences, intervals
                ) if selected_sequences else (True, [])
                if not selected_sequences or (_phase_done(manifest, "pass2", config) and pass2_valid):
                    continue
                if _phase_done(manifest, "pass2", config) and not pass2_valid:
                    print(f"[phase pass2 {index}/{len(DIAGNOSTIC_PROFILES)}] {config}: invalid resume state ({'; '.join(pass2_failures)}); rerunning")
                print(f"[phase pass2 {index}/{len(DIAGNOSTIC_PROFILES)}] {config}: full rerun of {','.join(selected_sequences)}; dense writes selected only")
                enforce_budget()
                runner(_worker_command(pass2_args, pass_id=2, config_id=config, run_id=manifest.run_id, selected=selected_path), check=True)
                pass2_valid, pass2_failures = config_complete(
                    2, config, selected_sequences, intervals
                )
                if not pass2_valid:
                    raise RuntimeError(
                        f"Pass 2 artifact verification failed for {config}: "
                        + "; ".join(pass2_failures)
                    )
                for sequence in selected_sequences:
                    manifest.mark("pass2", config, sequence, "complete")
                manifest.mark("pass2", config, "*", "complete")
                atomic_write_json(manifest_path, manifest.to_dict())
            print("[phase verify] rendering selected cases")
            build_cases(output, intervals, records, args, budget=output_budget)
            incomplete_cases = []
            for item in intervals:
                root = output / "cases" / item.sequence_id / f"{item.start_frame:06d}-{item.end_frame:06d}"
                valid, reason = _validate_case_artifacts(root)
                if not valid:
                    incomplete_cases.append(
                        f"{item.sequence_id}/{item.start_frame}-{item.end_frame}:{reason}"
                    )
            if incomplete_cases:
                raise RuntimeError("Selected case verification failed for: " + ", ".join(incomplete_cases))
            print("[phase cleanup] removing run-owned metrics-only engine cache")
            if run_temp.exists():
                cleanup_owned_directory(run_temp, manifest.run_id)
            manifest.status = "complete"
            atomic_write_json(manifest_path, manifest.to_dict())
            print("[phase report] building offline HTML/CSV report")
            output_budget.enforce(
                estimated_bytes=min(50 * 1024 * 1024, output_budget.max_bytes // 20)
            )
            build_report(output)
            output_budget.enforce()
            print(f"[phase complete] report: {output / 'report' / 'index.html'}")
            return 0
        except BaseException:
            manifest.status = "interrupted" if isinstance(sys.exc_info()[1], KeyboardInterrupt) else "incomplete"
            atomic_write_json(manifest_path, manifest.to_dict())
            try: build_report(output)
            except Exception: pass
            raise


def _load_kitti_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] != 12:
        raise ValueError(f"KITTI pose file must contain 12 columns: {path}")
    poses = np.repeat(np.eye(4)[None], len(values), axis=0)
    poses[:, :3, :] = values.reshape(-1, 3, 4)
    return poses


def _load_model(checkpoint: Path, device: str):
    import torch
    from pi3.models.pi3 import Pi3
    model = Pi3().to(device)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device=device)
    else:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    return model


def run_worker(args) -> int:
    import torch
    from inference_engine.streaming_window_engine import StreamingWindowEngine
    from utils.load_fn import load_and_preprocess_images
    from .sink import FileDiagnosticSink
    from .trajectory import evaluate_trajectory

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = args.device
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    expected = checkpoint_sha256(args.model_ckpt)
    manifest = json.loads((Path(args.output_dir) / "manifest.json").read_text())
    if expected != manifest["checkpoint_sha256"]:
        raise ValueError("worker checkpoint SHA-256 differs from manifest")
    run_temp = Path(args.temp_root) / args.run_id
    try:
        temp_owner = json.loads((run_temp / OWNER_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PermissionError(f"Cannot verify diagnostic temp ownership: {run_temp}") from exc
    if temp_owner.get("run_id") != args.run_id:
        raise PermissionError(f"Cannot verify diagnostic temp ownership: {run_temp}")
    profile = DIAGNOSTIC_PROFILES[args.config_id]
    selected = []
    if args.selected_intervals:
        selected = [SelectedInterval.from_dict(item) for item in json.loads(Path(args.selected_intervals).read_text())]
    output_root = Path(args.output_dir)
    artifact_root = output_root / "artifacts"
    checkpoint_root = output_root / "checkpoints" / f"pass{args.pass_id}" / args.config_id
    trajectory_root = output_root / "trajectory"
    if args.pass_id != 1:
        trajectory_root = trajectory_root / f"pass{args.pass_id}"

    def sequence_checkpoint(sequence):
        return checkpoint_root / f"{sequence}.json"

    def is_sequence_complete(sequence):
        valid, _ = _validate_sequence_checkpoint(
            output_root,
            sequence_checkpoint(sequence),
            run_id=args.run_id,
            config_id=args.config_id,
            sequence_id=sequence,
            pass_id=args.pass_id,
            checkpoint_hash=expected,
            expected_frames=int(manifest["budget"]["frame_counts"][sequence]),
            window_size=args.window_size,
            selected_intervals=selected,
            implementation_fingerprint=manifest.get("git_commit"),
            config_hash=manifest.get("config_hash"),
            dataset_hash=manifest.get("dataset_fingerprint"),
        )
        return valid

    pending_sequences = []
    for sequence in args.sequences:
        if is_sequence_complete(sequence):
            print(f"[worker pass{args.pass_id}] {args.config_id}/{sequence}: resume skip")
            continue
        # Remove only this run's incomplete sequence/pass shards before retrying,
        # so JSONL records cannot be duplicated across resumes.
        shutil.rmtree(
            artifact_root / args.config_id / sequence / f"pass{args.pass_id}",
            ignore_errors=True,
        )
        target = trajectory_root / args.config_id
        (target / f"{sequence}.json").unlink(missing_ok=True)
        (target / f"{sequence}.npz").unlink(missing_ok=True)
        sequence_checkpoint(sequence).unlink(missing_ok=True)
        pending_sequences.append(sequence)
    if not pending_sequences:
        return 0

    artifact_budget = StorageBudget(Path(args.output_dir), max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib, min_free_gib=args.min_free_gib)
    temp_budget = StorageBudget(Path(args.temp_root), max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib, min_free_gib=args.min_free_gib)
    sink = FileDiagnosticSink(artifact_root, selected_intervals=selected, budget=artifact_budget)
    model = _load_model(Path(args.model_ckpt), device)
    try:
        for sequence in pending_sequences:
            print(f"[worker pass{args.pass_id}] {args.config_id}/{sequence}")
            image_dir = Path(args.dataset_root) / "sequences" / sequence / "image_2"
            image_paths = sorted(str(path) for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
            cache_root = Path(args.temp_root) / args.run_id / args.config_id / sequence
            cache_root.mkdir(parents=True, exist_ok=True)
            engine = StreamingWindowEngine(
                model, inference_device=device, dtype=dtype, intermediate_device=device,
                process_device="cpu", top_conf_percentile=args.top_conf_percentile,
                window_size=args.window_size, overlap=args.overlap, depth_refine=True,
                cache_root=str(cache_root), benchmark_latency=False,
                segment_mode=profile["segment_mode"], geometry_seg_profile=profile["geometry_seg_profile"],
                normal_method=profile.get("normal_method", "cross"),
                split_score_thresh=profile.get("split_score_thresh", 0.10),
                split_aux_confirmation=profile.get("split_aux_confirmation", True),
                diagnostic_sink=sink, diagnostic_run_id=args.run_id,
                diagnostic_sequence_id=sequence, diagnostic_pass=args.pass_id,
                cache_policy="metrics-only",
                storage_budget=temp_budget,
            )
            engine.begin()
            for window in engine.img_sliding_window(image_paths):
                # Keep the producer/queue CPU-resident.  The inference worker
                # performs the sole upload, avoiding a GPU->CPU->GPU round trip.
                engine(load_and_preprocess_images(window))
            engine.end()
            poses = engine.parse_pose_cache_summary()["extrinsic"][0].cpu().numpy()
            if len(poses) != len(image_paths):
                raise RuntimeError(
                    f"Inference produced {len(poses)} poses for {len(image_paths)} images "
                    f"in sequence {sequence}; a worker thread may have failed."
                )
            gt = _load_kitti_poses(Path(args.dataset_root) / "poses" / f"{sequence}.txt")
            evaluation = evaluate_trajectory(poses, gt)
            aligned = evaluation.pop("aligned_poses", None)
            target = trajectory_root / args.config_id; target.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target / f"{sequence}.json", evaluation)
            artifact_budget.enforce(
                estimated_bytes=int(poses.nbytes + gt[:len(poses)].nbytes + (np.asarray(aligned).nbytes if aligned is not None else 0) + 4096)
            )
            atomic_write_npz(
                target / f"{sequence}.npz",
                predicted=poses,
                ground_truth=gt[:len(poses)],
                aligned=np.asarray(aligned) if aligned is not None else np.empty(0),
            )
            artifact_budget.enforce()
            shard_root = artifact_root / args.config_id / sequence / f"pass{args.pass_id}"
            inventory = _artifact_inventory(
                output_root,
                [
                    target / f"{sequence}.json",
                    target / f"{sequence}.npz",
                    shard_root,
                ],
            )
            atomic_write_json(sequence_checkpoint(sequence), {
                "run_id": args.run_id, "config_id": args.config_id,
                "sequence_id": sequence, "pass_id": args.pass_id,
                "frame_count": len(image_paths), "pose_count": len(poses),
                "checkpoint_sha256": expected,
                "implementation_fingerprint": manifest.get("git_commit"),
                "config_hash": manifest.get("config_hash"),
                "dataset_fingerprint": manifest.get("dataset_fingerprint"),
                "artifacts": inventory,
            })
            sink.refresh_usage()
            shutil.rmtree(cache_root, ignore_errors=True)
    finally:
        sink.close()
    return 0
