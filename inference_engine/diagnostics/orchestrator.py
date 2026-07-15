"""One-command, sequential, resumable two-pass KITTI diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .metrics import build_sequence_summary, evaluate_stability_guard
from .rendering import render_case
from .report import build_report
from .schema import RunManifest, SelectedInterval
from .selection import select_intervals
from .storage import RunLock, StorageBudget, atomic_write_json


DIAGNOSTIC_PROFILES = {
    "depth": {"segment_mode": "depth", "geometry_seg_profile": "baseline_params", "official": True},
    "geometry_baseline": {"segment_mode": "geometry", "geometry_seg_profile": "baseline_params", "official": True},
    "layer_atomic": {"segment_mode": "layer_atomic", "geometry_seg_profile": "baseline_params", "official": True},
    "geometry_legacy_reference": {"segment_mode": "geometry", "geometry_seg_profile": "legacy", "official": False},
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown-local-commit"


def _config_hash(args) -> str:
    payload = {
        "profiles": DIAGNOSTIC_PROFILES,
        "sequences": list(args.sequences), "window_size": args.window_size,
        "overlap": args.overlap, "top_conf_percentile": args.top_conf_percentile,
        "seed": args.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preflight(args) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    checkpoint = Path(args.model_ckpt)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    if args.overlap >= args.window_size or args.overlap < 0:
        raise ValueError("overlap must be non-negative and smaller than window_size")
    frame_counts = {}
    for sequence in args.sequences:
        image_dir = dataset_root / "sequences" / sequence / "image_2"
        pose_file = dataset_root / "poses" / f"{sequence}.txt"
        if not image_dir.is_dir() or not pose_file.is_file():
            raise FileNotFoundError(f"KITTI sequence {sequence} requires {image_dir} and {pose_file}")
        frames = [path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if len(frames) < 2:
            raise ValueError(f"KITTI sequence {sequence} has fewer than two images")
        frame_counts[sequence] = len(frames)
    temp = Path(args.temp_root); temp.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(temp, max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib, min_free_gib=args.min_free_gib)
    state = budget.enforce()
    return {
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "dataset_fingerprint": dataset_fingerprint(dataset_root, args.sequences),
        "frame_counts": frame_counts,
        "profiles": list(DIAGNOSTIC_PROFILES),
        "loop_closure": False,
        "git_commit": _git_commit(),
        "config_hash": _config_hash(args),
        "budget": state.__dict__,
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
            geometry_errors = np.asarray(trajectory_results.get("geometry_baseline", {}).get(sequence, {}).get("per_frame_translation_error") or [], dtype=float)
            base = Path(run_dir) / "artifacts" / config / sequence / "pass1"
            segmentation = _read_jsonl(base / "segmentation.jsonl")
            scale = {item["context"]["window_id"]: item for item in _read_jsonl(base / "scale.jsonl")}
            temporal = {item["context"]["window_id"]: item for item in _read_jsonl(base / "temporal.jsonl")}
            by_window: dict[int, list[dict]] = {}
            for item in segmentation:
                by_window.setdefault(int(item["context"]["window_id"]), []).append(item)
            window_count = max(1, int(np.ceil(max(len(errors) - args.overlap, 1) / stride)))
            for window_id in range(window_count):
                start = window_id * stride; end = min(start + args.window_size - 1, max(len(errors) - 1, start))
                frame_slice = errors[start:end + 1]
                reference_slice = geometry_errors[start:end + 1] if geometry_errors.size else np.asarray([])
                regret = float(np.nanmean(frame_slice - reference_slice)) if frame_slice.size and reference_slice.size == frame_slice.size else float(np.nanmean(frame_slice)) if frame_slice.size else 0.0
                frame_metrics = by_window.get(window_id, [])
                final_metrics = [item["metrics"].get("final", {}) for item in frame_metrics]
                merge_metrics = [item["metrics"].get("merge", {}) for item in frame_metrics]
                lsr = max((float(item.get("largest_segment_ratio") or 0) for item in final_metrics), default=0.0)
                growth = max((float(item.get("largest_growth_ratio") or 0) for item in final_metrics), default=0.0)
                cross = max((float(item.get("cross_coarse_merge_ratio") or 0) for item in merge_metrics), default=0.0)
                scale_item = scale.get(window_id, {}).get("metrics", {})
                scale_p90 = ((scale_item.get("scale_log_mad_quantiles") or {}).get("p90") or 0.0)
                temporal_item = temporal.get(window_id, {}).get("metrics", {})
                records.append({
                    "config_id": config, "sequence_id": sequence, "window_id": window_id,
                    "frame_start": start, "frame_end": end, "trajectory_regret": regret,
                    "merge_anomaly": lsr + cross + np.log1p(growth), "scale_dispersion": float(scale_p90),
                    "temporal_churn": float(temporal_item.get("segment_churn_ratio") or temporal_item.get("unmatched_area_ratio") or 0),
                    "gt_speed": 0.0, "gt_turn": 0.0, "confidence": 0.0,
                })
    return records


def build_cases(run_dir: Path, intervals: list[SelectedInterval], records: list[dict], args) -> None:
    stride = args.window_size - args.overlap
    for interval in intervals:
        center = (interval.start_frame + interval.end_frame) // 2
        for config in DIAGNOSTIC_PROFILES:
            trace_dir = run_dir / "artifacts" / config / interval.sequence_id / "pass2" / "traces"
            if not trace_dir.exists():
                continue
            candidates = [trace_dir / f"inputs-frame-{center:06d}.npz", trace_dir / f"segmentation-frame-{center:06d}.npz"]
            window_id = center // stride
            candidates.append(trace_dir / f"scale-window-{window_id:06d}.npz")
            paths = [path for path in candidates if path.exists()]
            if not any("segmentation" in path.name for path in paths):
                available = sorted(trace_dir.glob("segmentation-frame-*.npz"))
                if available:
                    nearest = min(available, key=lambda path: abs(int(path.stem.rsplit("-", 1)[1]) - center))
                    paths.append(nearest)
                    input_path = trace_dir / nearest.name.replace("segmentation-", "inputs-")
                    if input_path.exists(): paths.append(input_path)
            if not paths:
                continue
            case_dir = run_dir / "cases" / interval.sequence_id / f"{interval.start_frame:06d}-{interval.end_frame:06d}" / config
            render_case(paths, case_dir)
            matching = [row for row in records if row["config_id"] == config and row["sequence_id"] == interval.sequence_id and row["frame_start"] <= center <= row["frame_end"]]
            metrics = {"interval": interval.to_dict(), "config_id": config, **(matching[0] if matching else {})}
            atomic_write_json(case_dir / "metrics.json", metrics)


def run_master(args, *, runner=subprocess.run) -> int:
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    info = preflight(args)
    manifest_path = output / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError("--resume requested but manifest.json is missing")
        manifest = _resume_manifest(manifest_path, info)
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{info['checkpoint_sha256'][:8]}"
        manifest = _manifest(args, info, run_id)
        atomic_write_json(manifest_path, manifest.to_dict())
    if args.report_only:
        print("[phase report] rebuilding report without inference")
        build_report(output)
        return 0
    estimated = sum(info["frame_counts"].values()) * len(DIAGNOSTIC_PROFILES) * 2048
    print(f"[phase preflight] 4 sequential profiles, {sum(info['frame_counts'].values())} frames, metrics estimate {estimated / 2**20:.1f} MiB")
    if args.dry_run:
        manifest.status = "dry_run"
        atomic_write_json(manifest_path, manifest.to_dict())
        print("[phase dry-run] no model inference executed")
        return 0
    lock = RunLock(output / ".diagnostic.lock", manifest.run_id)
    with lock:
        budget = StorageBudget(args.temp_root, max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib, min_free_gib=args.min_free_gib)
        try:
            for index, config in enumerate(DIAGNOSTIC_PROFILES, 1):
                if _phase_done(manifest, "pass1", config):
                    print(f"[phase pass1 {index}/4] {config}: resume skip")
                    continue
                print(f"[phase pass1 {index}/4] {config}: starting")
                budget.enforce()
                runner(_worker_command(args, pass_id=1, config_id=config, run_id=manifest.run_id), check=True)
                manifest.mark("pass1", config, "*", "complete")
                atomic_write_json(manifest_path, manifest.to_dict())
            trajectory = _read_trajectory_results(output)
            summary = build_sequence_summary(trajectory)
            layer_ates = {seq: value["ate_rmse"] for seq, value in trajectory.get("layer_atomic", {}).items() if value.get("valid")}
            summary["stability_guard"] = evaluate_stability_guard(layer_ates, layer_ates) if layer_ates else {"passed": False, "failure_reasons": ["missing_layer_atomic"]}
            summary["sequence_metrics"] = trajectory
            summary["recovery"] = summary.get("recovery_gap", {})
            atomic_write_json(output / "summary.json", summary)
            records = build_selection_records(output, trajectory, args)
            atomic_write_json(output / "selection_records.json", records)
            intervals = select_intervals(records, limit=args.max_selected)
            selected_path = output / "selected_intervals.json"
            atomic_write_json(selected_path, [item.to_dict() for item in intervals])
            selected_sequences = sorted({item.sequence_id for item in intervals})
            pass2_args = argparse.Namespace(**vars(args)); pass2_args.sequences = selected_sequences
            for index, config in enumerate(DIAGNOSTIC_PROFILES, 1):
                if not selected_sequences or _phase_done(manifest, "pass2", config):
                    continue
                print(f"[phase pass2 {index}/4] {config}: full rerun of {','.join(selected_sequences)}; dense writes selected only")
                budget.enforce()
                runner(_worker_command(pass2_args, pass_id=2, config_id=config, run_id=manifest.run_id, selected=selected_path), check=True)
                manifest.mark("pass2", config, "*", "complete")
                atomic_write_json(manifest_path, manifest.to_dict())
            print("[phase verify] rendering selected cases")
            build_cases(output, intervals, records, args)
            manifest.status = "complete"
            atomic_write_json(manifest_path, manifest.to_dict())
            print("[phase report] building offline HTML/CSV report")
            build_report(output)
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
    profile = DIAGNOSTIC_PROFILES[args.config_id]
    selected = []
    if args.selected_intervals:
        selected = [SelectedInterval.from_dict(item) for item in json.loads(Path(args.selected_intervals).read_text())]
    artifact_root = Path(args.output_dir) / "artifacts"
    artifact_budget = StorageBudget(Path(args.output_dir), max_gib=args.max_temp_gib, warn_gib=args.warn_temp_gib, min_free_gib=args.min_free_gib)
    sink = FileDiagnosticSink(artifact_root, selected_intervals=selected, budget=artifact_budget)
    model = _load_model(Path(args.model_ckpt), device)
    try:
        for sequence in args.sequences:
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
                diagnostic_sink=sink, diagnostic_run_id=args.run_id,
                diagnostic_sequence_id=sequence, diagnostic_pass=args.pass_id,
                cache_policy="metrics-only",
            )
            engine.begin()
            for window in engine.img_sliding_window(image_paths):
                engine(load_and_preprocess_images(window).to(device))
            engine.end()
            poses = engine.parse_pose_cache_summary()["extrinsic"][0].cpu().numpy()
            gt = _load_kitti_poses(Path(args.dataset_root) / "poses" / f"{sequence}.txt")
            evaluation = evaluate_trajectory(poses, gt)
            aligned = evaluation.pop("aligned_poses", None)
            target_root = Path(args.output_dir) / "trajectory"
            if args.pass_id != 1: target_root = target_root / f"pass{args.pass_id}"
            target = target_root / args.config_id; target.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target / f"{sequence}.json", evaluation)
            np.savez_compressed(target / f"{sequence}.npz", predicted=poses, ground_truth=gt[:len(poses)], aligned=np.asarray(aligned) if aligned is not None else np.empty(0))
            shutil.rmtree(cache_root, ignore_errors=True)
    finally:
        sink.close()
    return 0
