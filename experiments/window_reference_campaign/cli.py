"""Import-light command line entry points for campaign planning."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


def _parse_methods(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(","))
    allowed = ("depth", "geometry", "atomic")
    if (
        not items
        or any(not item for item in items)
        or any(item not in allowed for item in items)
        or len(set(items)) != len(items)
    ):
        raise argparse.ArgumentTypeError(
            "methods must be a comma-separated subset of depth,geometry,atomic"
        )
    return tuple(item for item in allowed if item in items)


def _parse_refinements(value: str) -> tuple[bool, ...]:
    if value == "off":
        return (False,)
    if value == "on":
        return (True,)
    if value == "off,on":
        return (False, True)
    raise argparse.ArgumentTypeError("refinement must be exactly off, on, or off,on")


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/experiments/window_reference_campaign.yaml",
    )
    parser.add_argument("--preset", default="pointcloud-small")
    parser.add_argument("--scene", dest="scene_ids", action="append", default=[])
    parser.add_argument("--methods", type=_parse_methods)
    parser.add_argument("--refinement", type=_parse_refinements)
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-root")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--cache-policy", choices=("auto", "refresh", "readonly", "off"))
    parser.add_argument("--keep-artifacts", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=None)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    failure = parser.add_mutually_exclusive_group()
    failure.add_argument("--fail-fast", dest="failure_policy", action="store_const", const="fail-fast")
    failure.add_argument("--keep-going", dest="failure_policy", action="store_const", const="keep-going")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_window_reference_campaign.py")
    commands = parser.add_subparsers(dest="command")
    plan = commands.add_parser("plan", help="resolve a campaign matrix")
    _add_common_options(plan)

    bootstrap = commands.add_parser("bootstrap", help="bootstrap repository dependencies")
    bootstrap.add_argument(
        "--repository",
        default=str(Path(__file__).resolve().parents[2]),
    )
    bootstrap.add_argument("--checkpoint")
    bootstrap_mode = bootstrap.add_mutually_exclusive_group()
    bootstrap_mode.add_argument("--dry-run", action="store_true")
    bootstrap_mode.add_argument("--execute", action="store_true")

    preflight = commands.add_parser("preflight", help="validate campaign prerequisites")
    _add_common_options(preflight)
    preflight.add_argument("--allow-no-gpu", action="store_true")

    commands.add_parser("run", help="execute the planned campaign")
    commands.add_parser("summarize", help="validate compact records and write summaries")
    # Common options belong to both complete handlers; add them after parser
    # creation so the command descriptions above remain user-facing.
    _add_common_options(commands.choices["run"])
    _add_common_options(commands.choices["summarize"])
    return parser


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _campaign_overrides(arguments: argparse.Namespace):
    """Translate common parser values into the typed campaign boundary."""

    from .config import CachePolicy, CampaignOverrides

    return CampaignOverrides(
        preset=arguments.preset,
        scene_ids=tuple(arguments.scene_ids),
        methods=tuple(arguments.methods or ()),
        refinements=tuple(arguments.refinement or ()),
        data_root=Path(arguments.data_root) if arguments.data_root else None,
        output_root=Path(arguments.output_root) if arguments.output_root else None,
        checkpoint=Path(arguments.checkpoint) if arguments.checkpoint else None,
        start_frame=arguments.start_frame,
        max_frames=arguments.max_frames,
        frame_stride=arguments.frame_stride,
        gpu=arguments.gpu,
        cache_policy=(CachePolicy(arguments.cache_policy) if arguments.cache_policy else None),
        keep_artifacts=arguments.keep_artifacts,
    )


def _load_campaign(arguments: argparse.Namespace):
    from .config import load_campaign_config

    return load_campaign_config(Path(arguments.config), _campaign_overrides(arguments))


def _handle_plan(args: argparse.Namespace) -> int:
    # Keep all project imports below command dispatch so importing this module
    # remains safe in a process that intentionally blocks torch.
    previous_import_mode = os.environ.get("LASER_WINDOW_REFERENCE_IMPORT_LIGHT")
    # Every plan is a metadata-only operation.  In particular, publishing a
    # normal plan.json must not pull in the Torch-backed pipeline enum module.
    os.environ["LASER_WINDOW_REFERENCE_IMPORT_LIGHT"] = "1"
    try:
        from .config import load_campaign_config
        from .matrix import build_plan, plan_payload
    finally:
        if previous_import_mode is None:
            os.environ.pop("LASER_WINDOW_REFERENCE_IMPORT_LIGHT", None)
        else:
            os.environ["LASER_WINDOW_REFERENCE_IMPORT_LIGHT"] = previous_import_mode

    loaded = load_campaign_config(Path(args.config), _campaign_overrides(args))
    plan = build_plan(loaded)
    _reject_synthetic_frame_overrides(args, plan)
    payload = plan_payload(plan, loaded)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if args.dry_run:
        print(serialized)
        return 0
    target = loaded.config.campaign_root / "plan.json"
    _write_atomic(target, serialized + "\n")
    print(str(target))
    return 0


def _handle_bootstrap(args: argparse.Namespace) -> int:
    # Bootstrap is intentionally the only handler that can run before project
    # dependencies are installed.  Keep its imports below command dispatch.
    from .preflight import build_bootstrap_actions, run_bootstrap

    actions = build_bootstrap_actions(
        args.repository,
        python_executable=sys.executable,
        external_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )
    return run_bootstrap(actions, execute=bool(args.execute))


def _handle_preflight(args: argparse.Namespace) -> int:
    previous_import_mode = os.environ.get("LASER_WINDOW_REFERENCE_IMPORT_LIGHT")
    if args.dry_run:
        os.environ["LASER_WINDOW_REFERENCE_IMPORT_LIGHT"] = "1"
    try:
        from .config import load_campaign_config
        from .matrix import build_plan
        from .preflight import preflight_campaign, write_preflight_report
    finally:
        if previous_import_mode is None:
            os.environ.pop("LASER_WINDOW_REFERENCE_IMPORT_LIGHT", None)
        else:
            os.environ["LASER_WINDOW_REFERENCE_IMPORT_LIGHT"] = previous_import_mode

    loaded = load_campaign_config(Path(args.config), _campaign_overrides(args))
    plan = build_plan(loaded)
    _reject_synthetic_frame_overrides(args, plan)
    report = preflight_campaign(loaded, plan, allow_no_gpu=args.allow_no_gpu)
    target = write_preflight_report(report, loaded.config.campaign_root)
    print(f"{target} ({report.status})")
    return 0 if report.status in {"ok", "ok_with_warnings"} else 1


def _plan_is_synthetic(plan) -> bool:
    return bool(plan.runs) and all(item.dataset.value == "synthetic" for item in plan.runs)


def _reject_synthetic_frame_overrides(args: argparse.Namespace, plan) -> None:
    """Keep the literal synthetic fixture's four-frame identity immutable."""

    if not _plan_is_synthetic(plan):
        return
    if any(
        value is not None
        for value in (args.start_frame, args.max_frames, args.frame_stride)
    ):
        raise ValueError(
            "synthetic preset uses the fixed four-frame selection; "
            "start/max-frames/frame-stride overrides are not supported"
        )


def _resolve_scenes(loaded, plan):
    from .scenes import resolve_scene

    selected = {item.scene_id: item for item in loaded.config.selected_scenes}
    return {
        scene_id: resolve_scene(loaded.config, selected[scene_id])
        for scene_id in {item.scene_id for item in plan.runs}
    }


def _runtime_payload(*, synthetic: bool, gpu: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    if synthetic:
        return payload
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        payload["torch"] = getattr(torch, "__version__", None)
        payload["cuda"] = {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
        payload["gpu"] = gpu
    except Exception:
        payload["torch"] = None
        payload["cuda"] = None
        payload["gpu"] = None
    return payload


def _campaign_metadata(
    loaded,
    *,
    synthetic: bool,
    status: str,
    argv: Sequence[str],
    preflight,
) -> dict[str, object]:
    source_commit = preflight.git.commit
    source_dirty = preflight.git.dirty
    checkpoint_sha256 = None
    if not synthetic:
        from inference_engine.prediction_cache.fingerprint import sha256_file

        checkpoint_sha256 = sha256_file(loaded.config.storage.checkpoint)
    return {
        "schema_version": 1,
        "campaign_id": loaded.config.campaign_id,
        "preset": loaded.config.selected_preset,
        "argv": list(argv),
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "config_sha256": loaded.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "preflight": preflight.to_payload(),
        "runtime": _runtime_payload(synthetic=synthetic, gpu=loaded.config.runtime.gpu),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "status": status,
    }


def _update_campaign_metadata(path: Path, *, status: str, exit_code: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update({
        "status": status,
        "exit_code": exit_code,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    from .results import write_campaign_metadata

    write_campaign_metadata(path, payload)


def _handle_run(args: argparse.Namespace) -> int:
    if args.dry_run and args.resume is False:
        raise ValueError("--dry-run cannot be combined with --no-resume")
    if args.dry_run and (
        args.keep_artifacts is True or args.failure_policy is not None
    ):
        raise ValueError(
            "--dry-run cannot be combined with --keep-artifacts or failure-policy flags"
        )
    loaded = _load_campaign(args)
    from .matrix import build_plan, plan_payload

    plan = build_plan(loaded)
    _reject_synthetic_frame_overrides(args, plan)
    if args.dry_run:
        print(json.dumps(plan_payload(plan, loaded), sort_keys=True, separators=(",", ":")))
        return 0

    synthetic = _plan_is_synthetic(plan)
    from .preflight import preflight_campaign

    report = preflight_campaign(loaded, plan, allow_no_gpu=synthetic)
    if report.status == "error":
        for error in report.errors:
            print(error, file=sys.stderr)
        return 1
    scenes = _resolve_scenes(loaded, plan)
    root = loaded.config.campaign_root
    if root.is_symlink():
        raise ValueError("campaign root must not be a symlink")
    if root.exists() and not root.is_dir():
        raise ValueError("campaign root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "campaign.json"
    from .results import write_campaign_metadata

    write_campaign_metadata(
        metadata_path,
        _campaign_metadata(
            loaded,
            synthetic=synthetic,
            status="running",
            argv=tuple(getattr(args, "_argv", tuple(sys.argv[1:]))),
            preflight=report,
        ),
    )
    try:
        from .config import FailurePolicy
        from .runner import run_campaign

        resume = bool(args.resume) if args.resume is not None else False
        policy = FailurePolicy(args.failure_policy or loaded.config.runtime.failure_policy.value)
        outcome = run_campaign(
            loaded,
            plan,
            scenes,
            resume=resume,
            failure_policy=policy,
            keep_artifacts=(
                bool(args.keep_artifacts)
                if args.keep_artifacts is not None
                else loaded.config.storage.keep_artifacts
            ),
            source_state=(report.git.commit, report.git.dirty),
        )
    except Exception as exc:
        _update_campaign_metadata(metadata_path, status="failed", exit_code=1)
        print(f"campaign run failed: {exc}", file=sys.stderr)
        return 1
    _update_campaign_metadata(
        metadata_path,
        status="succeeded" if outcome.exit_code == 0 else "failed",
        exit_code=outcome.exit_code,
    )
    print(str(root / "summary"))
    return outcome.exit_code


def _expected_summary_seeds(loaded, plan):
    from .matrix import build_identity_seed, identity_slice_id
    from .runner import _source_metadata
    from .scenes import resolve_scene
    from .staging import preview_staging_manifest
    from .synthetic import synthetic_checkpoint_sha256

    commit, dirty = _source_metadata(loaded.config.repository_root)
    synthetic = _plan_is_synthetic(plan)
    checkpoint = (
        synthetic_checkpoint_sha256()
        if synthetic
        else __import__("inference_engine.prediction_cache.fingerprint", fromlist=["sha256_file"]).sha256_file(
            loaded.config.storage.checkpoint
        )
    )
    selected = {item.scene_id: item for item in loaded.config.selected_scenes}
    scene_info = {}
    for scene_id, selected_scene in selected.items():
        if scene_id not in {item.scene_id for item in plan.runs}:
            continue
        resolved = resolve_scene(loaded.config, selected_scene)
        if synthetic:
            from .runner import synthetic_staging_manifest_sha256

            manifest_sha = synthetic_staging_manifest_sha256(resolved)
        else:
            _, manifest_sha = preview_staging_manifest(resolved)
        scene_info[scene_id] = (resolved, manifest_sha)
    expected: dict[tuple[str, str, str, str], object] = {}
    for planned in plan.runs:
        resolved, manifest_sha = scene_info[planned.scene_id]
        seed = build_identity_seed(
            loaded=loaded,
            planned=planned,
            frame_start=resolved.selection.start,
            frame_stop=resolved.selection.stop,
            frame_stride=resolved.selection.stride,
            staged_manifest_sha256=manifest_sha,
            source_commit=commit,
            source_dirty=dirty,
            checkpoint_sha256=checkpoint,
        )
        key = (
            planned.dataset.value,
            planned.scene,
            identity_slice_id(
                resolved.selection.start,
                resolved.selection.stop,
                resolved.selection.stride,
            ),
            planned.variant.run_id,
        )
        expected[key] = seed
    return expected


def _handle_summarize(args: argparse.Namespace) -> int:
    loaded = _load_campaign(args)
    from .matrix import build_plan
    from .results import (
        RunStatus,
        load_valid_completed_run,
        read_run_record,
        write_summaries,
    )
    from .runner import validate_artifact_for_seed

    plan = build_plan(loaded)
    _reject_synthetic_frame_overrides(args, plan)
    try:
        expected = _expected_summary_seeds(loaded, plan)
        root = loaded.config.campaign_root
        records = []
        invalid = False
        for key, seed in expected.items():
            planned = next(
                item
                for item in plan.runs
                if (
                    item.dataset.value,
                    item.scene,
                    item.variant.run_id,
                ) == (key[0], key[1], key[3])
            )
            path = root / planned.relative_run_dir / "run.json"
            if not path.is_file():
                continue
            record = read_run_record(path)
            if record.status is RunStatus.SUCCEEDED:
                record = load_valid_completed_run(path, seed)
                artifact = root / planned.relative_run_dir / "artifact"
                if artifact.is_dir():
                    validated_key, digest = validate_artifact_for_seed(artifact, seed)
                    if record.identity is None or validated_key != record.identity.prediction_key:
                        raise ValueError("summary artifact prediction key mismatch")
                    if record.artifact_manifest_sha256 != digest:
                        raise ValueError("summary artifact digest mismatch")
            elif record.identity_seed != seed:
                raise ValueError("summary failed record identity does not match expected seed")
            records.append(record)
        paths = write_summaries(records, expected, root / "summary")
        from .results import atomic_json

        atomic_json(
            root / "summary.json",
            json.loads(paths.summary_json.read_text(encoding="utf-8")),
        )
        print("\n".join(str(path) for path in (
            paths.runs_csv,
            paths.diagnostics_csv,
            paths.trajectory_csv,
            paths.pointcloud_csv,
            paths.summary_json,
            paths.failures_json,
        )))
        summary_payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
        invalid = bool(summary_payload.get("missing_runs") or summary_payload.get("failed_runs"))
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"campaign summary failed: {exc}", file=sys.stderr)
        return 1
    return 1 if invalid else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args._argv = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    if args.command == "plan":
        try:
            return _handle_plan(args)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    if args.command == "bootstrap":
        try:
            return _handle_bootstrap(args)
        except (OSError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
    if args.command == "preflight":
        try:
            return _handle_preflight(args)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    if args.command == "run":
        try:
            return _handle_run(args)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    if args.command == "summarize":
        try:
            return _handle_summarize(args)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    parser.print_help()
    return 2


__all__ = ["main"]
