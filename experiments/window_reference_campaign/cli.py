"""Import-light command line entry points for campaign planning."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


def _parse_methods(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(","))
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError("methods must be comma-separated names")
    return items


def _parse_refinements(value: str) -> tuple[bool, ...]:
    result: list[bool] = []
    for item in (part.strip() for part in value.split(",")):
        if item == "off":
            result.append(False)
        elif item == "on":
            result.append(True)
        else:
            raise argparse.ArgumentTypeError("refinement values must be off or on")
    if not result:
        raise argparse.ArgumentTypeError("refinement must not be empty")
    return tuple(result)


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
    parser.add_argument("--resume", dest="resume", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fail-fast", dest="failure_policy", action="store_const", const="fail-fast")
    parser.add_argument("--keep-going", dest="failure_policy", action="store_const", const="keep-going")


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

    for name in ("run", "summarize"):
        command = commands.add_parser(name, help=f"{name} (added in a later task)")
        _add_common_options(command)
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


def _handle_plan(args: argparse.Namespace) -> int:
    # Keep all project imports below command dispatch so importing this module
    # remains safe in a process that intentionally blocks torch.
    previous_import_mode = os.environ.get("LASER_WINDOW_REFERENCE_IMPORT_LIGHT")
    if args.dry_run:
        os.environ["LASER_WINDOW_REFERENCE_IMPORT_LIGHT"] = "1"
    try:
        from .config import CachePolicy, CampaignOverrides, load_campaign_config
        from .matrix import build_plan, plan_payload
    finally:
        if previous_import_mode is None:
            os.environ.pop("LASER_WINDOW_REFERENCE_IMPORT_LIGHT", None)
        else:
            os.environ["LASER_WINDOW_REFERENCE_IMPORT_LIGHT"] = previous_import_mode

    methods = ()
    if args.methods is not None:
        methods = tuple(args.methods)
    refinements = args.refinement or ()
    overrides = CampaignOverrides(
        preset=args.preset,
        scene_ids=tuple(args.scene_ids),
        methods=methods,
        refinements=refinements,
        data_root=Path(args.data_root) if args.data_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        gpu=args.gpu,
        cache_policy=CachePolicy(args.cache_policy) if args.cache_policy else None,
        keep_artifacts=args.keep_artifacts,
    )
    loaded = load_campaign_config(Path(args.config), overrides)
    payload = plan_payload(build_plan(loaded), loaded)
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
        from .config import CachePolicy, CampaignOverrides, load_campaign_config
        from .matrix import build_plan
        from .preflight import preflight_campaign, write_preflight_report
    finally:
        if previous_import_mode is None:
            os.environ.pop("LASER_WINDOW_REFERENCE_IMPORT_LIGHT", None)
        else:
            os.environ["LASER_WINDOW_REFERENCE_IMPORT_LIGHT"] = previous_import_mode

    methods = tuple(args.methods) if args.methods is not None else ()
    overrides = CampaignOverrides(
        preset=args.preset,
        scene_ids=tuple(args.scene_ids),
        methods=methods,
        refinements=args.refinement or (),
        data_root=Path(args.data_root) if args.data_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        gpu=args.gpu,
        cache_policy=CachePolicy(args.cache_policy) if args.cache_policy else None,
        keep_artifacts=args.keep_artifacts,
    )
    loaded = load_campaign_config(Path(args.config), overrides)
    plan = build_plan(loaded)
    report = preflight_campaign(loaded, plan, allow_no_gpu=args.allow_no_gpu)
    target = write_preflight_report(report, loaded.config.campaign_root)
    print(f"{target} ({report.status})")
    return 0 if report.status in {"ok", "ok_with_warnings"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
    parser.print_help()
    return 2


__all__ = ["main"]
