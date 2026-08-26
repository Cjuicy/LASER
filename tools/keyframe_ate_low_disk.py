#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ate import evaluate_ate_inputs
from experiments.config import (
    CapabilityExperimentConfig,
    EvaluationKind,
    load_experiment_config,
)
from experiments.matrix import CapabilityMatrixEntry, reconstruction_identity
from inference_engine.prediction_cache.fingerprint import sha256_file
from pipeline.config import LoadedPipelineConfig, load_pipeline_config
from pipeline.artifacts import load_trajectory_estimate
from pipeline.manifest import discover_image_manifest
from pipeline.runner import PipelineRunner


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_URL_CREDENTIALS_PATTERN = re.compile(r"(://)[^/@\s]+@")
_ATTEMPT_PATTERN = re.compile(r"attempt-[0-9]{4,}\Z")


@dataclass(frozen=True)
class LowDiskEntryResult:
    entry: CapabilityMatrixEntry
    status: str
    resumed: bool
    run_identity: str | None
    reconstruction_identity: str | None
    artifact_dir: Path
    metrics_path: Path
    error: dict[str, str] | None = None


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _runtime_source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = {Path(__file__).resolve()}
    for directory_name in (
        "evaluation",
        "experiments",
        "inference_engine",
        "loop_closure",
        "pipeline",
        "pi3",
        "reconstruction",
        "utils",
    ):
        directory = root / directory_name
        for suffix in ("*.py", "*.pyx"):
            sources.update(path.resolve() for path in directory.rglob(suffix))
    sources.add((root / "loop_closure" / "fastloop" / "solve.cpp").resolve())
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda path: path.as_posix()):
        label = source.relative_to(root).as_posix().encode("utf-8")
        content = source.read_bytes()
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                payload,
                temporary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def _validated_scratch_path(
    path: str | Path,
    *,
    allowed_root: str | Path,
) -> Path:
    target = Path(path)
    root = Path(allowed_root).resolve()
    lexical_target = Path(os.path.abspath(target))
    try:
        lexical_relative = lexical_target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing cleanup outside allowed root: {target}") from exc
    current = root
    for part in lexical_relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"refusing cleanup through symbolic link: {target}")
    resolved = target.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing cleanup outside allowed root: {target}") from exc
    if not relative.parts:
        raise ValueError(f"refusing cleanup of allowed root itself: {target}")
    return target


def _validated_cleanup_target(
    path: str | Path,
    *,
    allowed_root: str | Path,
) -> Path | None:
    target = Path(path)
    if not target.exists() and not target.is_symlink():
        return None
    _validated_scratch_path(target, allowed_root=allowed_root)
    if not target.is_dir():
        raise ValueError(f"cleanup target is not a directory: {target}")
    return target


def _safe_remove_directory(
    path: str | Path,
    *,
    allowed_root: str | Path,
) -> None:
    target = _validated_cleanup_target(path, allowed_root=allowed_root)
    if target is None:
        return
    shutil.rmtree(target)


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _valid_metrics(
    path: Path,
    *,
    artifact_manifest_sha256: str | None = None,
) -> bool:
    payload = _load_json(path)
    if payload is None:
        return False
    numeric_names = (
        "ate_rmse_m",
        "rpe_translation_rmse_m",
        "rpe_rotation_rmse_deg",
    )
    for name in numeric_names:
        value = payload.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return False
    matched = payload.get("matched_frame_count")
    if isinstance(matched, bool) or not isinstance(matched, int) or matched < 2:
        return False
    manifest_digest = payload.get("artifact_manifest_sha256")
    if (
        not isinstance(manifest_digest, str)
        or _SHA256_PATTERN.fullmatch(manifest_digest) is None
    ):
        return False
    return (
        artifact_manifest_sha256 is None or manifest_digest == artifact_manifest_sha256
    )


def _valid_resume(record_path: Path, metrics_path: Path, run_identity: str) -> bool:
    record = _load_json(record_path)
    if (
        record is None
        or record.get("schema_version") != 1
        or record.get("status") != "passed"
        or record.get("run_identity") != run_identity
        or not _valid_metrics(metrics_path)
    ):
        return False
    metrics_digest = record.get("metrics_sha256")
    return (
        isinstance(metrics_digest, str)
        and _SHA256_PATTERN.fullmatch(metrics_digest) is not None
        and metrics_digest == sha256_file(metrics_path)
    )


def _recorded_artifact(
    record_path: Path,
    *,
    expected_status: str,
    entry: CapabilityMatrixEntry,
    run_identity: str,
    scratch_root: Path,
) -> Path | None:
    record = _load_json(record_path)
    if (
        record is None
        or record.get("status") != expected_status
        or record.get("run_identity") != run_identity
        or not isinstance(record.get("artifact_dir"), str)
    ):
        return None
    candidate = Path(record["artifact_dir"])
    try:
        _validate_attempt_artifact(
            candidate,
            scratch_root=scratch_root,
            entry_name=entry.name,
            run_identity=run_identity,
        )
    except ValueError:
        return None
    if not _valid_attempt_record(
        candidate,
        expected_status=(
            ("evaluated", "passed") if expected_status == "passed" else expected_status
        ),
        entry_name=entry.name,
        run_identity=run_identity,
    ):
        return None
    return candidate


def _validate_attempt_artifact(
    artifact_dir: str | Path,
    *,
    scratch_root: Path,
    entry_name: str,
    run_identity: str,
) -> Path:
    candidate = _validated_scratch_path(
        artifact_dir,
        allowed_root=scratch_root,
    )
    relative = Path(os.path.abspath(candidate)).relative_to(scratch_root.resolve())
    if (
        len(relative.parts) != 4
        or relative.parts[0] != entry_name
        or relative.parts[1] != run_identity
        or _SHA256_PATTERN.fullmatch(relative.parts[1]) is None
        or _ATTEMPT_PATTERN.fullmatch(relative.parts[2]) is None
        or relative.parts[3] != "artifact"
    ):
        raise ValueError(f"artifact path is not canonical: {candidate}")
    return candidate


def _valid_attempt_record(
    artifact_dir: Path,
    *,
    expected_status: str | tuple[str, ...] | None,
    entry_name: str,
    run_identity: str,
) -> bool:
    record = _load_json(_attempt_record_path(artifact_dir))
    if (
        record is None
        or record.get("schema_version") != 1
        or record.get("entry") != entry_name
        or record.get("run_identity") != run_identity
        or not isinstance(record.get("artifact_dir"), str)
        or Path(os.path.abspath(record["artifact_dir"]))
        != Path(os.path.abspath(artifact_dir))
    ):
        return False
    if expected_status is None:
        return True
    statuses = (
        (expected_status,) if isinstance(expected_status, str) else expected_status
    )
    return record.get("status") in statuses


def _next_attempt_artifact(entry_scratch: Path, run_identity: str) -> Path:
    identity_root = entry_scratch / run_identity
    attempt_index = 1
    while (identity_root / f"attempt-{attempt_index:04d}").exists():
        attempt_index += 1
    return identity_root / f"attempt-{attempt_index:04d}" / "artifact"


def _attempt_record_path(artifact_dir: Path) -> Path:
    return artifact_dir.parent / "attempt_record.json"


def _write_attempt_record(
    artifact_dir: Path,
    *,
    status: str,
    entry: CapabilityMatrixEntry,
    run_identity: str,
    reconstruction_digest: str | None,
    error: dict[str, str] | None = None,
    recorded_at_ns: int | None = None,
) -> Path:
    return _atomic_json(
        _attempt_record_path(artifact_dir),
        {
            "schema_version": 1,
            "status": status,
            "entry": entry.name,
            "run_identity": run_identity,
            "reconstruction_identity": reconstruction_digest,
            "artifact_dir": str(artifact_dir),
            "recorded_at_ns": (
                time.time_ns() if recorded_at_ns is None else recorded_at_ns
            ),
            "error": error,
        },
    )


def _atomic_copy_file(source: Path, target: Path) -> None:
    temporary_path: Path | None = None
    try:
        with source.open("rb") as source_stream, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(source_stream, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _copy_failure_metadata(
    artifact_dir: Path,
    *,
    scratch_root: Path,
) -> None:
    _copy_failure_metadata_to(
        artifact_dir,
        metadata_root=artifact_dir.parent / "evicted_artifact_metadata",
        scratch_root=scratch_root,
    )


def _copy_failure_metadata_to(
    artifact_dir: Path,
    *,
    metadata_root: Path,
    scratch_root: Path,
) -> None:
    _validated_scratch_path(artifact_dir, allowed_root=scratch_root)
    _validated_scratch_path(metadata_root, allowed_root=scratch_root)
    metadata_root.mkdir(parents=False, exist_ok=True)
    _validated_scratch_path(metadata_root, allowed_root=scratch_root)
    for name in (
        "manifest.json",
        "diagnostics.json",
        "resolved_reconstruction.yaml",
    ):
        source = artifact_dir / name
        if source.is_file() and not source.is_symlink():
            _atomic_copy_file(source, metadata_root / name)


def _iter_attempt_artifacts(
    scratch_root: Path,
    *,
    allowed_entry_names: frozenset[str],
):
    for artifact_dir in scratch_root.glob("*/*/attempt-*/artifact"):
        try:
            relative = Path(os.path.abspath(artifact_dir)).relative_to(
                scratch_root.resolve()
            )
        except ValueError:
            continue
        if (
            len(relative.parts) != 4
            or relative.parts[0] not in allowed_entry_names
            or _SHA256_PATTERN.fullmatch(relative.parts[1]) is None
            or _ATTEMPT_PATTERN.fullmatch(relative.parts[2]) is None
            or relative.parts[3] != "artifact"
        ):
            continue
        try:
            canonical = _validate_attempt_artifact(
                artifact_dir,
                scratch_root=scratch_root,
                entry_name=relative.parts[0],
                run_identity=relative.parts[1],
            )
        except ValueError:
            continue
        if canonical.is_dir() and not canonical.is_symlink():
            yield relative.parts[0], relative.parts[1], canonical


def _recover_interrupted_attempts(
    scratch_root: Path,
    *,
    entries_by_name: dict[str, CapabilityMatrixEntry],
) -> None:
    allowed = frozenset(entries_by_name)
    for entry_name, run_identity, artifact_dir in _iter_attempt_artifacts(
        scratch_root,
        allowed_entry_names=allowed,
    ):
        if _valid_attempt_record(
            artifact_dir,
            expected_status=None,
            entry_name=entry_name,
            run_identity=run_identity,
        ):
            continue
        _write_attempt_record(
            artifact_dir,
            status="failed",
            entry=entries_by_name[entry_name],
            run_identity=run_identity,
            reconstruction_digest=None,
            error={
                "exception_type": "InterruptedAttempt",
                "message": "recovered artifact without a valid attempt record",
            },
            recorded_at_ns=artifact_dir.stat().st_mtime_ns,
        )


def _find_reusable_artifact(
    entry_scratch: Path,
    *,
    entry: CapabilityMatrixEntry,
    run_identity: str,
    scratch_root: Path,
    artifact_validator: Callable[[Path], bool],
) -> Path | None:
    candidates = sorted(
        entry_scratch.glob(f"{run_identity}/attempt-*/artifact"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for artifact_dir in candidates:
        try:
            _validate_attempt_artifact(
                artifact_dir,
                scratch_root=scratch_root,
                entry_name=entry.name,
                run_identity=run_identity,
            )
        except ValueError:
            continue
        if _valid_attempt_record(
            artifact_dir,
            expected_status=("failed", "evaluated"),
            entry_name=entry.name,
            run_identity=run_identity,
        ) and artifact_validator(artifact_dir):
            return artifact_dir
    return None


def _artifact_ready_for_ate(artifact_dir: Path) -> bool:
    try:
        load_trajectory_estimate(artifact_dir)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _enforce_failed_artifact_limit(
    scratch_root: Path,
    *,
    max_retained_failures: int,
    protected_artifact: Path | None,
    allowed_entry_names: frozenset[str],
) -> None:
    candidates: list[tuple[int, Path, Path, dict[str, object]]] = []
    for entry_name, run_identity, artifact_dir in _iter_attempt_artifacts(
        scratch_root,
        allowed_entry_names=allowed_entry_names,
    ):
        record_path = _attempt_record_path(artifact_dir)
        record = _load_json(record_path)
        if (
            record is None
            or record.get("status") not in {"failed", "evaluated"}
            or not isinstance(record.get("recorded_at_ns"), int)
            or not _valid_attempt_record(
                artifact_dir,
                expected_status=("failed", "evaluated"),
                entry_name=entry_name,
                run_identity=run_identity,
            )
        ):
            continue
        candidates.append((record["recorded_at_ns"], record_path, artifact_dir, record))
    protected = None if protected_artifact is None else protected_artifact.resolve()
    removable = sorted(
        (
            candidate
            for candidate in candidates
            if protected is None or candidate[2].resolve() != protected
        ),
        key=lambda candidate: (candidate[0], candidate[1].as_posix()),
        reverse=True,
    )
    protected_count = int(
        protected is not None
        and any(candidate[2].resolve() == protected for candidate in candidates)
    )
    keep_older = max(0, max_retained_failures - protected_count)
    for _, record_path, artifact_dir, record in removable[keep_older:]:
        metadata_errors: list[dict[str, str]] = []
        metadata_root = artifact_dir.parent / "evicted_artifact_metadata"
        try:
            _validated_cleanup_target(
                artifact_dir,
                allowed_root=scratch_root,
            )
            try:
                _copy_failure_metadata(
                    artifact_dir,
                    scratch_root=scratch_root,
                )
            except Exception as error:
                metadata_errors.append(_safe_error(error))
                fallback_root = (
                    scratch_root
                    / "evicted-failure-metadata"
                    / _canonical_digest({"artifact_dir": str(artifact_dir.resolve())})
                )
                try:
                    fallback_root.parent.mkdir(parents=False, exist_ok=True)
                    _copy_failure_metadata_to(
                        artifact_dir,
                        metadata_root=fallback_root,
                        scratch_root=scratch_root,
                    )
                    metadata_root = fallback_root
                except Exception as fallback_error:
                    metadata_errors.append(_safe_error(fallback_error))
                    _atomic_json(
                        record_path,
                        {
                            **record,
                            "metadata_errors": metadata_errors,
                        },
                    )
                    continue
            _safe_remove_directory(
                artifact_dir,
                allowed_root=scratch_root,
            )
        except (OSError, ValueError):
            continue
        _atomic_json(
            record_path,
            {
                **record,
                "status": "artifact_evicted",
                "artifact_retained": False,
                "metadata_path": str(metadata_root),
                "metadata_errors": metadata_errors,
            },
        )


def _safe_error(error: Exception) -> dict[str, str]:
    message = " ".join(str(error).split())
    message = _URL_CREDENTIALS_PATTERN.sub(r"\1<redacted>@", message)
    return {
        "exception_type": type(error).__name__,
        "message": message[:2000],
    }


def _entry_overrides(
    entry: CapabilityMatrixEntry,
    *,
    cache_root: Path,
) -> tuple[str, ...]:
    return (
        f"segmentation.method={entry.segmentation_method.value}",
        "segmentation.window_reference.enabled="
        f"{str(entry.window_reference_enabled).lower()}",
        f"reconstruction.mode={entry.reconstruction_mode.value}",
        f"prediction_cache.root={cache_root}",
        "prediction_cache.mode=auto",
    )


def _entry_run_identity(
    *,
    experiment: CapabilityExperimentConfig,
    entry: CapabilityMatrixEntry,
    reconstruction_digest: str,
    evaluation_config_digest: str,
    ground_truth_digest: str,
    source_digest: str,
) -> str:
    trajectory_input = experiment.evaluator_inputs[EvaluationKind.ATE]
    return _canonical_digest(
        {
            "schema": "laser-low-disk-ate-v1",
            "entry": entry.name,
            "reconstruction_identity": reconstruction_digest,
            "dataset_name": experiment.dataset_name,
            "sequence": experiment.sequence,
            "evaluation_config_sha256": evaluation_config_digest,
            "ground_truth_sha256": ground_truth_digest,
            "ground_truth_format": trajectory_input.ground_truth_format,
            "runtime_source_sha256": source_digest,
        }
    )


def _result_payload(result: LowDiskEntryResult) -> dict[str, object]:
    return {
        "entry": result.entry.name,
        "status": result.status,
        "resumed": result.resumed,
        "run_identity": result.run_identity,
        "reconstruction_identity": result.reconstruction_identity,
        "artifact_dir": str(result.artifact_dir),
        "metrics_path": str(result.metrics_path),
        "error": result.error,
    }


def run_low_disk_ate(
    experiment: CapabilityExperimentConfig,
    *,
    work_root: str | Path,
    runner_factory: Callable = PipelineRunner,
    evaluator: Callable = evaluate_ate_inputs,
    source_digest_provider: Callable[[], str] = _runtime_source_digest,
    config_loader: Callable = load_pipeline_config,
    artifact_validator: Callable[[Path], bool] = _artifact_ready_for_ate,
    max_retained_failures: int = 1,
) -> tuple[LowDiskEntryResult, ...]:
    if not isinstance(experiment, CapabilityExperimentConfig):
        raise ValueError("low-disk ATE runner requires a version-2 config")
    if EvaluationKind.ATE not in experiment.evaluator_inputs:
        raise ValueError("low-disk ATE runner requires trajectory ground truth")
    if (
        isinstance(max_retained_failures, bool)
        or not isinstance(max_retained_failures, int)
        or max_retained_failures < 1
    ):
        raise ValueError("max_retained_failures must be a positive integer")

    root = Path(work_root).resolve()
    scratch_root = root / "scratch"
    metrics_root = root / "metrics"
    cache_root = root / "prediction-cache"
    scratch_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    entries = experiment.entries
    entries_by_name = {entry.name: entry for entry in entries}
    allowed_entry_names = frozenset(entries_by_name)
    _recover_interrupted_attempts(
        scratch_root,
        entries_by_name=entries_by_name,
    )
    _enforce_failed_artifact_limit(
        scratch_root,
        max_retained_failures=max_retained_failures,
        protected_artifact=None,
        allowed_entry_names=allowed_entry_names,
    )

    source_digest = source_digest_provider()
    if (
        not isinstance(source_digest, str)
        or _SHA256_PATTERN.fullmatch(source_digest) is None
    ):
        raise ValueError("source digest provider must return SHA256 hex")

    trajectory_input = experiment.evaluator_inputs[EvaluationKind.ATE]
    evaluation_config_digest = sha256_file(trajectory_input.config_path)
    ground_truth_digest = sha256_file(trajectory_input.ground_truth_path)
    manifest_digests: dict[tuple[str, int], str] = {}
    checkpoint_digests: dict[str, str] = {}
    results: list[LowDiskEntryResult] = []

    for entry_index, entry in enumerate(entries, start=1):
        entry_scratch = scratch_root / entry.name
        entry_metrics = metrics_root / entry.name
        metrics_path = entry_metrics / "trajectory_metrics.json"
        record_path = entry_metrics / "low_disk_record.json"
        run_identity: str | None = None
        reconstruction_digest: str | None = None
        artifact_dir = entry_scratch / "unidentified" / "artifact"
        try:
            loaded: LoadedPipelineConfig = config_loader(
                experiment.reconstruction_config,
                _entry_overrides(entry, cache_root=cache_root),
            )
            manifest_key = (
                str(Path(loaded.config.input.image_dir).resolve()),
                loaded.config.input.sample_stride,
            )
            if manifest_key not in manifest_digests:
                manifest = discover_image_manifest(*manifest_key)
                manifest_digests[manifest_key] = _manifest_digest(manifest.paths)
            checkpoint_path = str(Path(loaded.config.model.checkpoint).resolve())
            if checkpoint_path not in checkpoint_digests:
                checkpoint_digests[checkpoint_path] = sha256_file(checkpoint_path)
            reconstruction_digest = reconstruction_identity(
                loaded,
                input_manifest_sha256=manifest_digests[manifest_key],
                checkpoint_sha256=checkpoint_digests[checkpoint_path],
            )
            run_identity = _entry_run_identity(
                experiment=experiment,
                entry=entry,
                reconstruction_digest=reconstruction_digest,
                evaluation_config_digest=evaluation_config_digest,
                ground_truth_digest=ground_truth_digest,
                source_digest=source_digest,
            )

            if _valid_resume(record_path, metrics_path, run_identity):
                recorded_artifact = _recorded_artifact(
                    record_path,
                    expected_status="passed",
                    entry=entry,
                    run_identity=run_identity,
                    scratch_root=scratch_root,
                )
                if recorded_artifact is not None:
                    artifact_dir = recorded_artifact
                    _safe_remove_directory(
                        artifact_dir,
                        allowed_root=scratch_root,
                    )
                result = LowDiskEntryResult(
                    entry=entry,
                    status="passed",
                    resumed=True,
                    run_identity=run_identity,
                    reconstruction_identity=reconstruction_digest,
                    artifact_dir=artifact_dir,
                    metrics_path=metrics_path,
                )
                results.append(result)
                print(
                    f"[{entry_index}/{len(entries)}] RESUME {entry.name}",
                    flush=True,
                )
                _enforce_failed_artifact_limit(
                    scratch_root,
                    max_retained_failures=max_retained_failures,
                    protected_artifact=None,
                    allowed_entry_names=allowed_entry_names,
                )
                continue

            failed_artifact = _find_reusable_artifact(
                entry_scratch,
                entry=entry,
                run_identity=run_identity,
                scratch_root=scratch_root,
                artifact_validator=artifact_validator,
            )
            if failed_artifact is not None:
                artifact_dir = failed_artifact
                print(
                    f"[{entry_index}/{len(entries)}] RETRY-ATE {entry.name}",
                    flush=True,
                )
            else:
                artifact_dir = _next_attempt_artifact(
                    entry_scratch,
                    run_identity,
                )
                print(
                    f"[{entry_index}/{len(entries)}] RUN {entry.name}",
                    flush=True,
                )
                runner = runner_factory(loaded, artifact_output_dir=artifact_dir)
                runner.run()

            manifest_path = artifact_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"reconstruction manifest does not exist: {manifest_path}"
                )
            artifact_manifest_digest = sha256_file(manifest_path)
            produced_metrics = Path(
                evaluator(
                    artifact_dir,
                    ground_truth=trajectory_input.ground_truth_path,
                    ground_truth_format=trajectory_input.ground_truth_format,
                    evaluation_config=trajectory_input.config_path,
                    output_dir=entry_metrics,
                )
            )
            if produced_metrics.resolve() != metrics_path.resolve():
                raise ValueError(
                    "ATE evaluator returned an unexpected metrics path: "
                    f"{produced_metrics}"
                )
            if not _valid_metrics(
                metrics_path,
                artifact_manifest_sha256=artifact_manifest_digest,
            ):
                raise ValueError("ATE evaluator produced invalid metrics")
            _write_attempt_record(
                artifact_dir,
                status="evaluated",
                entry=entry,
                run_identity=run_identity,
                reconstruction_digest=reconstruction_digest,
            )
            _atomic_json(
                record_path,
                {
                    "schema_version": 1,
                    "status": "passed",
                    "entry": entry.name,
                    "run_identity": run_identity,
                    "reconstruction_identity": reconstruction_digest,
                    "artifact_dir": str(artifact_dir),
                    "metrics_sha256": sha256_file(metrics_path),
                },
            )
            _write_attempt_record(
                artifact_dir,
                status="passed",
                entry=entry,
                run_identity=run_identity,
                reconstruction_digest=reconstruction_digest,
            )
            _safe_remove_directory(artifact_dir, allowed_root=scratch_root)
            result = LowDiskEntryResult(
                entry=entry,
                status="passed",
                resumed=False,
                run_identity=run_identity,
                reconstruction_identity=reconstruction_digest,
                artifact_dir=artifact_dir,
                metrics_path=metrics_path,
            )
            print(f"[{entry_index}/{len(entries)}] PASS {entry.name}", flush=True)
        except Exception as error:
            error_payload = _safe_error(error)
            if (
                run_identity is not None
                and reconstruction_digest is not None
                and artifact_dir.is_dir()
                and not artifact_dir.is_symlink()
            ):
                try:
                    _write_attempt_record(
                        artifact_dir,
                        status="failed",
                        entry=entry,
                        run_identity=run_identity,
                        reconstruction_digest=reconstruction_digest,
                        error=error_payload,
                    )
                    _enforce_failed_artifact_limit(
                        scratch_root,
                        max_retained_failures=max_retained_failures,
                        protected_artifact=artifact_dir,
                        allowed_entry_names=allowed_entry_names,
                    )
                except Exception as retention_error:
                    retention = _safe_error(retention_error)
                    error_payload["retention_error"] = (
                        f"{retention['exception_type']}: {retention['message']}"
                    )
            _atomic_json(
                record_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "entry": entry.name,
                    "run_identity": run_identity,
                    "reconstruction_identity": reconstruction_digest,
                    "artifact_dir": str(artifact_dir),
                    "error": error_payload,
                },
            )
            result = LowDiskEntryResult(
                entry=entry,
                status="failed",
                resumed=False,
                run_identity=run_identity,
                reconstruction_identity=reconstruction_digest,
                artifact_dir=artifact_dir,
                metrics_path=metrics_path,
                error=error_payload,
            )
            print(
                f"[{entry_index}/{len(entries)}] FAIL {entry.name}: "
                f"{error_payload['exception_type']}: {error_payload['message']}",
                flush=True,
            )
        results.append(result)
        if result.status == "passed":
            _enforce_failed_artifact_limit(
                scratch_root,
                max_retained_failures=max_retained_failures,
                protected_artifact=None,
                allowed_entry_names=allowed_entry_names,
            )

    result_tuple = tuple(results)
    _atomic_json(
        root / "summary.json",
        {
            "schema_version": 1,
            "dataset_name": experiment.dataset_name,
            "sequence": experiment.sequence,
            "passed": sum(result.status == "passed" for result in result_tuple),
            "failed": sum(result.status == "failed" for result in result_tuple),
            "resumed": sum(result.resumed for result in result_tuple),
            "entries": [_result_payload(result) for result in result_tuple],
        },
    )
    return result_tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run keyframe-aware ATE with bounded artifact storage."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument(
        "--max-retained-failures",
        type=int,
        default=1,
        help=(
            "maximum failed reconstruction artifacts to retain after each "
            "entry (default: 1)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    experiment = load_experiment_config(arguments.config)
    if not isinstance(experiment, CapabilityExperimentConfig):
        raise ValueError("low-disk ATE runner requires a version-2 config")
    results = run_low_disk_ate(
        experiment,
        work_root=arguments.work_root,
        max_retained_failures=arguments.max_retained_failures,
    )
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    resumed = sum(result.resumed for result in results)
    print(
        f"entries={len(results)} passed={passed} failed={failed} resumed={resumed}",
        flush=True,
    )
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
