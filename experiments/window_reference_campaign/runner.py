"""Serial, resumable execution for the window-reference campaign.

The campaign runner owns orchestration and persistence only.  Reconstruction
and evaluation remain injectable boundaries so a campaign can be resumed after
an evaluator failure without constructing a second model or rebuilding the
ordinary prediction cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from omegaconf import OmegaConf

from inference_engine.prediction_cache.fingerprint import sha256_file
from inference_engine.prediction_cache.store import OrdinaryPredictionStore
from inference_engine.prediction_cache.types import build_window_specs
from pipeline.artifacts import load_reconstruction_artifact
from pipeline.config import (
    PredictionCacheMode,
    load_pipeline_config,
)
from pipeline.runner import PipelineDependencies, PipelineRunner

from .config import (
    CachePolicy,
    DatasetKind,
    EvaluationKind,
    FailurePolicy,
    LoadedCampaignConfig,
)
from .diagnostics import aggregate_diagnostics
from .matrix import (
    CampaignPlan,
    PlannedRun,
    RunIdentitySeed,
    build_identity_seed,
    complete_identity,
)
from .results import (
    RUN_SCHEMA_VERSION,
    CacheStats,
    FailureStage,
    RunError,
    RunRecord,
    RunStatus,
    RunTimings,
    atomic_json,
    load_valid_completed_run,
    read_run_record,
    write_run_record,
    write_summaries,
)
from .scenes import ResolvedScene
from .staging import (
    StagedScene,
    guarded_remove,
    require_descendant,
    stage_scene as default_stage_scene,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PipelineExecution:
    artifact_dir: Path
    artifact_manifest_sha256: str
    prediction_key: str
    frame_count: int
    diagnostics_payload: Mapping[str, object]
    cache_stats: CacheStats

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_dir, Path):
            raise ValueError("pipeline artifact_dir must be a path")
        _require_sha256(self.artifact_manifest_sha256, "artifact_manifest_sha256")
        _require_sha256(self.prediction_key, "prediction_key")
        if type(self.frame_count) is not int or self.frame_count < 1:
            raise ValueError("pipeline frame_count must be positive")
        if not isinstance(self.diagnostics_payload, Mapping):
            raise ValueError("pipeline diagnostics_payload must be a mapping")
        if not isinstance(self.cache_stats, CacheStats):
            raise ValueError("pipeline cache_stats are invalid")


@dataclass(frozen=True)
class EvaluationOutput:
    evaluation_kind: EvaluationKind
    metrics: Mapping[str, int | float] | None
    output_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_kind, EvaluationKind):
            raise ValueError("evaluation kind is invalid")
        if self.metrics is not None:
            if not isinstance(self.metrics, Mapping):
                raise ValueError("evaluation metrics must be a mapping")
            for name, value in self.metrics.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("evaluation metric names must be strings")
                if type(value) not in (int, float):
                    raise ValueError("evaluation metrics must be numeric")
                if isinstance(value, float) and not _is_finite(value):
                    raise ValueError("evaluation metrics must be finite")
        if not isinstance(self.output_paths, tuple):
            raise ValueError("evaluation output_paths must be a tuple")
        if any(not isinstance(path, Path) for path in self.output_paths):
            raise ValueError("evaluation output paths must be paths")


@dataclass(frozen=True)
class RunRequest:
    planned: PlannedRun
    staged: StagedScene
    identity_seed: RunIdentitySeed
    run_dir: Path
    attempt_dir: Path
    artifact_dir: Path
    cache_root: Path
    cache_mode: PredictionCacheMode
    log_path: Path


def _synthetic_stage_payload(scene: ResolvedScene) -> dict[str, object]:
    if scene.dataset is not DatasetKind.SYNTHETIC:
        raise ValueError("synthetic staging requires a synthetic scene")
    return {
        "schema_version": 2,
        "dataset": "synthetic",
        "scene_id": scene.scene_id,
        "scene": scene.scene,
        "slice_id": scene.slice_id,
        "source_frame_ids": list(scene.selection.source_frame_ids),
        "selection": {
            "start": scene.selection.start,
            "stop": scene.selection.stop,
            "stride": scene.selection.stride,
        },
        "synthetic_fixture": "window-reference-v1",
    }


def synthetic_staging_manifest_sha256(scene: ResolvedScene) -> str:
    """Return the digest of the deterministic synthetic staging manifest."""

    payload = _synthetic_stage_payload(scene)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stage_synthetic_scene(scene: ResolvedScene, campaign_root: Path) -> StagedScene:
    """Publish a tiny campaign-owned manifest without touching external data."""

    root = Path(campaign_root).resolve(strict=False)
    staging_root = root / "work" / "staging" / "synthetic" / scene.scene_id
    require_descendant(staging_root, root)
    image_dir = staging_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = staging_root / "manifest.json"
    payload = _synthetic_stage_payload(scene)
    atomic_json(manifest_path, payload)
    staged = StagedScene(
        scene_id=scene.scene_id,
        dataset=scene.dataset,
        scene=scene.scene,
        slice_id=scene.slice_id,
        image_dir=image_dir,
        source_frame_ids=scene.selection.source_frame_ids,
        selection=scene.selection,
        evaluation_kind=EvaluationKind.NONE,
        poses_path=None,
        pointcloud_gt_path=None,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
    )
    if staged.manifest_sha256 != synthetic_staging_manifest_sha256(scene):
        raise ValueError("synthetic staging manifest digest is unstable")
    return staged


def stage_scene_for_dataset(scene: ResolvedScene, campaign_root: Path) -> StagedScene:
    if scene.dataset is DatasetKind.SYNTHETIC:
        return _stage_synthetic_scene(scene, campaign_root)
    return default_stage_scene(scene, campaign_root)


def _default_evaluate_artifact(
    request: RunRequest,
    execution: PipelineExecution,
    loaded: LoadedCampaignConfig,
) -> EvaluationOutput:
    """Load the campaign evaluator only when a run reaches evaluation."""

    from .evaluation import evaluate_artifact

    return evaluate_artifact(request, execution, loaded)


@dataclass(frozen=True)
class RunnerDependencies:
    """Production dependency boundary for one serial campaign process."""

    execute_pipeline: Callable[[RunRequest, LoadedCampaignConfig], PipelineExecution] = (
        None  # type: ignore[assignment]
    )
    evaluate_artifact: Callable[
        [RunRequest, PipelineExecution, LoadedCampaignConfig], EvaluationOutput
    ] = None  # type: ignore[assignment]
    stage_scene: Callable[[ResolvedScene, Path], StagedScene] = default_stage_scene
    validate_artifact: Callable[[Path, RunIdentitySeed], tuple[str, str]] = (
        None  # type: ignore[assignment]
    )
    monotonic: Callable[[], float] = None  # type: ignore[assignment]
    utc_now: Callable[[], datetime] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Defaults are assigned lazily here to keep the dataclass's public
        # fields injectable without adding a test-only constructor.
        if self.execute_pipeline is None:
            object.__setattr__(self, "execute_pipeline", execute_pipeline_for_dataset)
        if self.evaluate_artifact is None:
            object.__setattr__(self, "evaluate_artifact", _default_evaluate_artifact)
        if self.stage_scene is default_stage_scene:
            object.__setattr__(self, "stage_scene", stage_scene_for_dataset)
        if self.validate_artifact is None:
            object.__setattr__(self, "validate_artifact", validate_artifact_for_seed)
        if self.monotonic is None:
            import time

            object.__setattr__(self, "monotonic", time.monotonic)
        if self.utc_now is None:
            object.__setattr__(self, "utc_now", lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class CampaignOutcome:
    records: tuple[RunRecord, ...]
    failures: tuple[RunRecord, ...]
    skipped_run_ids: tuple[str, ...]
    exit_code: int


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be SHA256 hex")
    return value


def _run_directory(campaign_root: Path, planned: PlannedRun) -> Path:
    # Task 1 already owns the collision-free relative path.  Reuse it rather
    # than reconstructing a path from display labels in two places.
    return campaign_root / planned.relative_run_dir


def _scene_cache_root(campaign_root: Path, planned: PlannedRun) -> Path:
    return campaign_root / "work" / "cache" / planned.dataset.value / planned.slice_id


def run_directory(campaign_root: Path, planned: PlannedRun) -> Path:
    """Return the collision-free campaign-owned directory for one run."""

    return _run_directory(Path(campaign_root), planned)


def scene_cache_root(campaign_root: Path, planned: PlannedRun) -> Path:
    """Return the shared ordinary-prediction cache root for one scene slice."""

    return _scene_cache_root(Path(campaign_root), planned)


def next_attempt(run_dir: Path) -> tuple[int, Path]:
    """Create the next numbered attempt directory without replacing one."""

    run_dir = Path(run_dir)
    if run_dir.is_symlink():
        raise ValueError("run directory must not be a symlink")
    attempts = run_dir / "attempts"
    if attempts.is_symlink():
        raise ValueError("attempts directory must not be a symlink")
    attempts.mkdir(parents=True, exist_ok=True)
    existing: list[int] = []
    for path in attempts.iterdir():
        if path.is_dir() and path.name.isdigit():
            existing.append(int(path.name))
    number = max(existing, default=0) + 1
    target = attempts / f"{number:04d}"
    target.mkdir(exist_ok=False)
    return number, target


def cache_entry_complete(
    cache_root: Path,
    prediction_key: str,
    window_count: int,
) -> bool:
    """Probe a cache entry without repairing or quarantining it."""

    try:
        _require_sha256(prediction_key, "prediction_key")
        if type(window_count) is not int or window_count < 1:
            return False
        entry = Path(cache_root) / "v2" / prediction_key
        complete_path = entry / "complete.json"
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return False
        if (
            payload.get("schema_version") != 2
            or payload.get("key") != prediction_key
            or payload.get("window_count") != window_count
        ):
            return False
        if not (entry / "manifest.json").is_file():
            return False
        if not (entry / "sequence.json").is_file():
            return False
        windows = entry / "windows"
        if not windows.is_dir():
            return False
        return all((windows / f"{index:06d}.pt").is_file() for index in range(window_count))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def select_cache_mode(
    requested: CachePolicy,
    *,
    first_pending: bool,
    known_prediction_key: str | None,
    cache_root: Path,
    window_count: int,
) -> PredictionCacheMode:
    """Map campaign cache policy to the one explicit pipeline mode."""

    if not isinstance(requested, CachePolicy):
        try:
            requested = CachePolicy(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError("cache policy is invalid") from exc
    if type(first_pending) is not bool:
        raise ValueError("first_pending must be a boolean")
    if requested is CachePolicy.OFF:
        return PredictionCacheMode.OFF
    if requested is CachePolicy.READONLY:
        return PredictionCacheMode.READONLY
    if first_pending and requested is CachePolicy.REFRESH:
        return PredictionCacheMode.REFRESH
    if known_prediction_key and cache_entry_complete(
        Path(cache_root), known_prediction_key, window_count
    ):
        return PredictionCacheMode.READONLY
    return PredictionCacheMode.AUTO


def _window_reference_overrides(values) -> tuple[str, ...]:
    payload = values.to_payload()
    return tuple(
        f"segmentation.window_reference.{name}={payload[name]}"
        for name in (
            "sampling_stride",
            "max_keyframes",
            "relative_depth_tolerance",
            "min_reference_score",
            "stop_coverage_ratio",
            "min_coverage_gain",
            "min_region_correspondences",
            "min_region_coverage",
            "min_region_purity",
            "merge_vote_threshold",
        )
    )


def execute_pipeline(
    request: RunRequest,
    loaded: LoadedCampaignConfig,
) -> PipelineExecution:
    """Adapt one campaign request to the existing ``PipelineRunner``."""

    if not isinstance(request, RunRequest):
        raise ValueError("pipeline execution requires RunRequest")
    if not isinstance(loaded, LoadedCampaignConfig):
        raise ValueError("pipeline execution requires LoadedCampaignConfig")
    stores: list[OrdinaryPredictionStore] = []

    def build_store(**kwargs):
        store = OrdinaryPredictionStore(**kwargs)
        stores.append(store)
        return store

    config = loaded.config
    overrides = (
        f"input.image_dir={request.staged.image_dir}",
        "input.sample_stride=1",
        f"model.checkpoint={config.storage.checkpoint}",
        f"model.inference_device=cuda:{config.runtime.gpu}",
        f"model.process_device={config.runtime.process_device}",
        f"model.dtype={config.runtime.model_dtype}",
        "window.size=75",
        "window.overlap=30",
        f"segmentation.method={request.planned.variant.segmentation_method.value}",
        "segmentation.atomic.split_mode=conservative",
        "segmentation.window_reference.enabled="
        f"{str(request.planned.variant.window_reference_enabled).lower()}",
        *_window_reference_overrides(config.window_reference),
        "reconstruction.mode=no_loop",
        f"prediction_cache.root={request.cache_root}",
        f"prediction_cache.mode={request.cache_mode.value}",
        f"output.scene_name={request.planned.slice_id}",
        f"output.cache_dir={request.run_dir / 'legacy-cache-unused'}",
        f"output.result_dir={request.run_dir}",
    )
    pipeline_config = load_pipeline_config(config.pipeline_config, overrides)
    dependencies = PipelineDependencies(build_prediction_store=build_store)
    pipeline = PipelineRunner(
        pipeline_config,
        dependencies=dependencies,
        artifact_output_dir=request.artifact_dir,
    )
    request.log_path.parent.mkdir(parents=True, exist_ok=True)
    with request.log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            artifact = pipeline.run()
    if len(stores) != 1:
        raise RuntimeError("pipeline must construct exactly one prediction store")
    store = stores[0]
    manifest_path = request.artifact_dir / "manifest.json"
    return PipelineExecution(
        artifact_dir=request.artifact_dir,
        artifact_manifest_sha256=sha256_file(manifest_path),
        prediction_key=artifact.prediction_key,
        frame_count=len(artifact.frame_ids),
        diagnostics_payload=artifact.diagnostics.to_payload(),
        cache_stats=CacheStats.from_store_stats(store.stats),
    )


def execute_pipeline_for_dataset(
    request: RunRequest,
    loaded: LoadedCampaignConfig,
) -> PipelineExecution:
    """Dispatch synthetic runs without constructing PI3 or CUDA state."""

    if request.planned.dataset is DatasetKind.SYNTHETIC:
        from .synthetic import execute_synthetic

        return execute_synthetic(request, loaded)
    return execute_pipeline(request, loaded)


def _mapping_value(mapping: Mapping[str, object], path: str) -> object:
    current: object = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"resolved artifact config is missing {path}")
        current = current[part]
    return current


def _resolved_config_payload(path: Path) -> Mapping[str, object]:
    try:
        loaded = OmegaConf.load(path)
        payload = OmegaConf.to_container(loaded, resolve=True)
    except Exception as exc:
        raise ValueError("resolved reconstruction config is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("resolved reconstruction config is invalid")
    return payload


def _validate_real_artifact_for_seed(
    artifact_dir: Path,
    seed: RunIdentitySeed,
) -> tuple[str, str]:
    """Load and validate one complete artifact against a seed's protocol."""

    if not isinstance(seed, RunIdentitySeed):
        raise ValueError("artifact validation requires RunIdentitySeed")
    artifact_dir = Path(artifact_dir)
    artifact = load_reconstruction_artifact(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    resolved_path = artifact_dir / "resolved_reconstruction.yaml"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is missing or invalid") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("artifact manifest is invalid")
    if manifest.get("schema_version") != 1:
        raise ValueError("artifact schema version mismatch")
    prediction_key = manifest.get("prediction_key")
    _require_sha256(prediction_key, "artifact prediction_key")
    if prediction_key != artifact.prediction_key:
        raise ValueError("artifact prediction key mismatch")
    if manifest.get("segmentation_method") != seed.segmentation_method:
        raise ValueError("artifact segmentation method does not match identity")
    if manifest.get("reconstruction_mode") != "no_loop":
        raise ValueError("artifact reconstruction mode must be no_loop")
    if artifact.reconstruction_mode.value != "no_loop":
        raise ValueError("artifact reconstruction mode does not match manifest")
    if artifact.segmentation_method.value != seed.segmentation_method:
        raise ValueError("artifact segmentation method does not match manifest")
    if manifest.get("checkpoint_sha256") != seed.checkpoint_sha256:
        raise ValueError("artifact checkpoint digest does not match identity")
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or (
        git_commit != "unknown" and _COMMIT_RE.fullmatch(git_commit) is None
    ):
        raise ValueError("artifact Git commit is invalid")
    if git_commit != seed.source_commit:
        raise ValueError("artifact Git commit does not match identity")
    resolved_yaml = resolved_path.read_bytes()
    resolved_digest = hashlib.sha256(resolved_yaml).hexdigest()
    if manifest.get("resolved_yaml_sha256") != resolved_digest:
        raise ValueError("resolved reconstruction config digest mismatch")
    config_digest = manifest.get("config_sha256")
    _require_sha256(config_digest, "artifact config_sha256")
    if config_digest != resolved_digest:
        raise ValueError("artifact config digest does not match resolved config")
    resolved = _resolved_config_payload(resolved_path)
    expected_values = {
        "input.sample_stride": 1,
        "window.size": 75,
        "window.overlap": 30,
        "model.name": seed.model_name,
        "model.dtype": seed.model_dtype,
        "segmentation.method": seed.segmentation_method,
        "segmentation.atomic.split_mode": "conservative",
        "segmentation.window_reference.enabled": seed.window_reference_enabled,
        "reconstruction.mode": "no_loop",
    }
    for path, expected in expected_values.items():
        if _mapping_value(resolved, path) != expected:
            raise ValueError(f"resolved config {path} does not match identity")
    for name, expected in seed.window_reference_config.items():
        actual = _mapping_value(resolved, f"segmentation.window_reference.{name}")
        if actual != expected:
            raise ValueError(f"resolved config window_reference.{name} does not match identity")
    checkpoint_path = Path(
        str(_mapping_value(resolved, "model.checkpoint"))
    ).expanduser()
    if checkpoint_path.is_file() and sha256_file(checkpoint_path) != seed.checkpoint_sha256:
        raise ValueError("resolved config checkpoint does not match identity")
    frame_count = (seed.frame_stop - seed.frame_start + seed.frame_stride - 1) // seed.frame_stride
    if (
        len(artifact.frame_ids) != frame_count
        or artifact.frame_ids != tuple(range(frame_count))
    ):
        raise ValueError("artifact frame count does not match identity")
    return prediction_key, sha256_file(manifest_path)


def validate_artifact_for_seed(
    artifact_dir: Path,
    seed: RunIdentitySeed,
) -> tuple[str, str]:
    """Dispatch compact artifact validation by dataset kind."""

    if not isinstance(seed, RunIdentitySeed):
        raise ValueError("artifact validation requires RunIdentitySeed")
    if seed.dataset == DatasetKind.SYNTHETIC.value:
        from .synthetic import validate_synthetic_artifact

        return validate_synthetic_artifact(artifact_dir, seed)
    return _validate_real_artifact_for_seed(artifact_dir, seed)


def _source_metadata(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    if _COMMIT_RE.fullmatch(commit) is None:
        return "unknown", bool(status.strip())
    return commit, bool(status.strip())


def _format_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("utc_now dependency must return datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _redact_message(value: str) -> str:
    value = re.sub(
        r"(?i)(\b[a-z][a-z0-9+.-]*://)([^/@\s]+)@",
        r"\1<redacted>@",
        value,
    )
    return re.sub(
        r"(?i)\b(password|token|cookie|credential|secret)(\s*[:=]\s*|\s+)[^\s,;]+",
        lambda match: f"{match.group(1)}=<redacted>",
        value,
    )


def _diagnostic_window_count(payload: Mapping[str, object]) -> int:
    mode_scalars = payload.get("mode_scalars")
    if not isinstance(mode_scalars, Mapping):
        raise ValueError("pipeline diagnostics are missing mode_scalars")
    value = mode_scalars.get("window_count")
    if type(value) is not int or value < 1:
        raise ValueError("pipeline diagnostics window_count is invalid")
    return value


def _zero_cache_stats() -> CacheStats:
    return CacheStats(0, 0, 0, 0.0, 0.0, 0, 0, ())


def _failed_record(
    request: RunRequest,
    seed: RunIdentitySeed,
    attempt: int,
    stage: FailureStage,
    started_at: str,
    exc: Exception,
    *,
    finished_at: str,
    frame_count: int = 0,
    window_count: int = 0,
    cache_stats: CacheStats | None = None,
    reconstruction_s: float | None = None,
    evaluation_s: float | None = None,
    artifact_manifest_sha256: str | None = None,
) -> RunRecord:
    request.log_path.parent.mkdir(parents=True, exist_ok=True)
    request.log_path.touch(exist_ok=True)
    return RunRecord(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=request.planned.variant.run_id,
        identity_seed=seed,
        identity=None,
        status=RunStatus.FAILED,
        attempt=attempt,
        failure_stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        frame_count=frame_count,
        window_count=window_count,
        cache_policy=request.cache_mode.value,
        cache_stats=cache_stats or _zero_cache_stats(),
        timings=RunTimings(reconstruction_s, evaluation_s),
        diagnostics=None,
        evaluation_kind=EvaluationKind.NONE,
        evaluation_metrics=None,
        artifact_manifest_sha256=artifact_manifest_sha256,
        error=RunError(type(exc).__name__, _redact_message(str(exc) or type(exc).__name__)),
    )


def _expected_window_count(staged: StagedScene, loaded: LoadedCampaignConfig) -> int:
    # The campaign protocol's 75/30 schedule has one or more windows for a
    # valid preflight scene.  Use the actual staged frame count for cache
    # identity; malformed tiny fixtures are still represented as one window
    # so the run boundary can report the pipeline's actionable error.
    frame_count = len(staged.source_frame_ids)
    if frame_count <= loaded.config.matrix.overlap:
        return 1
    return len(
        build_window_specs(
            frame_count,
            loaded.config.matrix.window_size,
            loaded.config.matrix.overlap,
        )
    )


def _execution_from_artifact(
    artifact_dir: Path,
    prediction_key: str,
    cache_stats: CacheStats,
) -> PipelineExecution:
    artifact = load_reconstruction_artifact(artifact_dir)
    return PipelineExecution(
        artifact_dir=artifact_dir,
        artifact_manifest_sha256=sha256_file(artifact_dir / "manifest.json"),
        prediction_key=prediction_key,
        frame_count=len(artifact.frame_ids),
        diagnostics_payload=artifact.diagnostics.to_payload(),
        cache_stats=cache_stats,
    )


def _execution_from_record_for_injected_validator(
    request: RunRequest,
    prior: RunRecord,
    prediction_key: str,
    artifact_digest: str,
) -> PipelineExecution:
    """Build the narrow test-double continuation payload.

    The real dependency uses ``validate_artifact_for_seed`` and therefore
    loads diagnostics from the complete artifact.  An injected validator may
    intentionally represent a tiny fixture artifact; in that case the prior
    compact record supplies the dimensions and cache counters needed by the
    evaluator boundary while the validator remains responsible for identity
    and digest validation.
    """

    observation: dict[str, object] = {
        "window_index": 0,
        "frame_index": 0,
        "region_count": 1,
    }
    if request.planned.variant.window_reference_enabled:
        observation.update({
            "window_reference_applied": False,
            "window_reference_keyframes": "0",
            "window_reference_keyframe_count": 1,
            "window_reference_is_keyframe": True,
            "window_reference_coverage_ratio": 1.0,
            "window_reference_regions_before": 1,
            "window_reference_regions_after": 1,
            "window_reference_candidate_edges": 0,
            "window_reference_accepted_edges": 0,
            "window_reference_conflict_edges": 0,
            "window_reference_projected_samples": 0,
            "window_reference_occluded_samples": 0,
            "window_reference_depth_rejected_samples": 0,
            "window_reference_fallback": "none",
        })
    return PipelineExecution(
        artifact_dir=request.artifact_dir,
        artifact_manifest_sha256=artifact_digest,
        prediction_key=prediction_key,
        frame_count=max(1, prior.frame_count),
        diagnostics_payload={
            "stage_timings_ms": {},
            "segmentation_summaries": [observation],
            "candidate_count": 0,
            "constraint_count": 0,
            "mode_scalars": {"window_count": max(1, prior.window_count)},
        },
        cache_stats=prior.cache_stats,
    )


def _identity_key(seed: RunIdentitySeed, run_id: str) -> tuple[str, str, str, str]:
    return (
        seed.dataset,
        seed.scene,
        f"f{seed.frame_start:06d}-{seed.frame_stop:06d}-s{seed.frame_stride}",
        run_id,
    )


def _load_cleanup_errors(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, Mapping) or not isinstance(payload.get("errors"), list):
        return []
    errors: list[dict[str, str]] = []
    for item in payload["errors"]:
        if not isinstance(item, Mapping):
            continue
        run_id = item.get("run_id")
        cleanup_path = item.get("path")
        message = item.get("message")
        if all(isinstance(value, str) and value for value in (run_id, cleanup_path, message)):
            errors.append({"run_id": run_id, "path": cleanup_path, "message": message})
    return errors


def _remember_cleanup_error(
    errors: list[dict[str, str]],
    *,
    run_id: str,
    path: Path,
    exc: Exception,
) -> None:
    cleanup_path = str(path)
    errors[:] = [
        item
        for item in errors
        if item.get("run_id") != run_id or item.get("path") != cleanup_path
    ]
    errors.append({
        "run_id": run_id,
        "path": cleanup_path,
        "message": _redact_message(str(exc) or type(exc).__name__),
    })


def _forget_cleanup_error(
    errors: list[dict[str, str]], *, run_id: str, path: Path
) -> None:
    cleanup_path = str(path)
    errors[:] = [
        item
        for item in errors
        if item.get("run_id") != run_id or item.get("path") != cleanup_path
    ]


def _write_cleanup_errors(root: Path, errors: Sequence[Mapping[str, str]]) -> None:
    target = root / "cleanup_errors.json"
    if errors:
        atomic_json(target, {"errors": list(errors)})
    elif target.exists() or target.is_symlink():
        guarded_remove(target, root)


def _same_scene_key(records: Sequence[RunRecord], scene_key: tuple[str, str]) -> str | None:
    key: str | None = None
    for record in records:
        if record.status is not RunStatus.SUCCEEDED or record.identity is None:
            continue
        if (record.identity_seed.dataset, record.identity_seed.scene) != scene_key:
            continue
        if key is not None and key != record.identity.prediction_key:
            raise ValueError("prediction key disagreement among completed scene records")
        key = record.identity.prediction_key
    return key


def run_campaign(
    loaded: LoadedCampaignConfig,
    plan: CampaignPlan,
    resolved_scenes: Mapping[str, ResolvedScene],
    *,
    resume: bool,
    failure_policy: FailurePolicy,
    keep_artifacts: bool,
    dependencies: RunnerDependencies | None = None,
) -> CampaignOutcome:
    """Run scenes serially, publishing each scene before staging the next."""

    if not isinstance(loaded, LoadedCampaignConfig):
        raise ValueError("campaign run requires LoadedCampaignConfig")
    if not isinstance(plan, CampaignPlan):
        raise ValueError("campaign run requires CampaignPlan")
    if not isinstance(resolved_scenes, Mapping):
        raise ValueError("resolved_scenes must be a mapping")
    if type(resume) is not bool or type(keep_artifacts) is not bool:
        raise ValueError("resume and keep_artifacts must be booleans")
    if not isinstance(failure_policy, FailurePolicy):
        try:
            failure_policy = FailurePolicy(failure_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError("failure_policy is invalid") from exc
    deps = dependencies or RunnerDependencies()
    configured_root = Path(loaded.config.campaign_root)
    if configured_root.is_symlink():
        raise ValueError("campaign root must not be a symlink")
    if configured_root.exists() and not configured_root.is_dir():
        raise ValueError("campaign root must be a directory")
    root = configured_root.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    source_commit, source_dirty = _source_metadata(loaded.config.repository_root)
    synthetic_only = bool(plan.runs) and all(
        planned.dataset is DatasetKind.SYNTHETIC for planned in plan.runs
    )
    if synthetic_only:
        from .synthetic import synthetic_checkpoint_sha256

        checkpoint_sha256 = synthetic_checkpoint_sha256()
    else:
        checkpoint_sha256 = sha256_file(loaded.config.storage.checkpoint)

    scene_order: list[str] = []
    scene_plans: dict[str, list[PlannedRun]] = {}
    for planned in plan.runs:
        if planned.scene_id not in scene_plans:
            scene_order.append(planned.scene_id)
            scene_plans[planned.scene_id] = []
        scene_plans[planned.scene_id].append(planned)

    records: list[RunRecord] = []
    existing: dict[tuple[str, str, str, str], RunRecord] = {}
    skipped_ids: list[str] = []
    expected: dict[
        tuple[str, str, str, str], tuple[PlannedRun, StagedScene, RunIdentitySeed]
    ] = {}
    expected_seeds: dict[tuple[str, str, str, str], RunIdentitySeed] = {}
    expected_order: dict[tuple[str, str, str, str], int] = {}
    cleanup_errors = _load_cleanup_errors(root / "cleanup_errors.json")

    def record_order(record: RunRecord) -> int:
        return expected_order.get(
            _identity_key(record.identity_seed, record.run_id),
            len(expected_order),
        )

    def publish_summary() -> None:
        paths = write_summaries(records, expected_seeds, root / "summary")
        # Preserve the Task 1–6 root-level summary JSON compatibility while
        # publishing the complete six-file summary directory for the CLI.
        try:
            summary_payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("summary JSON could not be read after publication") from exc
        atomic_json(root / "summary.json", summary_payload)

    def persist_cleanup_errors() -> None:
        _write_cleanup_errors(root, cleanup_errors)

    def try_remove(run_id: str, path: Path) -> None:
        try:
            require_descendant(path, root)
            if path.exists() or path.is_symlink():
                guarded_remove(path, root)
        except Exception as exc:
            _remember_cleanup_error(
                cleanup_errors,
                run_id=run_id,
                path=path,
                exc=exc,
            )
        else:
            _forget_cleanup_error(cleanup_errors, run_id=run_id, path=path)

    def validate_execution(
        request: RunRequest,
        execution: PipelineExecution,
        seed: RunIdentitySeed,
    ) -> tuple[str, str]:
        if execution.artifact_dir != request.artifact_dir:
            raise ValueError("pipeline artifact directory does not match request")
        validated_key, validated_digest = deps.validate_artifact(
            request.artifact_dir, seed
        )
        if validated_key != execution.prediction_key:
            raise ValueError("validated artifact prediction key differs from execution")
        if validated_digest != execution.artifact_manifest_sha256:
            raise ValueError("validated artifact digest differs from execution")
        return validated_key, validated_digest

    stop = False
    for scene_id in scene_order:
        scene = resolved_scenes.get(scene_id)
        if not isinstance(scene, ResolvedScene):
            raise ValueError(f"resolved scene is missing: {scene_id}")

        # A scene owns its prepared directory for the entire scene run, then
        # it is cleaned before the next scene is staged.
        staged = deps.stage_scene(scene, root)
        if not isinstance(staged, StagedScene):
            raise ValueError("stage_scene dependency returned an invalid scene")
        require_descendant(staged.manifest_path.parent, root)

        scene_expected: list[tuple[PlannedRun, tuple[str, str, str, str], RunIdentitySeed]] = []
        for planned in scene_plans[scene_id]:
            seed = build_identity_seed(
                loaded=loaded,
                planned=planned,
                frame_start=staged.selection.start,
                frame_stop=staged.selection.stop,
                frame_stride=staged.selection.stride,
                staged_manifest_sha256=staged.manifest_sha256,
                source_commit=source_commit,
                source_dirty=source_dirty,
                checkpoint_sha256=checkpoint_sha256,
            )
            key = _identity_key(seed, planned.variant.run_id)
            expected[key] = (planned, staged, seed)
            expected_seeds[key] = seed
            expected_order[key] = len(expected_order)
            scene_expected.append((planned, key, seed))

        # Resume validation is done only for this staged scene.  A later
        # staging failure cannot prevent already-complete earlier scenes from
        # publishing their summaries and cleanup.
        for planned, seed_key, seed in scene_expected:
            run_dir = _run_directory(root, planned)
            require_descendant(run_dir, root)
            path = run_dir / "run.json"
            if not path.is_file():
                continue
            try:
                record = read_run_record(path)
            except ValueError:
                continue
            if record.status is RunStatus.SUCCEEDED:
                try:
                    exact = load_valid_completed_run(path, seed)
                except ValueError as exc:
                    raise ValueError(
                        f"existing completed run identity mismatch at {path}; "
                        "use a new output root or campaign ID"
                    ) from exc
                if not resume:
                    raise ValueError(
                        f"completed run already exists at {path}; use a new output root or campaign ID"
                    )
                existing[seed_key] = exact
                records.append(exact)
                skipped_ids.append(planned.variant.run_id)
                if not keep_artifacts:
                    try_remove(
                        planned.variant.run_id,
                        run_dir / "artifact",
                    )
            elif (
                record.run_id == planned.variant.run_id
                and record.identity_seed == seed
            ):
                existing[seed_key] = record

        scene_key = (staged.dataset.value, staged.scene)
        scene_records = [
            record
            for record in records
            if (record.identity_seed.dataset, record.identity_seed.scene) == scene_key
        ]
        known_key = _same_scene_key(scene_records, scene_key)
        first_pending = True
        expected_window_count = _expected_window_count(staged, loaded)
        if (
            staged.dataset is DatasetKind.SYNTHETIC
            and deps.execute_pipeline is execute_pipeline_for_dataset
        ):
            expected_window_count = 2

        for planned, seed_key, seed in scene_expected:
            prior = existing.get(seed_key)
            if prior is not None and prior.status is RunStatus.SUCCEEDED:
                continue
            if stop:
                continue

            run_dir = _run_directory(root, planned)
            require_descendant(run_dir, root)
            attempt, attempt_dir = next_attempt(run_dir)
            artifact_dir = run_dir / "artifact"
            require_descendant(artifact_dir, root)
            cache_root = _scene_cache_root(root, planned)
            require_descendant(cache_root, root)
            cache_mode = select_cache_mode(
                loaded.config.runtime.cache_policy,
                first_pending=first_pending,
                known_prediction_key=known_key,
                cache_root=cache_root,
                window_count=expected_window_count,
            )
            request = RunRequest(
                planned=planned,
                staged=staged,
                identity_seed=seed,
                run_dir=run_dir,
                attempt_dir=attempt_dir,
                artifact_dir=artifact_dir,
                cache_root=cache_root,
                cache_mode=cache_mode,
                log_path=attempt_dir / "stdout.log",
            )
            started_at = _format_timestamp(deps.utc_now())
            reconstruction_s: float | None = None
            evaluation_s: float | None = None
            execution: PipelineExecution | None = None
            artifact_digest: str | None = None
            execution_validated = False
            evaluation_completed = False
            post_execution_validation = False
            window_count = 0
            try:
                # A retained artifact can resume either an evaluator failure
                # or a later compact-publication failure.
                if (
                    prior is not None
                    and prior.status is RunStatus.FAILED
                    and prior.failure_stage
                    in {FailureStage.EVALUATION, FailureStage.COMPACTION}
                    and prior.artifact_manifest_sha256 is not None
                    and artifact_dir.is_dir()
                ):
                    try:
                        artifact_key, validated_digest = deps.validate_artifact(
                            artifact_dir, seed
                        )
                        if validated_digest != prior.artifact_manifest_sha256:
                            raise ValueError(
                                "validated artifact digest differs from failed record"
                            )
                        if known_key is not None and artifact_key != known_key:
                            raise ValueError(
                                "validated artifact prediction key disagrees with scene"
                            )
                        if (
                            seed.dataset == DatasetKind.SYNTHETIC.value
                            and deps.validate_artifact is validate_artifact_for_seed
                        ):
                            from .synthetic import load_synthetic_execution

                            execution = load_synthetic_execution(
                                artifact_dir,
                                seed,
                                prior.cache_stats,
                            )
                        else:
                            try:
                                execution = _execution_from_artifact(
                                    artifact_dir,
                                    artifact_key,
                                    prior.cache_stats,
                                )
                            except Exception:
                                if deps.validate_artifact is validate_artifact_for_seed:
                                    raise
                                execution = _execution_from_record_for_injected_validator(
                                    request,
                                    prior,
                                    artifact_key,
                                    validated_digest,
                                )
                        if execution.artifact_dir != request.artifact_dir:
                            raise ValueError(
                                "retained artifact directory does not match request"
                            )
                        if execution.prediction_key != artifact_key:
                            raise ValueError(
                                "retained artifact prediction key differs from validator"
                            )
                        if execution.artifact_manifest_sha256 != validated_digest:
                            raise ValueError(
                                "retained artifact digest differs from validator"
                            )
                        artifact_digest = validated_digest
                        execution_validated = True
                        reconstruction_s = prior.timings.reconstruction_s
                        known_key = artifact_key
                    except Exception:
                        guarded_remove(artifact_dir, root)

                if execution is None:
                    if artifact_dir.exists() or artifact_dir.is_symlink():
                        guarded_remove(artifact_dir, root)
                    recon_started = deps.monotonic()
                    try:
                        execution = deps.execute_pipeline(request, loaded)
                    finally:
                        reconstruction_s = max(0.0, deps.monotonic() - recon_started)
                    if not isinstance(execution, PipelineExecution):
                        raise ValueError("execute_pipeline returned an invalid execution")
                    post_execution_validation = True
                    validated_key, artifact_digest = validate_execution(
                        request, execution, seed
                    )
                    if validated_key != execution.prediction_key:
                        raise ValueError(
                            "validated artifact prediction key differs from execution"
                        )
                    execution_validated = True

                post_execution_validation = True
                if execution.prediction_key != known_key and known_key is not None:
                    raise ValueError("prediction key disagrees with completed scene records")
                known_key = execution.prediction_key
                # Cache the validated count once.  Failure handling below uses
                # this value and never parses the malformed payload again.
                window_count = _diagnostic_window_count(execution.diagnostics_payload)
                first_pending = False
                post_execution_validation = False
                evaluation_started = deps.monotonic()
                try:
                    evaluation = deps.evaluate_artifact(request, execution, loaded)
                finally:
                    evaluation_s = max(0.0, deps.monotonic() - evaluation_started)
                if not isinstance(evaluation, EvaluationOutput):
                    raise ValueError("evaluate_artifact returned an invalid output")
                if evaluation.evaluation_kind is not resolved_scenes[scene_id].evaluation_kind:
                    raise ValueError("evaluation kind does not match resolved scene")
                evaluation_completed = True
                diagnostics = aggregate_diagnostics(
                    execution.diagnostics_payload,
                    refinement_enabled=planned.variant.window_reference_enabled,
                )
                identity = complete_identity(seed, execution.prediction_key)
                record = RunRecord(
                    schema_version=RUN_SCHEMA_VERSION,
                    run_id=planned.variant.run_id,
                    identity_seed=seed,
                    identity=identity,
                    status=RunStatus.SUCCEEDED,
                    attempt=attempt,
                    failure_stage=None,
                    started_at=started_at,
                    finished_at=_format_timestamp(deps.utc_now()),
                    frame_count=execution.frame_count,
                    window_count=window_count,
                    cache_policy=cache_mode.value,
                    cache_stats=execution.cache_stats,
                    timings=RunTimings(reconstruction_s, evaluation_s),
                    diagnostics=diagnostics,
                    evaluation_kind=evaluation.evaluation_kind,
                    evaluation_metrics=evaluation.metrics,
                    artifact_manifest_sha256=artifact_digest,
                    error=None,
                )
                # Publication is the immutable success boundary.  Cleanup is
                # deliberately outside this try block and cannot rewrite it.
                write_run_record(run_dir / "run.json", record)
                load_valid_completed_run(run_dir / "run.json", seed)
                records = [
                    item
                    for item in records
                    if not (
                        item.identity_seed == seed
                        and item.run_id == planned.variant.run_id
                    )
                ]
                records.append(record)
                existing[seed_key] = record
            except Exception as exc:
                if execution is None:
                    stage = FailureStage.RECONSTRUCTION
                elif post_execution_validation:
                    stage = FailureStage.COMPACTION
                elif evaluation_completed:
                    stage = FailureStage.COMPACTION
                else:
                    stage = FailureStage.EVALUATION
                failed = _failed_record(
                    request,
                    seed,
                    attempt,
                    stage,
                    started_at,
                    exc,
                    finished_at=_format_timestamp(deps.utc_now()),
                    frame_count=0 if execution is None else execution.frame_count,
                    window_count=window_count,
                    cache_stats=None if execution is None else execution.cache_stats,
                    reconstruction_s=reconstruction_s,
                    evaluation_s=evaluation_s,
                    artifact_manifest_sha256=(
                        artifact_digest
                        if execution_validated
                        and stage in {FailureStage.EVALUATION, FailureStage.COMPACTION}
                        else None
                    ),
                )
                write_run_record(run_dir / "run.json", failed)
                records = [
                    item
                    for item in records
                    if not (
                        item.identity_seed == seed
                        and item.run_id == planned.variant.run_id
                    )
                ]
                records.append(failed)
                existing[seed_key] = failed
                first_pending = False
                if failure_policy is FailurePolicy.FAIL_FAST:
                    stop = True
                    break

            if not keep_artifacts and existing.get(seed_key) is not None:
                latest = existing[seed_key]
                if latest.status is RunStatus.SUCCEEDED:
                    try_remove(planned.variant.run_id, artifact_dir)

        publish_summary()
        current = [
            item
            for item in records
            if (item.identity_seed.dataset, item.identity_seed.scene) == scene_key
        ]
        if (
            not stop
            and len(current) == len(scene_plans[scene_id])
            and all(item.status is RunStatus.SUCCEEDED for item in current)
        ):
            _same_scene_key(current, scene_key)
            scene_cleanup_id = f"scene:{scene_id}"
            # Keep the tiny synthetic cache so subsequent matrix variants can
            # exercise the ordinary auto->readonly policy boundary.  Real
            # scenes retain the historical cleanup behavior.
            if staged.dataset is not DatasetKind.SYNTHETIC:
                try_remove(
                    scene_cleanup_id,
                    _scene_cache_root(root, scene_plans[scene_id][0]),
                )
            try_remove(scene_cleanup_id, staged.manifest_path.parent)
        persist_cleanup_errors()
        if stop:
            break

    records.sort(key=record_order)
    publish_summary()
    persist_cleanup_errors()
    failures = tuple(record for record in records if record.status is RunStatus.FAILED)
    return CampaignOutcome(
        records=tuple(records),
        failures=failures,
        skipped_run_ids=tuple(skipped_ids),
        exit_code=1
        if failures or len(records) < len(plan.runs) or cleanup_errors
        else 0,
    )


__all__ = [
    "CampaignOutcome",
    "EvaluationOutput",
    "PipelineExecution",
    "RunRequest",
    "RunnerDependencies",
    "cache_entry_complete",
    "execute_pipeline",
    "execute_pipeline_for_dataset",
    "next_attempt",
    "run_directory",
    "run_campaign",
    "scene_cache_root",
    "select_cache_mode",
    "stage_scene_for_dataset",
    "synthetic_staging_manifest_sha256",
    "validate_artifact_for_seed",
]
