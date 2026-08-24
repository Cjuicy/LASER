from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from inference_engine.prediction_cache.fingerprint import sha256_file
from pipeline.config import PredictionCacheMode, load_pipeline_config
from pipeline.manifest import discover_image_manifest
from pipeline.runner import PipelineRunner

from .ate import evaluate_ate_artifact
from .config import (
    CapabilityExperimentConfig,
    EvaluationKind,
    ExperimentConfig,
)
from .evaluation_bundle import (
    EvaluationBundleRunner,
    EvaluatorRecord,
    EvaluatorStatus,
)
from .matrix import (
    ArtifactRepository,
    CapabilityMatrixEntry,
    MatrixEntry,
    reconstruction_identity,
    validate_matrix_entry,
)
from .pointcloud import evaluate_pointcloud_artifact


@dataclass(frozen=True)
class ExperimentRunRecord:
    entry: MatrixEntry
    reconstruction_identity: str
    artifact_dir: Path
    evaluation_output: Path
    prediction_cache_mode: PredictionCacheMode


@dataclass(frozen=True)
class CapabilityExperimentRunRecord:
    entry: CapabilityMatrixEntry
    reconstruction_identity: str | None
    artifact_dir: Path | None
    evaluations: tuple[EvaluatorRecord, ...]
    prediction_cache_mode: PredictionCacheMode

    @property
    def scheduled_evaluator_count(self) -> int:
        return len(self.entry.evaluator_kinds)

    @property
    def succeeded(self) -> bool:
        return all(
            item.status in {EvaluatorStatus.PASSED, EvaluatorStatus.SKIPPED}
            for item in self.evaluations
        )


def _manifest_digest(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def matrix_cache_mode(
    requested: PredictionCacheMode,
    entry_index: int,
) -> PredictionCacheMode:
    if requested is PredictionCacheMode.REFRESH and entry_index > 0:
        return PredictionCacheMode.READONLY
    return requested


def _entry_overrides(
    experiment: ExperimentConfig,
    entry: MatrixEntry,
    entry_index: int,
    overrides: Sequence[str],
):
    values = [
        *overrides,
        f"segmentation.method={entry.segmentation_method.value}",
        f"reconstruction.mode={entry.reconstruction_mode.value}",
    ]
    if experiment.evaluation is EvaluationKind.POINTCLOUD:
        values.append("segmentation.confidence_quantile_method=nearest")
    preliminary = load_pipeline_config(experiment.reconstruction_config, values)
    cache_mode = matrix_cache_mode(
        preliminary.config.prediction_cache.mode,
        entry_index,
    )
    values.append(f"prediction_cache.mode={cache_mode.value}")
    return tuple(values), cache_mode


def _capability_entry_overrides(
    experiment: CapabilityExperimentConfig,
    entry: CapabilityMatrixEntry,
    entry_index: int,
    overrides: Sequence[str],
) -> tuple[tuple[str, ...], PredictionCacheMode]:
    if (
        entry.reconstruction_mode.value == "traditional"
        and entry.window_reference_enabled
    ):
        raise ValueError("traditional refinement must be disabled")
    values = [
        *overrides,
        f"segmentation.method={entry.segmentation_method.value}",
        "segmentation.window_reference.enabled="
        f"{str(entry.window_reference_enabled).lower()}",
        f"reconstruction.mode={entry.reconstruction_mode.value}",
    ]
    preliminary = load_pipeline_config(experiment.reconstruction_config, values)
    cache_mode = matrix_cache_mode(
        preliminary.config.prediction_cache.mode,
        entry_index,
    )
    values.append(f"prediction_cache.mode={cache_mode.value}")
    return tuple(values), cache_mode


def _evaluator_source_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parents[1]
    return tuple(
        sorted(
            {
                Path(__file__).resolve(),
                (root / "experiments" / "evaluation_bundle.py").resolve(),
                (root / "experiments" / "ate.py").resolve(),
                (root / "experiments" / "pointcloud.py").resolve(),
                (root / "pipeline" / "artifacts.py").resolve(),
                *(
                    path.resolve()
                    for path in (root / "evaluation").rglob("*.py")
                ),
            },
            key=lambda path: path.as_posix(),
        )
    )


def _source_revision(
    source_paths: Sequence[str | Path] | None = None,
) -> str:
    root = Path(__file__).resolve().parents[1]
    sources = tuple(
        Path(path).resolve()
        for path in (
            _evaluator_source_paths()
            if source_paths is None
            else source_paths
        )
    )
    if not sources:
        raise ValueError("evaluator source set must not be empty")
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda path: path.as_posix()):
        if not source.is_file():
            raise FileNotFoundError(
                f"evaluator source does not exist: {source}"
            )
        try:
            label = source.relative_to(root).as_posix()
        except ValueError:
            label = source.as_posix()
        label_bytes = label.encode("utf-8")
        content = source.read_bytes()
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
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


def _evaluation_payload(record: EvaluatorRecord) -> dict[str, object]:
    return {
        "kind": record.kind.value,
        "status": record.status.value,
        "identity_digest": record.identity_digest,
        "output_path": record.output_path,
        "error": (
            None
            if record.error is None
            else {
                "exception_type": record.error.exception_type,
                "message": record.error.message,
            }
        ),
    }


def _write_capability_summary(
    experiment: CapabilityExperimentConfig,
    records: Sequence[CapabilityExperimentRunRecord],
) -> Path:
    payload = {
        "schema_version": 1,
        "overall_success": all(record.succeeded for record in records),
        "entries": [
            {
                "entry": record.entry.name,
                "segmentation_method": record.entry.segmentation_method.value,
                "reconstruction_mode": record.entry.reconstruction_mode.value,
                "window_reference_enabled": (
                    record.entry.window_reference_enabled
                ),
                "reconstruction_identity": record.reconstruction_identity,
                "artifact_dir": (
                    None
                    if record.artifact_dir is None
                    else str(record.artifact_dir)
                ),
                "prediction_cache_mode": record.prediction_cache_mode.value,
                "succeeded": record.succeeded,
                "evaluations": [
                    _evaluation_payload(item) for item in record.evaluations
                ],
            }
            for record in records
        ],
    }
    return _atomic_json(
        Path(experiment.output_root) / "evaluation_summary.json",
        payload,
    )


def run_capability_matrix(
    experiment: CapabilityExperimentConfig,
    *,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    runner_factory: Callable = PipelineRunner,
    bundle_runner: EvaluationBundleRunner | None = None,
    source_revision_provider: Callable[[], str] = _source_revision,
) -> tuple[CapabilityExperimentRunRecord, ...]:
    if not isinstance(experiment, CapabilityExperimentConfig):
        raise ValueError("capability runner requires version-2 config")
    repository = ArtifactRepository(Path(experiment.output_root) / "artifacts")
    bundle = bundle_runner or EvaluationBundleRunner()
    source_revision = None
    if not dry_run:
        source_revision = source_revision_provider()
        if not isinstance(source_revision, str) or not source_revision:
            raise ValueError("source revision provider returned invalid value")
    records = []
    for entry_index, entry in enumerate(experiment.entries):
        entry_overrides, cache_mode = _capability_entry_overrides(
            experiment,
            entry,
            entry_index,
            overrides,
        )
        loaded = load_pipeline_config(
            experiment.reconstruction_config,
            entry_overrides,
        )
        manifest = discover_image_manifest(
            loaded.config.input.image_dir,
            loaded.config.input.sample_stride,
        )
        identity = reconstruction_identity(
            loaded,
            input_manifest_sha256=_manifest_digest(manifest.paths),
            checkpoint_sha256=sha256_file(loaded.config.model.checkpoint),
        )
        artifact_dir = repository.path_for(identity)
        evaluation_root = (
            Path(experiment.output_root) / "evaluation" / entry.name
        )
        evaluations: tuple[EvaluatorRecord, ...] = ()
        if dry_run:
            print(
                f"{entry.name} window={loaded.config.window.size} "
                f"overlap={loaded.config.window.overlap} "
                f"cache={cache_mode.value} identity={identity} "
                f"artifact={artifact_dir}"
            )
        else:

            def reconstruct(target: Path) -> Path:
                runner = runner_factory(loaded, artifact_output_dir=target)
                runner.run()
                return target

            try:
                artifact_dir = repository.get_or_create(identity, reconstruct)
            except Exception as error:
                evaluations = bundle.blocked(
                    entry=entry,
                    experiment=experiment,
                    output_root=evaluation_root,
                    error=error,
                )
                records.append(
                    CapabilityExperimentRunRecord(
                        entry=entry,
                        reconstruction_identity=None,
                        artifact_dir=None,
                        evaluations=evaluations,
                        prediction_cache_mode=cache_mode,
                    )
                )
                continue
            evaluations = bundle.run(
                artifact_dir=artifact_dir,
                reconstruction_identity=identity,
                entry=entry,
                experiment=experiment,
                output_root=evaluation_root,
                source_revision=source_revision,
            )
        records.append(
            CapabilityExperimentRunRecord(
                entry=entry,
                reconstruction_identity=identity,
                artifact_dir=artifact_dir,
                evaluations=evaluations,
                prediction_cache_mode=cache_mode,
            )
        )
    result = tuple(records)
    if not dry_run:
        _write_capability_summary(experiment, result)
    return result


def _run_legacy_matrix(
    experiment: ExperimentConfig,
    *,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    runner_factory: Callable = PipelineRunner,
) -> tuple[ExperimentRunRecord, ...]:
    repository = ArtifactRepository(Path(experiment.output_root) / "artifacts")
    records = []
    for entry_index, entry in enumerate(experiment.entries):
        validate_matrix_entry(experiment.evaluation, entry)
        entry_overrides, cache_mode = _entry_overrides(
            experiment,
            entry,
            entry_index,
            overrides,
        )
        loaded = load_pipeline_config(
            experiment.reconstruction_config,
            entry_overrides,
        )
        manifest = discover_image_manifest(
            loaded.config.input.image_dir,
            loaded.config.input.sample_stride,
        )
        identity = reconstruction_identity(
            loaded,
            input_manifest_sha256=_manifest_digest(manifest.paths),
            checkpoint_sha256=sha256_file(loaded.config.model.checkpoint),
        )
        artifact_dir = repository.path_for(identity)
        evaluation_output = (
            Path(experiment.output_root)
            / "evaluation"
            / experiment.evaluation.value
            / entry.name
        )
        if dry_run:
            print(
                f"{entry.name} window={loaded.config.window.size} "
                f"overlap={loaded.config.window.overlap} "
                f"cache={cache_mode.value} identity={identity} "
                f"artifact={artifact_dir}"
            )
        else:
            def reconstruct(target: Path) -> Path:
                runner = runner_factory(loaded, artifact_output_dir=target)
                runner.run()
                return target

            artifact_dir = repository.get_or_create(identity, reconstruct)
            evaluator = (
                evaluate_ate_artifact
                if experiment.evaluation is EvaluationKind.ATE
                else evaluate_pointcloud_artifact
            )
            evaluator(artifact_dir, experiment, evaluation_output)
        records.append(
            ExperimentRunRecord(
                entry,
                identity,
                artifact_dir,
                evaluation_output,
                cache_mode,
            )
        )
    return tuple(records)


def run_matrix(
    experiment: ExperimentConfig | CapabilityExperimentConfig,
    *,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    runner_factory: Callable = PipelineRunner,
    bundle_runner: EvaluationBundleRunner | None = None,
) -> tuple[ExperimentRunRecord, ...] | tuple[CapabilityExperimentRunRecord, ...]:
    if isinstance(experiment, CapabilityExperimentConfig):
        return run_capability_matrix(
            experiment,
            overrides=overrides,
            dry_run=dry_run,
            runner_factory=runner_factory,
            bundle_runner=bundle_runner,
        )
    if isinstance(experiment, ExperimentConfig):
        if bundle_runner is not None:
            raise ValueError("legacy matrix does not accept evaluator bundle")
        return _run_legacy_matrix(
            experiment,
            overrides=overrides,
            dry_run=dry_run,
            runner_factory=runner_factory,
        )
    raise ValueError("experiment config type is invalid")
