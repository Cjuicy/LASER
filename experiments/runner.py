from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from inference_engine.prediction_cache.fingerprint import sha256_file
from pipeline.config import PredictionCacheMode, load_pipeline_config
from pipeline.manifest import discover_image_manifest
from pipeline.runner import PipelineRunner

from .ate import evaluate_ate_artifact
from .config import EvaluationKind, ExperimentConfig
from .matrix import (
    ArtifactRepository,
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


def run_matrix(
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
