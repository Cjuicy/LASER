from __future__ import annotations

import importlib.metadata
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from inference_engine.prediction_cache.fingerprint import sha256_file
from mv_recon.geometry_metrics import (
    GeometryBackend,
    GeometryEvaluation,
    Open3DGeometryBackend,
    evaluate_point_maps,
)
from mv_recon.protocol import (
    PAPER_DATASETS,
    DatasetPlan,
    GeometryProtocol,
    ResolvedEvaluationProtocol,
    SequenceSpec,
    build_dataset_plans,
    digest_ground_truth,
    manifest_digest_for_paths,
    resolve_evaluation_protocol,
    validate_dataset_plan,
)
from mv_recon.results import (
    METRIC_SCHEMA_VERSION,
    FailureRecord,
    ResultStore,
    RunIdentity,
    RunResults,
    SequenceResult,
)
from pipeline.config import PipelineConfig
from pipeline.diagnostics import collect_prediction_diagnostics
from pipeline.manifest import ImageManifest
from pipeline.runner import (
    StreamingPipelineModel,
    require_local_model_checkpoint,
    run_windows,
)
from utils.load_fn import load_and_preprocess_images
from utils.messages import set_default_arg


LOGGER = logging.getLogger("mv_recon-eval")


@dataclass(frozen=True)
class InferenceOutput:
    points: np.ndarray
    confidence: np.ndarray
    ordinary_prediction_key: str
    cache_diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class LoadedSequence:
    image_paths: tuple[str, ...]
    images: torch.Tensor
    ground_truth_points: np.ndarray
    valid_mask: np.ndarray
    input_manifest_sha256: str
    ground_truth_sha256: str


InferPointMaps = Callable[
    [list[str], object, DictConfig, tuple[int, int]],
    InferenceOutput,
]
EvaluateGeometry = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        GeometryProtocol,
        GeometryBackend,
    ],
    GeometryEvaluation,
]


@dataclass(frozen=True)
class EvaluationDependencies:
    instantiate_dataset: Callable[[DictConfig], object]
    model_factory: Callable[[PipelineConfig], object]
    infer_point_maps: InferPointMaps
    evaluate_geometry: EvaluateGeometry
    geometry_backend_factory: Callable[[], GeometryBackend]
    checkpoint_digest: Callable[[Path], str]
    git_commit: Callable[[], str]
    runtime_metadata: Callable[[], Mapping[str, object]]
    empty_cuda_cache: Callable[[], None]


class EvaluationFailed(RuntimeError):
    """A strict sequence failure preserved partial artifacts and stopped."""


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        if distribution == "open3d":
            raise RuntimeError(
                "strict LASER paper evaluation requires Open3D; create the "
                "repository Python 3.11 environment and install requirements"
            ) from exc
        raise RuntimeError(
            f"required evaluation dependency is not installed: {distribution}"
        ) from exc


def _collect_runtime_metadata() -> Mapping[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    capability = (
        tuple(int(value) for value in torch.cuda.get_device_capability())
        if cuda_available
        else None
    )
    gpu_name = torch.cuda.get_device_name() if cuda_available else None
    return {
        "python": sys.version.split()[0],
        "open3d": _package_version("open3d"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_capability": capability,
        "gpu_name": gpu_name,
    }


def _default_evaluate_geometry(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    geometry: GeometryProtocol,
    backend: GeometryBackend,
) -> GeometryEvaluation:
    return evaluate_point_maps(
        predicted,
        ground_truth,
        valid_mask,
        geometry,
        backend=backend,
    )


def default_dependencies() -> EvaluationDependencies:
    return EvaluationDependencies(
        instantiate_dataset=hydra.utils.instantiate,
        model_factory=lambda config: StreamingPipelineModel(config).eval(),
        infer_point_maps=run_streaming_inference,
        evaluate_geometry=_default_evaluate_geometry,
        geometry_backend_factory=Open3DGeometryBackend,
        checkpoint_digest=lambda path: sha256_file(path),
        git_commit=_git_commit,
        runtime_metadata=_collect_runtime_metadata,
        empty_cuda_cache=torch.cuda.empty_cache,
    )


def run_streaming_inference(
    filelist: list[str],
    inference_model: object,
    hydra_cfg: DictConfig,
    data_size: tuple[int, int],
) -> InferenceOutput:
    images = load_and_preprocess_images(filelist)
    manifest = ImageManifest(
        paths=tuple(Path(path).resolve() for path in filelist)
    )
    engine = inference_model.prepare(images, manifest)
    config = engine.pipeline_config
    store = engine.prediction_store
    with store.entry_lock():
        caches = run_windows(
            engine,
            manifest,
            images,
            engine.window_specs,
            config,
        )
    constraints = engine.loop_strategy.build_constraints(caches, ())
    solution = engine.loop_strategy.optimize(caches, constraints)
    result = engine.loop_strategy.aggregate(caches, solution)
    points = result.payload["points"]
    resized_points = F.interpolate(
        points.permute(0, 3, 1, 2),
        data_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)
    prediction_diagnostics = collect_prediction_diagnostics(
        config=config,
        fingerprint=store.fingerprint,
        model=engine.model_handle,
        store=store,
    )
    return InferenceOutput(
        points=resized_points.detach().cpu().numpy(),
        confidence=result.payload["confidence"].detach().cpu().numpy(),
        ordinary_prediction_key=store.fingerprint.key,
        cache_diagnostics=prediction_diagnostics,
    )


def _load_sequence(
    dataset: object,
    sequence: SequenceSpec,
) -> LoadedSequence:
    data = dataset.get_data(
        sequence_name=sequence.name,
        ids=list(sequence.frame_ids),
    )
    required = {"image_paths", "images", "pointclouds", "valid_mask"}
    if not isinstance(data, Mapping) or not required <= set(data):
        missing = sorted(required - set(data) if isinstance(data, Mapping) else required)
        raise ValueError(f"dataset sequence data is missing fields: {missing}")
    image_paths = tuple(str(path) for path in data["image_paths"])
    images = data["images"]
    ground_truth = np.asarray(data["pointclouds"])
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    expected_frames = len(sequence.frame_ids)
    if len(image_paths) != expected_frames:
        raise ValueError(
            f"sequence {sequence.name} returned {len(image_paths)} image paths; "
            f"expected {expected_frames}"
        )
    if (
        not isinstance(images, torch.Tensor)
        or images.ndim != 4
        or images.shape[0] != expected_frames
        or images.shape[1] != 3
    ):
        raise ValueError(
            f"sequence {sequence.name} images must have shape (N,3,H,W)"
        )
    if (
        ground_truth.ndim != 4
        or ground_truth.shape[0] != expected_frames
        or ground_truth.shape[-1] != 3
    ):
        raise ValueError(
            f"sequence {sequence.name} point maps must have shape (N,H,W,3)"
        )
    if valid_mask.shape != ground_truth.shape[:-1]:
        raise ValueError(
            f"sequence {sequence.name} valid mask does not match point maps"
        )
    if tuple(images.shape[-2:]) != tuple(ground_truth.shape[1:3]):
        raise ValueError(
            f"sequence {sequence.name} image and point-map sizes do not match"
        )
    return LoadedSequence(
        image_paths=image_paths,
        images=images,
        ground_truth_points=ground_truth,
        valid_mask=valid_mask,
        input_manifest_sha256=manifest_digest_for_paths(image_paths),
        ground_truth_sha256=digest_ground_truth(
            ground_truth,
            valid_mask,
        ),
    )


def run_sequence(
    *,
    dataset_name: str,
    sequence: SequenceSpec,
    loaded: LoadedSequence,
    model: object,
    hydra_cfg: DictConfig,
    geometry: GeometryProtocol,
    backend: GeometryBackend,
    infer_point_maps: InferPointMaps,
    evaluate_geometry: EvaluateGeometry,
) -> SequenceResult:
    data_size = tuple(int(value) for value in loaded.images.shape[-2:])
    inference = infer_point_maps(
        list(loaded.image_paths),
        model,
        hydra_cfg,
        data_size,
    )
    geometry_result = evaluate_geometry(
        np.asarray(inference.points),
        loaded.ground_truth_points,
        loaded.valid_mask,
        geometry,
        backend,
    )
    return SequenceResult(
        dataset=dataset_name,
        sequence=sequence.name,
        frame_count=len(sequence.frame_ids),
        input_manifest_sha256=loaded.input_manifest_sha256,
        ground_truth_sha256=loaded.ground_truth_sha256,
        ordinary_prediction_key=inference.ordinary_prediction_key,
        primary=geometry_result.primary,
        diagnostics=geometry_result.diagnostics,
        cache_diagnostics=dict(inference.cache_diagnostics),
    )


def _validate_runtime_device(
    resolved: ResolvedEvaluationProtocol,
    runtime_metadata: Mapping[str, object],
) -> None:
    inference_device = resolved.pipeline.config.model.inference_device
    if inference_device.startswith("cuda") and not bool(
        runtime_metadata.get("cuda_available", False)
    ):
        raise RuntimeError(
            "CUDA was requested by the paper profile but is unavailable"
        )


def _instantiate_and_validate_datasets(
    hydra_cfg: DictConfig,
    plans: tuple[DatasetPlan, ...],
    dependencies: EvaluationDependencies,
) -> dict[str, object]:
    datasets = {}
    for plan in plans:
        dataset = dependencies.instantiate_dataset(
            hydra_cfg.data[plan.name].cfg
        )
        validate_dataset_plan(dataset, plan)
        datasets[plan.name] = dataset
    return datasets


def _build_protocol_manifest(
    *,
    resolved: ResolvedEvaluationProtocol,
    plans: tuple[DatasetPlan, ...],
    checkpoint_sha256: str,
    runtime_metadata: Mapping[str, object],
    git_commit: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "git_commit": git_commit,
        "evaluation_mode": resolved.protocol.mode,
        "segmentation_method": (
            resolved.pipeline.config.segmentation.method.value
        ),
        "resolved_protocol_sha256": resolved.sha256,
        "protocol_identity_sha256": resolved.identity_sha256,
        "resolved_pipeline_sha256": resolved.pipeline.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "sequence_map_sha256": {
            plan.name: plan.sequence_map_sha256 for plan in plans
        },
        "runtime": dict(runtime_metadata),
        "pipeline": asdict(resolved.pipeline.config),
        "selected_sequences": {
            plan.name: [sequence.name for sequence in plan.sequences]
            for plan in plans
        },
        "selected_sequence_counts": {
            plan.name: len(plan.sequences) for plan in plans
        },
        "expected_sequence_counts": {
            plan.name: plan.expected_sequence_count for plan in plans
        },
        "attempted_sequences": 0,
        "successful_sequences": 0,
        "failed_sequences": 0,
        "sequence_cache": [],
    }


def run_evaluation(
    hydra_cfg: DictConfig,
    *,
    dependencies: EvaluationDependencies | None = None,
    repository_root: str | Path | None = None,
) -> RunResults:
    selected_dependencies = dependencies or default_dependencies()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    runtime_metadata = dict(selected_dependencies.runtime_metadata())
    raw_capability = runtime_metadata.get("cuda_capability")
    capability = (
        tuple(int(value) for value in raw_capability)
        if raw_capability is not None
        else None
    )
    resolved = resolve_evaluation_protocol(
        hydra_cfg,
        root,
        cuda_capability=capability,
    )
    _validate_runtime_device(resolved, runtime_metadata)
    plans = build_dataset_plans(resolved, hydra_cfg.data, root)
    datasets = _instantiate_and_validate_datasets(
        hydra_cfg,
        plans,
        selected_dependencies,
    )
    local_checkpoint = Path(
        require_local_model_checkpoint(resolved.pipeline.config.model.checkpoint)
    )
    checkpoint_sha256 = selected_dependencies.checkpoint_digest(
        local_checkpoint
    )
    identity = RunIdentity(
        protocol_identity_sha256=resolved.identity_sha256,
        pipeline_sha256=resolved.pipeline.sha256,
        checkpoint_sha256=checkpoint_sha256,
        sequence_map_sha256={
            plan.name: plan.sequence_map_sha256 for plan in plans
        },
    )
    store = ResultStore(
        resolved.output_dir,
        identity,
        resume=resolved.protocol.resume,
    )
    expected_sequences = {
        plan.name: tuple(sequence.name for sequence in plan.sequences)
        for plan in plans
    }
    full_sequence_counts = {
        plan.name: plan.expected_sequence_count for plan in plans
    }
    subset = (
        resolved.protocol.mode == "comparison"
        or resolved.datasets != PAPER_DATASETS
        or any(
            len(plan.sequences) != plan.expected_sequence_count
            for plan in plans
        )
    )
    selected_references = {
        dataset: resolved.protocol.paper_reference[dataset]
        for dataset in resolved.datasets
    }
    initial_result = store.initialize(
        expected_sequences=expected_sequences,
        full_sequence_counts=full_sequence_counts,
        paper_reference=selected_references,
        subset=subset,
        preflight=resolved.protocol.preflight_only,
    )
    protocol_manifest = _build_protocol_manifest(
        resolved=resolved,
        plans=plans,
        checkpoint_sha256=checkpoint_sha256,
        runtime_metadata=runtime_metadata,
        git_commit=selected_dependencies.git_commit(),
    )
    store.write_protocol_artifacts(
        resolved_protocol_yaml=resolved.resolved_yaml,
        resolved_pipeline_yaml=resolved.pipeline.resolved_yaml,
        manifest=protocol_manifest,
    )
    if resolved.protocol.preflight_only:
        LOGGER.info("LASER paper point-map preflight completed without a model")
        return initial_result

    model = selected_dependencies.model_factory(resolved.pipeline.config)
    backend = selected_dependencies.geometry_backend_factory()
    cache_records: list[dict[str, object]] = []
    attempted = 0
    successful = 0
    failed = 0

    def persist_progress(*, run_state: str | None = None) -> None:
        updates: dict[str, object] = {
            "attempted_sequences": attempted,
            "successful_sequences": successful,
            "failed_sequences": failed,
            "sequence_cache": cache_records,
        }
        if run_state is not None:
            updates["run_state"] = run_state
        protocol_manifest.update(updates)
        store.update_protocol_manifest(protocol_manifest)

    for plan in plans:
        dataset = datasets[plan.name]
        for sequence in plan.sequences:
            attempted += 1
            try:
                loaded = _load_sequence(dataset, sequence)
                reused = store.reusable_sequence(
                    plan.name,
                    sequence.name,
                    loaded.input_manifest_sha256,
                    loaded.ground_truth_sha256,
                )
                if reused is not None:
                    successful += 1
                    cache_records.append(
                        {
                            "dataset": plan.name,
                            "sequence": sequence.name,
                            "input_manifest_sha256": (
                                reused.input_manifest_sha256
                            ),
                            "ground_truth_sha256": (
                                reused.ground_truth_sha256
                            ),
                            "ordinary_prediction_key": (
                                reused.ordinary_prediction_key
                            ),
                            "resumed_result": True,
                            **dict(reused.cache_diagnostics),
                        }
                    )
                    persist_progress()
                    continue
                sequence_result = run_sequence(
                    dataset_name=plan.name,
                    sequence=sequence,
                    loaded=loaded,
                    model=model,
                    hydra_cfg=hydra_cfg,
                    geometry=resolved.protocol.geometry,
                    backend=backend,
                    infer_point_maps=selected_dependencies.infer_point_maps,
                    evaluate_geometry=selected_dependencies.evaluate_geometry,
                )
                store.record_sequence(sequence_result)
                successful += 1
                cache_records.append(
                    {
                        "dataset": plan.name,
                        "sequence": sequence.name,
                        "input_manifest_sha256": (
                            sequence_result.input_manifest_sha256
                        ),
                        "ground_truth_sha256": (
                            sequence_result.ground_truth_sha256
                        ),
                        "ordinary_prediction_key": (
                            sequence_result.ordinary_prediction_key
                        ),
                        "resumed_result": False,
                        **dict(sequence_result.cache_diagnostics),
                    }
                )
            except KeyboardInterrupt as exc:
                failed += 1
                failure = FailureRecord(
                    dataset=plan.name,
                    sequence=sequence.name,
                    category=type(exc).__name__,
                    message=str(exc),
                )
                store.record_failure(failure)
                interrupted_result = store.finalize()
                persist_progress(run_state=interrupted_result.state)
                raise
            except Exception as exc:
                failed += 1
                failure = FailureRecord(
                    dataset=plan.name,
                    sequence=sequence.name,
                    category=type(exc).__name__,
                    message=str(exc),
                )
                store.record_failure(failure)
                failed_result = store.finalize()
                persist_progress(run_state=failed_result.state)
                raise EvaluationFailed(
                    f"strict point-map evaluation failed at "
                    f"{plan.name}/{sequence.name}: {exc}"
                ) from exc
            finally:
                selected_dependencies.empty_cuda_cache()
            persist_progress()

    result = store.finalize()
    persist_progress(run_state=result.state)
    LOGGER.info("Finished LASER paper point-map evaluation: %s", result.state)
    return result


@hydra.main(
    version_base="1.2",
    config_path="../configs",
    config_name="eval_mv_recon_dense",
)
def main(hydra_cfg: DictConfig) -> None:
    run_evaluation(hydra_cfg)


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon_laser_paper")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    with torch.no_grad():
        main()
