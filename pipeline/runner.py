from __future__ import annotations

import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Sequence

import torch

from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.models.lazy import LazyModelHandle
from inference_engine.models.loader import build_model_adapter
from inference_engine.prediction_cache.fingerprint import (
    PredictionFingerprint,
    build_prediction_fingerprint,
)
from inference_engine.prediction_cache.provider import OrdinaryPredictionProvider
from inference_engine.prediction_cache.store import OrdinaryPredictionStore
from inference_engine.prediction_cache.types import build_window_specs
from inference_engine.segmentation import (
    build_segmentation_strategy,
    build_window_reference_refiner,
)
from loop_closure.constraint_estimation import JointAlignmentEstimator
from loop_closure.detection import SaladLoopDetector
from pipeline.artifacts import (
    ReconstructionArtifact,
    ReconstructionDiagnostics,
    StagedReconstructionArtifacts,
    write_reconstruction_artifact,
    write_staged_reconstruction_artifacts,
)
from pipeline.config import (
    LoadedPipelineConfig,
    ModelConfig,
    ReconstructionMode,
    load_pipeline_config,
)
from pipeline.manifest import ImageManifest, discover_image_manifest
from pipeline.preflight import validate_preflight
from reconstruction.modes.base import ReconstructionContext
from reconstruction.prediction_stream import iter_window_predictions
from reconstruction.registry import (
    ReconstructionServices,
    build_reconstruction_mode,
)


def _load_images(manifest: ImageManifest) -> torch.Tensor:
    from utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images(manifest.as_strings())


def resolve_model_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError:
        raise ValueError(f"unsupported model dtype: {name!r}") from None


def require_local_model_checkpoint(checkpoint: str | Path) -> str:
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "streaming prediction caching requires a local checkpoint file; "
            f"download the model first: {checkpoint}"
        )
    return str(path.resolve())


def build_model_handle(
    config: ModelConfig,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> LazyModelHandle:
    def construct():
        if expected_checkpoint_sha256 is None:
            return build_model_adapter(config)
        return build_model_adapter(
            config,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )

    return LazyModelHandle(
        construct,
        inference_device=config.inference_device,
        dtype=resolve_model_dtype(config.dtype),
    )


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


def _loop_evidence(
    *,
    model,
    images,
    manifest,
    chunk_size,
    confidence_keep_ratio,
):
    return JointAlignmentEstimator(
        model=model,
        images=images,
        manifest=manifest,
        chunk_size=chunk_size,
        confidence_keep_ratio=confidence_keep_ratio,
    )


@dataclass(frozen=True)
class PipelineDependencies:
    validate_preflight: Callable = validate_preflight
    build_model_handle: Callable = build_model_handle
    build_prediction_fingerprint: Callable = build_prediction_fingerprint
    build_prediction_store: Callable = OrdinaryPredictionStore
    build_prediction_provider: Callable = OrdinaryPredictionProvider
    load_images: Callable = _load_images
    build_segmentation_strategy: Callable = build_segmentation_strategy
    build_window_reference_refiner: Callable = build_window_reference_refiner
    build_anchor_propagator: Callable = AnchorPropagator
    build_reconstruction_mode: Callable = build_reconstruction_mode
    build_loop_detector: Callable = SaladLoopDetector
    build_loop_evidence: Callable = _loop_evidence
    write_artifact: Callable = write_reconstruction_artifact
    write_staged_artifacts: Callable = write_staged_reconstruction_artifacts
    cuda_available: Callable = torch.cuda.is_available
    git_commit: Callable = _git_commit


def _build_services(
    mode: ReconstructionMode,
    *,
    loaded: LoadedPipelineConfig,
    dependencies: PipelineDependencies,
    model,
    manifest: ImageManifest,
    images: torch.Tensor,
    output_dir: Path,
) -> ReconstructionServices:
    if mode is ReconstructionMode.NO_LOOP:
        return ReconstructionServices()
    loop = loaded.config.loop
    if loop is None:
        raise ValueError(f"{mode.value} requires loop configuration")
    detector = dependencies.build_loop_detector(
        loop.detection,
        output_path=(
            output_dir.parent / f"{output_dir.name}.loop_candidates.json"
        ),
    )
    evidence = dependencies.build_loop_evidence(
        model=model,
        images=images,
        manifest=manifest,
        chunk_size=loop.constraint.chunk_size,
        confidence_keep_ratio=(
            loaded.config.registration.confidence_keep_ratio
        ),
    )
    return ReconstructionServices(
        detector=detector,
        evidence=evidence,
        optimizer_config=loop.optimizer,
    )


class PipelineRunner:
    def __init__(
        self,
        loaded: LoadedPipelineConfig,
        *,
        dependencies: PipelineDependencies | None = None,
        artifact_output_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(loaded, LoadedPipelineConfig):
            raise ValueError("PipelineRunner requires LoadedPipelineConfig")
        self.loaded = loaded
        self.dependencies = dependencies or PipelineDependencies()
        self._artifact_output_dir = (
            None if artifact_output_dir is None else Path(artifact_output_dir)
        )
        self.artifact_dir: Path | None = None
        self.stage_artifact_dirs = MappingProxyType({})

    def run(self) -> ReconstructionArtifact:
        config = self.loaded.config
        dependencies = self.dependencies
        started = time.perf_counter()
        manifest = discover_image_manifest(
            config.input.image_dir,
            config.input.sample_stride,
        )
        dependencies.validate_preflight(
            config,
            manifest,
            dependencies.cuda_available(),
        )
        manifest_ms = (time.perf_counter() - started) * 1000

        images = dependencies.load_images(manifest)
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[0] != len(manifest)
            or images.shape[1] != 3
        ):
            raise ValueError("loaded images must have shape (N,3,H,W)")
        specs = build_window_specs(
            len(manifest),
            config.window.size,
            config.window.overlap,
        )
        fingerprint = dependencies.build_prediction_fingerprint(
            model=config.model,
            manifest=manifest,
            image_shape=tuple(int(size) for size in images.shape),
            sample_stride=config.input.sample_stride,
            window_size=config.window.size,
            overlap=config.window.overlap,
            specs=specs,
        )
        store = dependencies.build_prediction_store(
            root=config.prediction_cache.root,
            fingerprint=fingerprint,
            mode=config.prediction_cache.mode,
            expected_specs=specs,
        )
        model = dependencies.build_model_handle(
            config.model,
            expected_checkpoint_sha256=fingerprint.checkpoint_sha256,
        )
        provider = dependencies.build_prediction_provider(
            store=store,
            model=model,
        )
        segmenter = dependencies.build_segmentation_strategy(
            config.segmentation
        )
        window_reference_refiner = dependencies.build_window_reference_refiner(
            config.segmentation
        )
        anchor = dependencies.build_anchor_propagator(
            config.anchor_propagation.correspondence_iou_threshold
        )
        output_dir = self._artifact_output_dir or (
            Path(config.output.result_dir)
            / config.output.scene_name
            / f"{config.segmentation.method.value}-{config.reconstruction.mode.value}"
        )
        services = _build_services(
            config.reconstruction.mode,
            loaded=self.loaded,
            dependencies=dependencies,
            model=model,
            manifest=manifest,
            images=images,
            output_dir=output_dir,
        )
        mode = dependencies.build_reconstruction_mode(
            config.reconstruction.mode,
            services,
        )
        predictions = iter_window_predictions(
            provider,
            specs,
            images,
            config.model.process_device,
        )
        context = ReconstructionContext(
            predictions=predictions,
            frame_ids=tuple(range(len(manifest))),
            segmentation_strategy=segmenter,
            window_reference_refiner=window_reference_refiner,
            anchor_propagator=anchor,
            segmentation_config=config.segmentation,
            anchor_config=config.anchor_propagation,
            registration_config=config.registration,
            window_config=config.window,
            reconstruction_mode=config.reconstruction.mode,
            image_manifest=manifest,
            images=images,
        )
        lock_factory = getattr(store, "entry_lock", None)
        lock = lock_factory() if callable(lock_factory) else nullcontext()
        reconstruction_started = time.perf_counter()
        with lock:
            result = mode.run(context)
        reconstruction_ms = (time.perf_counter() - reconstruction_started) * 1000
        artifact = (
            result.primary
            if isinstance(result, StagedReconstructionArtifacts)
            else result
        )
        if artifact.reconstruction_mode is not config.reconstruction.mode:
            raise ValueError("reconstruction mode returned the wrong artifact mode")
        if artifact.segmentation_method is not config.segmentation.method:
            raise ValueError(
                "reconstruction mode returned the wrong segmentation method"
            )
        artifact = replace(
            artifact,
            diagnostics=replace(
                artifact.diagnostics,
                stage_timings_ms={
                    **dict(artifact.diagnostics.stage_timings_ms),
                    "manifest_preflight": manifest_ms,
                    "reconstruction": reconstruction_ms,
                },
            ),
        )
        metadata = {
            "resolved_yaml": self.loaded.resolved_yaml,
            "config_sha256": self.loaded.sha256,
            "checkpoint_sha256": fingerprint.checkpoint_sha256,
            "git_commit": dependencies.git_commit(),
        }
        if isinstance(result, StagedReconstructionArtifacts):
            staged = replace(result, stage2=artifact)
            paths = dependencies.write_staged_artifacts(
                staged,
                output_dir,
                **metadata,
            )
            self.stage_artifact_dirs = MappingProxyType(dict(paths))
            self.artifact_dir = self.stage_artifact_dirs["stage2"]
        else:
            self.artifact_dir = dependencies.write_artifact(
                artifact,
                output_dir,
                **metadata,
            )
            self.stage_artifact_dirs = MappingProxyType(
                {"stage2": self.artifact_dir}
            )
        return artifact


def run_from_config(
    path: str | Path,
    overrides: Sequence[str] = (),
    *,
    dependencies: PipelineDependencies | None = None,
) -> ReconstructionArtifact:
    return PipelineRunner(
        load_pipeline_config(path, overrides),
        dependencies=dependencies,
    ).run()


__all__ = [
    "PipelineDependencies",
    "PipelineRunner",
    "build_model_handle",
    "require_local_model_checkpoint",
    "resolve_model_dtype",
    "run_from_config",
]
