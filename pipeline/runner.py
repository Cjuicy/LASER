from __future__ import annotations

import glob
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.models.lazy import LazyModelHandle
from inference_engine.models.loader import build_model_adapter
from inference_engine.prediction_cache.fingerprint import (
    PredictionFingerprint,
    build_prediction_fingerprint,
)
from inference_engine.prediction_cache.provider import (
    OrdinaryPredictionProvider,
)
from inference_engine.prediction_cache.store import (
    OrdinaryPredictionStore,
)
from inference_engine.inference_utils import (
    estimate_pseudo_depth_and_intrinsics,
)
from inference_engine.segmentation import build_segmentation_strategy
from inference_engine.prediction_cache.types import (
    WindowSpec,
    build_window_specs,
    validate_window_specs,
)
from loop_closure.constraint_estimation import JointAlignmentEstimator
from loop_closure.methods.base import (
    ReconstructionResult,
    WindowCache,
)
from loop_closure.methods.registry import build_loop_strategy
from loop_closure.methods.shared import detect_loop_candidates
from pipeline.config import (
    LoadedPipelineConfig,
    LoopMethod,
    ModelConfig,
    PipelineConfig,
    ReconstructionMode,
    load_pipeline_config,
)
from pipeline.diagnostics import (
    collect_prediction_diagnostics,
    write_diagnostics,
    write_resolved_config,
)
from pipeline.manifest import (
    ImageManifest,
    discover_image_manifest,
)
from pipeline.preflight import validate_preflight


def _load_images(manifest: ImageManifest) -> torch.Tensor:
    from utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images(manifest.as_strings())


def resolve_model_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def require_local_model_checkpoint(checkpoint: str | Path) -> str:
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "streaming prediction caching requires a local checkpoint "
            f"file; download the model first: {checkpoint}"
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


def expected_window_count(
    image_count: int,
    window_size: int,
    overlap: int,
) -> int:
    return len(build_window_specs(image_count, window_size, overlap))


def _legacy_loop_method(config: PipelineConfig) -> LoopMethod:
    if config.reconstruction.mode is ReconstructionMode.NO_LOOP:
        raise ValueError("no_loop requires the dedicated reconstruction mode")
    return LoopMethod(config.reconstruction.mode.value)


def _require_loop_config(config: PipelineConfig):
    if config.loop is None:
        raise ValueError(
            f"{config.reconstruction.mode.value} requires loop configuration"
        )
    return config.loop


def run_windows(
    engine,
    manifest: ImageManifest,
    images: torch.Tensor,
    specs: Sequence[WindowSpec],
    config: PipelineConfig,
) -> tuple[WindowCache, ...]:
    if (
        not isinstance(images, torch.Tensor)
        or images.ndim != 4
        or images.shape[0] != len(manifest)
        or images.shape[1] != 3
    ):
        raise ValueError(
            "images must have shape (manifest_frames,3,H,W)"
        )
    normalized_specs = validate_window_specs(
        specs,
        frame_count=len(manifest),
        window_size=config.window.size,
        overlap=config.window.overlap,
    )
    windows = tuple(
        images[spec.frame_start : spec.frame_end]
        for spec in normalized_specs
    )
    expected = len(normalized_specs)

    engine.begin()
    for spec, window in zip(normalized_specs, windows, strict=True):
        engine(window, window_spec=spec)
    engine.end()

    cache_files = sorted(
        glob.glob(str(engine.temp_cache_dir / "window_cache_*.pt")),
        key=lambda path: int(Path(path).stem.rsplit("_", 1)[-1]),
    )
    caches = tuple(
        WindowCache.from_payload(
            torch.load(
                cache_file,
                map_location="cpu",
                weights_only=False,
            ),
            expected_method=_legacy_loop_method(config),
            expected_prediction_key=engine.prediction_key,
            expected_model_name=engine.model_name,
            expected_checkpoint_digest=engine.checkpoint_digest,
        )
        for cache_file in cache_files
    )
    if len(caches) != expected:
        raise RuntimeError(
            "window cache count mismatch: "
            f"{len(caches)} != {expected}"
        )
    return caches


def _save_for_viser(
    payload,
    scene_name,
    result_dir,
    inverse_extrinsic,
):
    from eval.save_func import save_for_viser

    return save_for_viser(
        payload,
        scene_name,
        result_dir,
        inverse_extrinsic=inverse_extrinsic,
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


@dataclass(frozen=True)
class PipelineDependencies:
    validate_preflight: Callable = validate_preflight
    build_model_handle: Callable = build_model_handle
    build_prediction_fingerprint: Callable = build_prediction_fingerprint
    build_prediction_store: Callable = OrdinaryPredictionStore
    build_prediction_provider: Callable = OrdinaryPredictionProvider
    load_images: Callable = _load_images
    build_segmentation_strategy: Callable = build_segmentation_strategy
    build_anchor_propagator: Callable = AnchorPropagator
    build_loop_strategy: Callable = build_loop_strategy
    run_windows: Callable = run_windows
    detect_loop_candidates: Callable = detect_loop_candidates
    build_constraint_estimator: Callable = JointAlignmentEstimator
    save_for_viser: Callable = _save_for_viser
    cuda_available: Callable = torch.cuda.is_available
    git_commit: Callable = _git_commit


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def complete_reconstruction_payload(
    result: ReconstructionResult,
    images: torch.Tensor,
) -> ReconstructionResult:
    payload = dict(result.payload)
    local_points = torch.as_tensor(payload["local_points"])
    confidence = torch.as_tensor(payload["confidence"])
    camera_poses = torch.as_tensor(payload["camera_poses"])
    depth, intrinsic = estimate_pseudo_depth_and_intrinsics(local_points)
    payload.update(
        {
            "images": images.detach().cpu(),
            "depth": depth.detach().cpu(),
            "depth_conf": confidence.detach().cpu(),
            "intrinsic": intrinsic.detach().cpu(),
            "extrinsic": camera_poses.detach().cpu(),
        }
    )
    return ReconstructionResult(
        payload=payload,
        summary=dict(result.summary),
    )


def _viser_payload(payload) -> dict[str, np.ndarray]:
    required = (
        "images",
        "depth",
        "depth_conf",
        "intrinsic",
        "extrinsic",
    )
    return {key: _to_numpy(payload[key]) for key in required}


def build_default_window_engine(
    config: PipelineConfig,
    prediction_provider,
    fingerprint: PredictionFingerprint,
):
    segmenter = build_segmentation_strategy(config.segmentation)
    anchor = AnchorPropagator(
        config.anchor_propagation.correspondence_iou_threshold
    )
    loop_config = _require_loop_config(config)
    loop_strategy = build_loop_strategy(
        _legacy_loop_method(config),
        optimizer_config=loop_config.optimizer,
        registration_confidence_keep_ratio=(
            config.registration.confidence_keep_ratio
        ),
    )
    engine = loop_strategy.create_window_engine(
        delegate=prediction_provider,
        inference_device=config.model.inference_device,
        dtype=resolve_model_dtype(config.model.dtype),
        segmentation_strategy=segmenter,
        anchor_propagator=anchor,
        registration_confidence_keep_ratio=(
            config.registration.confidence_keep_ratio
        ),
        anchor_enabled=config.anchor_propagation.enabled,
        temporal_iou_threshold=(
            config.segmentation.temporal_iou_threshold
        ),
        window_size=config.window.size,
        overlap=config.window.overlap,
        cache_root=config.output.cache_dir,
        intermediate_device=config.model.inference_device,
        process_device=config.model.process_device,
        prediction_key=fingerprint.key,
        model_name=config.model.name,
        checkpoint_digest=fingerprint.checkpoint_sha256,
    )
    engine.pipeline_config = config
    engine.loop_strategy = loop_strategy
    return engine


def prepare_default_window_engine(
    config: PipelineConfig,
    images: torch.Tensor,
    manifest: ImageManifest,
    model_handle: LazyModelHandle,
    *,
    specs: Sequence[WindowSpec] | None = None,
    fingerprint: PredictionFingerprint | None = None,
):
    if specs is None:
        specs = build_window_specs(
            len(manifest),
            config.window.size,
            config.window.overlap,
        )
    if fingerprint is None:
        fingerprint = build_prediction_fingerprint(
            model=config.model,
            manifest=manifest,
            image_shape=tuple(int(size) for size in images.shape),
            sample_stride=config.input.sample_stride,
            window_size=config.window.size,
            overlap=config.window.overlap,
            specs=specs,
        )
    store = OrdinaryPredictionStore(
        root=config.prediction_cache.root,
        fingerprint=fingerprint,
        mode=config.prediction_cache.mode,
        expected_specs=specs,
    )
    provider = OrdinaryPredictionProvider(
        store=store,
        model=model_handle,
    )
    engine = build_default_window_engine(
        config,
        provider,
        fingerprint,
    )
    engine.model_handle = model_handle
    engine.prediction_store = store
    engine.window_specs = specs
    return engine


class StreamingPipelineModel(torch.nn.Module):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__()
        if not isinstance(config, PipelineConfig):
            raise ValueError(
                "streaming pipeline model requires a PipelineConfig"
            )
        self.pipeline_config = config
        self.model_handle: LazyModelHandle | None = None
        self._checkpoint_sha256: str | None = None

    def prepare(
        self,
        images: torch.Tensor,
        manifest: ImageManifest,
    ):
        specs = build_window_specs(
            len(manifest),
            self.pipeline_config.window.size,
            self.pipeline_config.window.overlap,
        )
        fingerprint = build_prediction_fingerprint(
            model=self.pipeline_config.model,
            manifest=manifest,
            image_shape=tuple(int(size) for size in images.shape),
            sample_stride=self.pipeline_config.input.sample_stride,
            window_size=self.pipeline_config.window.size,
            overlap=self.pipeline_config.window.overlap,
            specs=specs,
        )
        if self._checkpoint_sha256 != fingerprint.checkpoint_sha256:
            self.model_handle = build_model_handle(
                self.pipeline_config.model,
                expected_checkpoint_sha256=(
                    fingerprint.checkpoint_sha256
                ),
            )
            self._checkpoint_sha256 = fingerprint.checkpoint_sha256
        return prepare_default_window_engine(
            self.pipeline_config,
            images,
            manifest,
            self.model_handle,
            specs=specs,
            fingerprint=fingerprint,
        )


class PipelineRunner:
    def __init__(
        self,
        loaded: LoadedPipelineConfig,
        *,
        dependencies: PipelineDependencies | None = None,
    ) -> None:
        self.loaded = loaded
        self.dependencies = dependencies or PipelineDependencies()

    def run(self) -> ReconstructionResult:
        config = self.loaded.config
        dependencies = self.dependencies
        timings: dict[str, float] = {}

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
        timings["manifest_preflight"] = (
            time.perf_counter() - started
        ) * 1000

        output_root = (
            Path(config.output.result_dir) / config.output.scene_name
        )
        write_resolved_config(output_root, self.loaded)

        started = time.perf_counter()
        images = dependencies.load_images(manifest)
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
            expected_checkpoint_sha256=(
                fingerprint.checkpoint_sha256
            ),
        )
        prediction_provider = dependencies.build_prediction_provider(
            store=store,
            model=model,
        )
        segmenter = dependencies.build_segmentation_strategy(
            config.segmentation
        )
        anchor = dependencies.build_anchor_propagator(
            config.anchor_propagation.correspondence_iou_threshold
        )
        loop_config = _require_loop_config(config)
        loop_strategy = dependencies.build_loop_strategy(
            _legacy_loop_method(config),
            optimizer_config=loop_config.optimizer,
            registration_confidence_keep_ratio=(
                config.registration.confidence_keep_ratio
            ),
        )
        engine = loop_strategy.create_window_engine(
            delegate=prediction_provider,
            inference_device=config.model.inference_device,
            dtype=resolve_model_dtype(config.model.dtype),
            segmentation_strategy=segmenter,
            anchor_propagator=anchor,
            registration_confidence_keep_ratio=(
                config.registration.confidence_keep_ratio
            ),
            anchor_enabled=config.anchor_propagation.enabled,
            temporal_iou_threshold=(
                config.segmentation.temporal_iou_threshold
            ),
            window_size=config.window.size,
            overlap=config.window.overlap,
            cache_root=config.output.cache_dir,
            intermediate_device=config.model.inference_device,
            process_device=config.model.process_device,
            prediction_key=fingerprint.key,
            model_name=config.model.name,
            checkpoint_digest=fingerprint.checkpoint_sha256,
        )
        timings["initialization"] = (
            time.perf_counter() - started
        ) * 1000

        started = time.perf_counter()
        with store.entry_lock():
            caches = tuple(
                dependencies.run_windows(
                    engine,
                    manifest,
                    images,
                    specs,
                    config,
                )
            )
        expected = expected_window_count(
            len(manifest),
            config.window.size,
            config.window.overlap,
        )
        if len(caches) != expected:
            raise RuntimeError(
                f"window cache count mismatch: {len(caches)} != {expected}"
            )
        timings["window_inference"] = (
            time.perf_counter() - started
        ) * 1000

        started = time.perf_counter()
        candidates = (
            dependencies.detect_loop_candidates(
                loop_config.detection,
                manifest,
                output_root / "loop_candidates.json",
            )
        )
        timings["loop_detection"] = (
            time.perf_counter() - started
        ) * 1000

        started = time.perf_counter()
        constraint_estimator = (
            dependencies.build_constraint_estimator(
                model=model,
                images=images,
                manifest=manifest,
                chunk_size=loop_config.constraint.chunk_size,
                confidence_keep_ratio=(
                    config.registration.confidence_keep_ratio
                ),
            )
            if candidates
            else None
        )
        constraints = loop_strategy.build_constraints(
            caches,
            candidates,
            constraint_estimator=constraint_estimator,
        )
        timings["loop_constraints"] = (
            time.perf_counter() - started
        ) * 1000

        started = time.perf_counter()
        solution = loop_strategy.optimize(caches, constraints)
        timings["loop_optimization"] = (
            time.perf_counter() - started
        ) * 1000

        started = time.perf_counter()
        raw_result = loop_strategy.aggregate(caches, solution)
        result = complete_reconstruction_payload(raw_result, images)
        timings["aggregation"] = (
            time.perf_counter() - started
        ) * 1000

        summary = {
            **dict(result.summary),
            "segmentation_method": config.segmentation.method.value,
            "reconstruction_mode": config.reconstruction.mode.value,
        }
        result = ReconstructionResult(
            payload=result.payload,
            summary=summary,
        )
        prediction_diagnostics = collect_prediction_diagnostics(
            config=config,
            fingerprint=fingerprint,
            model=model,
            store=store,
        )
        diagnostics_summary = write_diagnostics(
            output_root,
            self.loaded,
            manifest,
            caches,
            candidates,
            constraints,
            solution,
            result,
            git_commit=dependencies.git_commit(),
            stage_timings_ms=timings,
            prediction_diagnostics=prediction_diagnostics,
        )
        result = ReconstructionResult(
            payload=result.payload,
            summary=diagnostics_summary,
        )
        dependencies.save_for_viser(
            _viser_payload(result.payload),
            config.output.scene_name,
            config.output.result_dir,
            inverse_extrinsic=False,
        )
        return result


def run_from_config(
    path: str | Path,
    overrides: Sequence[str] = (),
    *,
    dependencies: PipelineDependencies | None = None,
) -> ReconstructionResult:
    loaded = load_pipeline_config(path, overrides)
    return PipelineRunner(
        loaded,
        dependencies=dependencies,
    ).run()
