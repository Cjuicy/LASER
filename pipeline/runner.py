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
from inference_engine.inference_utils import (
    estimate_pseudo_depth_and_intrinsics,
)
from inference_engine.segmentation import build_segmentation_strategy
from loop_closure.methods.base import (
    ReconstructionResult,
    WindowCache,
)
from loop_closure.methods.registry import build_loop_strategy
from loop_closure.methods.shared import detect_loop_candidates
from pipeline.config import (
    LoadedPipelineConfig,
    ModelConfig,
    PipelineConfig,
    load_pipeline_config,
)
from pipeline.diagnostics import (
    write_diagnostics,
    write_resolved_config,
)
from pipeline.manifest import (
    ImageManifest,
    discover_image_manifest,
)
from pipeline.preflight import validate_preflight


def _load_pi3(config: ModelConfig):
    from pi3.models.pi3 import Pi3

    checkpoint_path = Path(config.checkpoint)
    if checkpoint_path.suffix.casefold() == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    model = Pi3()
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    return model.to(config.inference_device).eval()


def _load_images(manifest: ImageManifest) -> torch.Tensor:
    from utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images(manifest.as_strings())


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def expected_window_count(
    image_count: int,
    window_size: int,
    overlap: int,
) -> int:
    step = window_size - overlap
    return sum(
        1
        for start in range(0, image_count, step)
        if start == 0 or image_count - start > overlap
    )


def run_windows(
    engine,
    manifest: ImageManifest,
    images: torch.Tensor,
    config: PipelineConfig,
) -> tuple[WindowCache, ...]:
    windows = engine.img_sliding_window(images)
    expected = expected_window_count(
        len(manifest),
        config.window.size,
        config.window.overlap,
    )
    if len(windows) != expected:
        raise RuntimeError(
            "sliding-window generation count mismatch: "
            f"{len(windows)} != {expected}"
        )

    engine.begin()
    for window in windows:
        engine(window.to(config.model.inference_device))
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
            expected_method=config.loop.method,
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
    load_pi3: Callable = _load_pi3
    load_images: Callable = _load_images
    build_segmentation_strategy: Callable = build_segmentation_strategy
    build_anchor_propagator: Callable = AnchorPropagator
    build_loop_strategy: Callable = build_loop_strategy
    run_windows: Callable = run_windows
    detect_loop_candidates: Callable = detect_loop_candidates
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


def build_default_window_engine(config: PipelineConfig, model):
    segmenter = build_segmentation_strategy(config.segmentation)
    anchor = AnchorPropagator(
        config.anchor_propagation.correspondence_iou_threshold
    )
    loop_strategy = build_loop_strategy(
        config.loop.method,
        optimizer_config=config.loop.optimizer,
        registration_confidence_keep_ratio=(
            config.loop.registration.confidence_keep_ratio
        ),
    )
    engine = loop_strategy.create_window_engine(
        delegate=model,
        inference_device=config.model.inference_device,
        dtype=_dtype(config.model.dtype),
        segmentation_strategy=segmenter,
        anchor_propagator=anchor,
        registration_confidence_keep_ratio=(
            config.loop.registration.confidence_keep_ratio
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
    )
    engine.pipeline_config = config
    engine.loop_strategy = loop_strategy
    return engine


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
        model = dependencies.load_pi3(config.model)
        segmenter = dependencies.build_segmentation_strategy(
            config.segmentation
        )
        anchor = dependencies.build_anchor_propagator(
            config.anchor_propagation.correspondence_iou_threshold
        )
        loop_strategy = dependencies.build_loop_strategy(
            config.loop.method,
            optimizer_config=config.loop.optimizer,
            registration_confidence_keep_ratio=(
                config.loop.registration.confidence_keep_ratio
            ),
        )
        engine = loop_strategy.create_window_engine(
            delegate=model,
            inference_device=config.model.inference_device,
            dtype=_dtype(config.model.dtype),
            segmentation_strategy=segmenter,
            anchor_propagator=anchor,
            registration_confidence_keep_ratio=(
                config.loop.registration.confidence_keep_ratio
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
        )
        images = dependencies.load_images(manifest)
        timings["initialization"] = (
            time.perf_counter() - started
        ) * 1000

        started = time.perf_counter()
        caches = tuple(
            dependencies.run_windows(
                engine,
                manifest,
                images,
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
                config.loop.detection,
                manifest,
                output_root / "loop_candidates.json",
            )
            if config.loop.enabled
            else ()
        )
        timings["loop_detection"] = (
            time.perf_counter() - started
        ) * 1000

        started = time.perf_counter()
        constraints = loop_strategy.build_constraints(
            caches,
            candidates,
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
            "loop_method": config.loop.method.value,
        }
        result = ReconstructionResult(
            payload=result.payload,
            summary=summary,
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
