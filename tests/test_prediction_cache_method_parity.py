from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_engine.inference_utils import (
    unproject_depth_to_local_points,
)
from inference_engine.models.lazy import LazyModelHandle
from inference_engine.prediction_cache.fingerprint import (
    build_prediction_fingerprint,
)
from inference_engine.prediction_cache.provider import (
    OrdinaryPredictionProvider,
)
from inference_engine.prediction_cache.store import OrdinaryPredictionStore
from inference_engine.prediction_cache.types import build_window_specs
from loop_closure.methods.base import WindowCache
from pipeline.config import (
    LoopMethod,
    PredictionCacheMode,
    SegmentationMethod,
    load_pipeline_config,
)
from pipeline.manifest import ImageManifest
from pipeline.runner import build_default_window_engine, run_windows


class DeterministicAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.intrinsic = torch.tensor(
            [
                [4.0, 0.0, 2.0],
                [0.0, 4.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, frames, _, height, width = images.shape
        depth = (
            torch.arange(frames, dtype=torch.float32)
            .view(frames, 1, 1)
            .expand(frames, height, width)
            + 2.0
        )
        points = unproject_depth_to_local_points(
            depth,
            self.intrinsic,
        ).unsqueeze(0).expand(batch, -1, -1, -1, -1)
        return {
            "local_points": points,
            "camera_poses": torch.eye(4).repeat(
                batch,
                frames,
                1,
                1,
            ),
            "conf": torch.ones((batch, frames, height, width)),
            "images": images,
        }


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _assert_cache_equal(left: WindowCache, right: WindowCache):
    assert left.window_index == right.window_index
    assert left.frame_start == right.frame_start
    assert left.frame_end == right.frame_end
    torch.testing.assert_close(
        left.local_points,
        right.local_points,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        left.camera_poses,
        right.camera_poses,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        left.confidence,
        right.confidence,
        rtol=0.0,
        atol=0.0,
    )
    _assert_nested_equal(left.segmentation_labels, right.segmentation_labels)
    _assert_nested_equal(left.anchor_scale_mask, right.anchor_scale_mask)
    _assert_nested_equal(left.loop_state, right.loop_state)


def _run_mode(
    *,
    tmp_path,
    base_config,
    manifest,
    images,
    method,
    mode,
    run_name,
):
    config = replace(
        base_config,
        output=replace(
            base_config.output,
            cache_dir=str(tmp_path / "methods" / run_name),
        ),
        prediction_cache=replace(
            base_config.prediction_cache,
            root=str(tmp_path / "predictions"),
            mode=mode,
        ),
        segmentation=replace(
            base_config.segmentation,
            method=SegmentationMethod.DEPTH,
        ),
        anchor_propagation=replace(
            base_config.anchor_propagation,
            enabled=True,
        ),
        loop=replace(
            base_config.loop,
            enabled=False,
            method=method,
        ),
    )
    specs = build_window_specs(
        len(manifest),
        config.window.size,
        config.window.overlap,
    )
    fingerprint = build_prediction_fingerprint(
        model=config.model,
        manifest=manifest,
        image_shape=tuple(images.shape),
        sample_stride=config.input.sample_stride,
        window_size=config.window.size,
        overlap=config.window.overlap,
        specs=specs,
    )
    constructions = []

    def factory():
        constructions.append("constructed")
        if mode is PredictionCacheMode.READONLY:
            raise AssertionError("warm parity run constructed Pi3")
        return DeterministicAdapter()

    handle = LazyModelHandle(
        factory,
        inference_device="cpu",
        dtype=torch.float32,
    )
    store = OrdinaryPredictionStore(
        root=config.prediction_cache.root,
        fingerprint=fingerprint,
        mode=mode,
        expected_specs=specs,
    )
    provider = OrdinaryPredictionProvider(store=store, model=handle)
    engine = build_default_window_engine(config, provider, fingerprint)
    with store.entry_lock():
        caches = run_windows(
            engine,
            manifest,
            images,
            specs,
            config,
        )
    constraints = engine.loop_strategy.build_constraints(caches, ())
    solution = engine.loop_strategy.optimize(caches, constraints)
    result = engine.loop_strategy.aggregate(caches, solution)
    return caches, result, handle


@pytest.mark.parametrize("method", tuple(LoopMethod))
def test_off_cold_and_warm_paths_preserve_method_math(
    tmp_path,
    method,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = []
    for index in range(12):
        path = image_dir / f"frame-{index:04d}.png"
        path.write_bytes(f"frame-{index}".encode("ascii"))
        paths.append(path)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"deterministic-checkpoint")
    loaded = load_pipeline_config("configs/pipeline/test.yaml")
    base_config = replace(
        loaded.config,
        input=replace(
            loaded.config.input,
            image_dir=str(image_dir),
            sample_stride=1,
        ),
        model=replace(
            loaded.config.model,
            checkpoint=str(checkpoint),
            inference_device="cpu",
            process_device="cpu",
            dtype="float32",
        ),
        window=replace(loaded.config.window, size=10, overlap=5),
    )
    manifest = ImageManifest(paths=tuple(paths))
    images = torch.arange(
        12 * 3 * 4 * 4,
        dtype=torch.float32,
    ).reshape(12, 3, 4, 4)

    off_caches, off_result, off_handle = _run_mode(
        tmp_path=tmp_path,
        base_config=base_config,
        manifest=manifest,
        images=images,
        method=method,
        mode=PredictionCacheMode.OFF,
        run_name=f"{method.value}-off",
    )
    cold_caches, cold_result, cold_handle = _run_mode(
        tmp_path=tmp_path,
        base_config=base_config,
        manifest=manifest,
        images=images,
        method=method,
        mode=PredictionCacheMode.AUTO,
        run_name=f"{method.value}-cold",
    )
    warm_caches, warm_result, warm_handle = _run_mode(
        tmp_path=tmp_path,
        base_config=base_config,
        manifest=manifest,
        images=images,
        method=method,
        mode=PredictionCacheMode.READONLY,
        run_name=f"{method.value}-warm",
    )

    assert [(cache.frame_start, cache.frame_end) for cache in off_caches] == [
        (0, 10),
        (5, 12),
    ]
    for off, cold, warm in zip(
        off_caches,
        cold_caches,
        warm_caches,
        strict=True,
    ):
        _assert_cache_equal(off, cold)
        _assert_cache_equal(off, warm)
    _assert_nested_equal(off_result.payload, cold_result.payload)
    _assert_nested_equal(off_result.payload, warm_result.payload)
    assert off_handle.stats.ordinary_forward_count == 2
    assert cold_handle.stats.ordinary_forward_count == 2
    assert warm_handle.stats.ordinary_forward_count == 0
    assert warm_handle.stats.model_constructed is False
