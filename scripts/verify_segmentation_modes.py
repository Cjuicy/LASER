#!/usr/bin/env python3
"""Run a deterministic CPU smoke test for every public segmentation method."""

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference_engine.segmentation import (  # noqa: E402
    build_segmentation_strategy,
    build_temporal_graphs,
)
from pipeline.config import load_pipeline_config  # noqa: E402


SEGMENTATION_METHODS = ("depth", "geometry", "atomic")


def make_fixture(frames=2, height=48, width=64):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    depth = 1.0 + 0.002 * xx + 0.003 * yy
    x = (xx - width / 2) * depth / 100.0
    y = (yy - height / 2) * depth / 100.0
    points = np.stack((x, y, depth), axis=-1)
    point_maps = np.stack(
        [points, points + np.array([0.001, 0.0, 0.002], dtype=np.float32)],
        axis=0,
    )
    confidence = np.linspace(
        0.1,
        1.0,
        frames * height * width,
        dtype=np.float32,
    ).reshape(frames, height, width)
    return point_maps, confidence


def verify_partition(results, shape):
    if not results:
        raise AssertionError("segmentation returned no frames")
    for frame_idx, result in enumerate(results):
        labels = np.asarray(result.labels)
        if labels.shape != shape:
            raise AssertionError(
                f"frame {frame_idx} has invalid label shape"
            )
        unique = np.unique(labels)
        np.testing.assert_array_equal(unique, np.arange(unique.size))


def main():
    point_maps, confidence = make_fixture()
    for method in SEGMENTATION_METHODS:
        config = load_pipeline_config(
            "configs/pipeline/test.yaml",
            (
                f"segmentation.method={method}",
                "segmentation.felzenszwalb.min_size=20",
            ),
        ).config.segmentation
        strategy = build_segmentation_strategy(config)
        results = strategy.segment(
            point_maps,
            confidence,
            images=None,
        )
        verify_partition(results, point_maps.shape[1:3])
        graph = build_temporal_graphs(
            results,
            config.temporal_iou_threshold,
        )
        if len(graph) != len(results):
            raise AssertionError("temporal graph frame count mismatch")
        print(
            f"[PASS] method={method} frames={len(results)} "
            f"scale={config.felzenszwalb.scale} "
            f"sigma={config.felzenszwalb.sigma} "
            f"min_size={config.felzenszwalb.min_size}"
        )


if __name__ == "__main__":
    main()
