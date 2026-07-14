#!/usr/bin/env python3
"""Run a deterministic CPU smoke test for every segmentation mode."""

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference_engine.utils.lsa import (  # noqa: E402
    SEGMENT_MODES,
    get_felzenszwalb_params,
    make_sp_graph,
)


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


def verify_partition(graph, shape):
    if not graph:
        raise AssertionError("segmentation returned no frames")
    for frame_idx, vertices in enumerate(graph):
        if not vertices:
            raise AssertionError(f"frame {frame_idx} returned no segments")
        coverage = np.zeros(shape, dtype=np.int16)
        for vertex in vertices:
            mask = np.asarray(vertex.data)
            if mask.shape != shape or mask.dtype != np.bool_:
                raise AssertionError(
                    f"frame {frame_idx} contains an invalid segment mask"
                )
            coverage += mask
        if not np.all(coverage == 1):
            raise AssertionError(
                f"frame {frame_idx} masks do not form an exact image partition"
            )


def main():
    point_maps, confidence = make_fixture()
    for mode in SEGMENT_MODES:
        graph = make_sp_graph(
            point_maps,
            conf_map=confidence,
            top_conf_percentile=0.5,
            segment_mode=mode,
            normal_method="cross",
            geometry_seg_profile="baseline_params",
        )
        verify_partition(graph, point_maps.shape[1:3])
        params = get_felzenszwalb_params(mode, "baseline_params")
        print(
            f"[PASS] mode={mode} frames={len(graph)} "
            f"scale={params['seg_scale']} sigma={params['seg_sigma']} "
            f"min_size={params['seg_min_size']}"
        )


if __name__ == "__main__":
    main()
