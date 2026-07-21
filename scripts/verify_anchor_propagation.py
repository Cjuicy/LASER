#!/usr/bin/env python3
"""Run a deterministic CPU smoke test for HART with all segmentation modes."""

from pathlib import Path
import sys
import warnings

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inference_engine.anchor_propagation import (  # noqa: E402
    HartAnchorPropagator,
    RegistrationState,
    build_segmentation_window,
)
from inference_engine.utils.lsa import SEGMENT_MODES  # noqa: E402


def make_fixture(frames=3, height=48, width=64):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    depth = 1.5 + 0.002 * xx + 0.003 * yy
    depth[:, width // 2 :] += 0.35
    x = (xx - width / 2) * depth / 100.0
    y = (yy - height / 2) * depth / 100.0
    points = np.stack((x, y, depth), axis=-1)
    point_maps = np.stack([points.copy() for _ in range(frames)], axis=0)
    confidence = np.linspace(
        0.1, 1.0, height * width, dtype=np.float32
    ).reshape(height, width)
    confidence = np.stack([confidence.copy() for _ in range(frames)], axis=0)
    rgb = np.zeros((frames, 3, height, width), dtype=np.float32)
    rgb[:, 0] = xx / max(width - 1, 1)
    rgb[:, 1] = yy / max(height - 1, 1)
    rgb[:, 2, :, width // 2 :] = 1.0
    return point_maps, confidence, rgb


def verify_mode(mode, point_maps, confidence, rgb):
    overlap = 2
    propagator = HartAnchorPropagator(
        anchor_min_pixels=16,
        confidence_quantile=0.5,
    )
    previous_segments = build_segmentation_window(
        point_maps,
        conf_map=confidence,
        top_conf_percentile=0.5,
        segment_mode=mode,
        rgb_images=rgb,
        n_jobs=1,
    )
    first = propagator.refine(
        previous_registration_state=None,
        previous_anchor_state=None,
        current_base_points=torch.from_numpy(point_maps),
        current_confidence=confidence,
        current_segments=previous_segments,
        overlap=overlap,
    )

    current_points = point_maps * np.float32(0.8)
    current_segments = build_segmentation_window(
        current_points,
        conf_map=confidence,
        top_conf_percentile=0.5,
        segment_mode=mode,
        rgb_images=rgb,
        n_jobs=1,
    )
    registration_state = RegistrationState(
        final_base_points_tail=torch.from_numpy(point_maps[-overlap:]),
        final_base_poses_tail=torch.eye(4).repeat(overlap, 1, 1),
        pose_support_mask_tail=first.pose_support_mask[-overlap:].copy(),
    )
    second = propagator.refine(
        previous_registration_state=registration_state,
        previous_anchor_state=first.next_state,
        current_base_points=torch.from_numpy(current_points),
        current_confidence=confidence,
        current_segments=current_segments,
        overlap=overlap,
    )
    residual = second.local_residual_mask
    if residual.shape != (*current_points.shape[:3], 1):
        raise AssertionError(
            f"unexpected HART residual shape: {residual.shape}"
        )
    if np.any(~np.isfinite(residual)) or np.any(residual <= 0):
        raise AssertionError("HART residual contains invalid scale values")
    direct_count = int(second.diagnostics.get("direct_anchor_count", 0))
    if direct_count < 1:
        raise AssertionError(f"mode={mode} produced no direct anchor")
    if not np.isclose(second.window_scale, 1.25, rtol=1e-4):
        raise AssertionError(
            f"mode={mode} expected g=1.25, got {second.window_scale:.6f}"
        )
    if not np.allclose(residual, 1.0, rtol=1e-4):
        raise AssertionError(
            f"mode={mode} did not isolate uniform scale into g"
        )
    support_pixels = int(np.count_nonzero(second.pose_support_mask))
    if support_pixels < 1:
        raise AssertionError(f"mode={mode} produced no pose support")
    support_ratio = support_pixels / second.pose_support_mask.size
    print(
        f"[PASS] mode={mode} direct_anchors={direct_count} "
        f"g={second.window_scale:.4f} "
        f"residual_min={residual.min():.4f} "
        f"residual_median={np.median(residual):.4f} "
        f"residual_max={residual.max():.4f} "
        f"pose_support_ratio={support_ratio:.4f}"
    )


def main():
    point_maps, confidence, rgb = make_fixture()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Got image with third dimension of 4",
            category=RuntimeWarning,
        )
        for mode in SEGMENT_MODES:
            verify_mode(mode, point_maps, confidence, rgb)


if __name__ == "__main__":
    main()
