"""Extract one camera scale and local residuals from regional HART scales."""

from collections import defaultdict

import numpy as np

from .consensus import (
    STATUS_DIRECT,
    STATUS_LEAF_FILL,
    STATUS_PARENT_FILL,
    complete_link_groups,
)
from .types import PoseConsensus


def _identity_consensus(*, group_count, valid_pixels, support_pixels=0):
    support_ratio = (
        float(support_pixels / valid_pixels) if valid_pixels else 0.0
    )
    return PoseConsensus(
        window_scale=1.0,
        segment_ids=frozenset(),
        group_count=int(group_count),
        support_pixels=int(support_pixels),
        valid_pixels=int(valid_pixels),
        support_ratio=support_ratio,
        accepted=False,
    )


def select_pose_consensus(
    direct_anchors,
    segment_scales,
    *,
    valid_pixel_count,
    threshold,
):
    """Select a strict-majority scale group without weighting its median."""
    valid_pixels = int(valid_pixel_count)
    if valid_pixels < 0:
        raise ValueError("valid_pixel_count must be non-negative")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    ordered = sorted(
        (int(segment_id), float(scale))
        for segment_id, scale in segment_scales.items()
    )
    if not ordered:
        return _identity_consensus(group_count=0, valid_pixels=valid_pixels)

    segment_ids = np.asarray([item[0] for item in ordered], dtype=np.int64)
    scales = np.asarray([item[1] for item in ordered], dtype=np.float64)
    groups = complete_link_groups(scales, threshold)

    support_by_segment = defaultdict(int)
    for anchor in direct_anchors:
        pixel_count = int(anchor.pixel_count)
        if pixel_count < 0:
            raise ValueError("direct anchor pixel_count must be non-negative")
        support_by_segment[int(anchor.current_segment_id)] += pixel_count

    candidates = []
    for order, group in enumerate(groups):
        group_segments = frozenset(int(value) for value in segment_ids[group])
        support_pixels = sum(
            support_by_segment[segment_id] for segment_id in group_segments
        )
        candidates.append(
            (
                int(support_pixels),
                len(group_segments),
                -order,
                group_segments,
                group,
            )
        )

    support_pixels, _, _, selected_segments, selected_group = max(candidates)
    if support_pixels > valid_pixels:
        raise ValueError(
            "pose consensus support pixels cannot exceed valid pixels"
        )
    support_ratio = (
        float(support_pixels / valid_pixels) if valid_pixels else 0.0
    )
    if valid_pixels == 0 or support_pixels * 2 <= valid_pixels:
        return _identity_consensus(
            group_count=len(groups),
            valid_pixels=valid_pixels,
            support_pixels=support_pixels,
        )

    window_scale = float(np.median(scales[selected_group]))
    if not np.isfinite(window_scale) or window_scale <= 0:
        raise ValueError("pose consensus produced an invalid window scale")
    return PoseConsensus(
        window_scale=window_scale,
        segment_ids=selected_segments,
        group_count=len(groups),
        support_pixels=support_pixels,
        valid_pixels=valid_pixels,
        support_ratio=support_ratio,
        accepted=True,
    )


def decompose_regional_scales(
    regional_scale_mask,
    status_maps,
    segment_maps,
    *,
    window_scale,
    pose_segment_ids,
):
    """Split resolved regional scales into a global scale and local residual."""
    regional = np.asarray(regional_scale_mask, dtype=np.float64)
    statuses = np.asarray(status_maps)
    segments = np.asarray(segment_maps)
    if regional.shape != statuses.shape or regional.shape != segments.shape:
        raise ValueError("regional, status, and segment maps must share shape")
    if regional.ndim != 3:
        raise ValueError("regional, status, and segment maps must have shape (N,H,W)")
    if np.any(~np.isfinite(regional)) or np.any(regional <= 0):
        raise ValueError("regional scales must be finite and positive")

    scale = float(window_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("window_scale must be finite and positive")

    resolved = np.isin(
        statuses,
        (STATUS_DIRECT, STATUS_LEAF_FILL, STATUS_PARENT_FILL),
    )
    residual = np.ones(regional.shape, dtype=np.float64)
    residual[resolved] = regional[resolved] / scale
    if np.any(~np.isfinite(residual)) or np.any(residual <= 0):
        raise ValueError("local residuals must be finite and positive")

    selected = tuple(int(value) for value in pose_segment_ids)
    pose_support = (statuses == STATUS_DIRECT) & np.isin(segments, selected)
    return residual.astype(np.float32), pose_support.astype(np.bool_)
