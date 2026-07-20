"""HART-AP orchestration for one online streaming window."""

from collections import defaultdict
import time

import numpy as np

from .anchors import estimate_direct_anchors
from .consensus import aggregate_segment_scales, build_scale_mask
from .correspondence import build_hierarchical_tracks
from .types import AnchorPropagationState, PropagationResult


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class HartAnchorPropagator:
    def __init__(
        self,
        *,
        corr_iou_thresh=0.3,
        anchor_min_pixels=64,
        scale_consistency_thresh=0.05,
        confidence_quantile=0.5,
    ):
        if not 0 <= corr_iou_thresh <= 1:
            raise ValueError("corr_iou_thresh must be in [0, 1]")
        if int(anchor_min_pixels) <= 0:
            raise ValueError("anchor_min_pixels must be positive")
        if scale_consistency_thresh < 0:
            raise ValueError("scale_consistency_thresh must be non-negative")
        if not 0 <= confidence_quantile <= 1:
            raise ValueError("confidence_quantile must be in [0, 1]")
        self.corr_iou_thresh = float(corr_iou_thresh)
        self.anchor_min_pixels = int(anchor_min_pixels)
        self.scale_consistency_thresh = float(scale_consistency_thresh)
        self.confidence_quantile = float(confidence_quantile)

    def refine(
        self,
        *,
        previous_registration_state,
        previous_anchor_state,
        current_base_points,
        current_confidence,
        current_segments,
        overlap,
    ):
        start = time.perf_counter()
        points = _numpy(current_base_points)
        confidence = _numpy(current_confidence)
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("current_base_points must have shape (N, H, W, 3)")
        if confidence.shape != points.shape[:3]:
            raise ValueError("current_confidence must have shape (N, H, W)")
        if current_segments.shape != points.shape[:3]:
            raise ValueError("current_segments must align with current_base_points")
        if not 0 < overlap <= points.shape[0]:
            raise ValueError("overlap must be in [1, number of frames]")
        if (previous_registration_state is None) != (
            previous_anchor_state is None
        ):
            raise ValueError(
                "previous registration and anchor states must be provided together"
            )

        correspondence_start = time.perf_counter()
        leaf_tracks, anchor_tracks = build_hierarchical_tracks(
            current_segments.frames, threshold=self.corr_iou_thresh
        )
        correspondence_ms = (time.perf_counter() - correspondence_start) * 1000.0
        diagnostics = defaultdict(int)
        diagnostics.update(
            leaf_track_count=leaf_tracks.segment_count,
            anchor_track_count=anchor_tracks.segment_count,
            primary_edge_count=sum(len(step) for step in anchor_tracks.primary_edges),
            secondary_edge_count=sum(len(step) for step in anchor_tracks.secondary_edges),
        )

        if previous_registration_state is None or previous_anchor_state is None:
            mask = np.ones(points.shape[:3], dtype=np.float32)
            anchor_ms = 0.0
            consensus_ms = 0.0
            consensus_diagnostics = {
                "direct_pixel_ratio": 0.0,
                "filled_pixel_ratio": 0.0,
                "conflict_pixel_ratio": 0.0,
                "no_anchor_pixel_ratio": 1.0,
                "scale_mask_min": 1.0,
                "scale_mask_median": 1.0,
                "scale_mask_max": 1.0,
            }
        else:
            previous_frames = previous_anchor_state.segments_tail
            if len(previous_frames) != overlap:
                raise ValueError("previous anchor state must contain exactly overlap frames")
            previous_base = _numpy(previous_registration_state.base_points_tail)
            if previous_base.shape != (overlap, *points.shape[1:]):
                raise ValueError(
                    "previous base points must contain aligned overlap frames"
                )
            previous_scale = np.asarray(previous_anchor_state.local_scale_tail)
            if previous_scale.shape != (overlap, *points.shape[1:3]):
                raise ValueError(
                    "previous local scale must contain aligned overlap frames"
                )
            if np.any(~np.isfinite(previous_scale)) or np.any(previous_scale <= 0):
                raise ValueError("previous local scale must be finite and positive")
            previous_depth = previous_base[..., -1] * previous_scale
            current_depth = points[:overlap, ..., -1]
            previous_confidence = np.asarray(previous_anchor_state.confidence_tail)
            if previous_confidence.shape != (overlap, *points.shape[1:3]):
                raise ValueError(
                    "previous confidence must contain aligned overlap frames"
                )
            previous_leaf_tracks, previous_anchor_tracks = build_hierarchical_tracks(
                previous_frames, threshold=self.corr_iou_thresh
            )
            del previous_leaf_tracks

            anchor_start = time.perf_counter()
            direct_anchors, anchor_diagnostics = estimate_direct_anchors(
                previous_depth,
                current_depth,
                previous_confidence,
                confidence[:overlap],
                previous_frames,
                current_segments.frames[:overlap],
                previous_anchor_tracks,
                anchor_tracks,
                confidence_quantile=self.confidence_quantile,
                corr_iou_thresh=self.corr_iou_thresh,
                anchor_min_pixels=self.anchor_min_pixels,
            )
            anchor_ms = (time.perf_counter() - anchor_start) * 1000.0
            diagnostics.update(anchor_diagnostics)

            consensus_start = time.perf_counter()
            segment_scales, conflict_segments = aggregate_segment_scales(
                direct_anchors, self.scale_consistency_thresh
            )
            mask, _, consensus_diagnostics = build_scale_mask(
                current_segments.frames,
                anchor_tracks,
                segment_scales,
                conflict_segments,
                self.scale_consistency_thresh,
            )
            consensus_ms = (time.perf_counter() - consensus_start) * 1000.0

        diagnostics.update(consensus_diagnostics)
        diagnostics.update(
            correspondence_runtime_ms=correspondence_ms,
            anchor_runtime_ms=anchor_ms,
            consensus_runtime_ms=consensus_ms,
            propagation_runtime_ms=(time.perf_counter() - start) * 1000.0,
        )
        next_state = AnchorPropagationState(
            local_scale_tail=mask[-overlap:].copy(),
            confidence_tail=confidence[-overlap:].copy(),
            segments_tail=tuple(current_segments.frames[-overlap:]),
        )
        return PropagationResult(
            local_scale_mask=mask[..., None],
            next_state=next_state,
            diagnostics=dict(diagnostics),
        )
