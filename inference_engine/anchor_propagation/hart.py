"""HART-AP orchestration for one online streaming window."""

from collections import defaultdict
import time

import numpy as np

from .anchors import estimate_direct_anchors
from .consensus import (
    STATUS_NO_ANCHOR,
    aggregate_segment_scales,
    build_scale_mask,
)
from .correspondence import build_hierarchical_tracks
from .pose_consensus import (
    decompose_regional_scales,
    select_pose_consensus,
)
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
            regional_scale = np.ones(points.shape[:3], dtype=np.float32)
            status_maps = np.full(
                points.shape[:3], STATUS_NO_ANCHOR, dtype=np.uint8
            )
            direct_anchors = ()
            segment_scales = {}
            anchor_ms = 0.0
            consensus_ms = 0.0
            consensus_diagnostics = {
                "direct_pixel_ratio": 0.0,
                "filled_pixel_ratio": 0.0,
                "conflict_pixel_ratio": 0.0,
                "no_anchor_pixel_ratio": 1.0,
                "regional_scale_min": 1.0,
                "regional_scale_median": 1.0,
                "regional_scale_max": 1.0,
            }
            anchor_diagnostics = {"pose_consensus_valid_pixels": 0}
        else:
            previous_frames = previous_anchor_state.segments_tail
            if len(previous_frames) != overlap:
                raise ValueError("previous anchor state must contain exactly overlap frames")
            previous_base = _numpy(
                previous_registration_state.final_base_points_tail
            )
            if previous_base.shape != (overlap, *points.shape[1:]):
                raise ValueError(
                    "previous base points must contain aligned overlap frames"
                )
            previous_residual = np.asarray(
                previous_anchor_state.local_residual_tail
            )
            if previous_residual.shape != (overlap, *points.shape[1:3]):
                raise ValueError(
                    "previous local residual must contain aligned overlap frames"
                )
            if np.any(~np.isfinite(previous_residual)) or np.any(
                previous_residual <= 0
            ):
                raise ValueError(
                    "previous local residual must be finite and positive"
                )
            previous_depth = previous_base[..., -1] * previous_residual
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
            regional_scale, status_maps, consensus_diagnostics = build_scale_mask(
                current_segments.frames,
                anchor_tracks,
                segment_scales,
                conflict_segments,
                self.scale_consistency_thresh,
            )
            consensus_ms = (time.perf_counter() - consensus_start) * 1000.0

        pose_consensus = select_pose_consensus(
            direct_anchors,
            segment_scales,
            valid_pixel_count=int(
                anchor_diagnostics.get("pose_consensus_valid_pixels", 0)
            ),
            threshold=self.scale_consistency_thresh,
        )
        segment_maps = np.stack(
            [
                segment_ids[frame.anchor_labels]
                for frame, segment_ids in zip(
                    current_segments.frames,
                    anchor_tracks.segment_ids,
                    strict=True,
                )
            ]
        )
        local_residual, pose_support = decompose_regional_scales(
            regional_scale,
            status_maps,
            segment_maps,
            window_scale=pose_consensus.window_scale,
            pose_segment_ids=pose_consensus.segment_ids,
        )

        diagnostics.update(consensus_diagnostics)
        diagnostics.update(
            window_scale=pose_consensus.window_scale,
            pose_consensus_group_count=pose_consensus.group_count,
            pose_consensus_selected_segment_count=len(
                pose_consensus.segment_ids
            ),
            pose_consensus_support_pixels=pose_consensus.support_pixels,
            pose_consensus_valid_pixels=pose_consensus.valid_pixels,
            pose_consensus_support_ratio=pose_consensus.support_ratio,
            pose_consensus_accepted=pose_consensus.accepted,
            pose_support_pixel_ratio=float(
                np.count_nonzero(pose_support) / max(pose_support.size, 1)
            ),
            local_residual_min=float(local_residual.min()),
            local_residual_median=float(np.median(local_residual)),
            local_residual_max=float(local_residual.max()),
            correspondence_runtime_ms=correspondence_ms,
            anchor_runtime_ms=anchor_ms,
            consensus_runtime_ms=consensus_ms,
            propagation_runtime_ms=(time.perf_counter() - start) * 1000.0,
        )
        next_state = AnchorPropagationState(
            local_residual_tail=local_residual[-overlap:].copy(),
            confidence_tail=confidence[-overlap:].copy(),
            segments_tail=tuple(current_segments.frames[-overlap:]),
        )
        return PropagationResult(
            window_scale=pose_consensus.window_scale,
            local_residual_mask=local_residual[..., None],
            pose_support_mask=pose_support,
            next_state=next_state,
            diagnostics=dict(diagnostics),
        )
