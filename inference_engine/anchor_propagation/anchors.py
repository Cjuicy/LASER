"""High-confidence direct anchor estimation for HART."""

from collections import defaultdict

import numpy as np

from inference_engine.utils.depth import align_depth_irls

from .correspondence import candidate_relations, leaf_allowed_anchor_pairs
from .types import DirectAnchor


def high_confidence_mask(confidence, quantile):
    confidence = np.asarray(confidence)
    finite = np.isfinite(confidence)
    if not np.any(finite):
        return np.zeros(confidence.shape, dtype=bool)
    threshold = np.quantile(
        confidence[finite], quantile, method="nearest"
    )
    return finite & (confidence >= threshold)


def estimate_direct_anchors(
    previous_depth,
    current_depth,
    previous_confidence,
    current_confidence,
    previous_frames,
    current_frames,
    previous_tracks,
    current_tracks,
    *,
    confidence_quantile,
    corr_iou_thresh,
    anchor_min_pixels,
    irls=align_depth_irls,
):
    anchors = []
    diagnostics = defaultdict(int)
    for frame_index, (prev_frame, cur_frame) in enumerate(
        zip(previous_frames, current_frames, strict=True)
    ):
        allowed_anchor, _ = leaf_allowed_anchor_pairs(
            prev_frame, cur_frame, threshold=corr_iou_thresh
        )
        relations = candidate_relations(
            prev_frame.anchor_labels,
            cur_frame.anchor_labels,
            threshold=corr_iou_thresh,
            allowed_pairs=allowed_anchor,
        )
        prev_high = high_confidence_mask(
            previous_confidence[frame_index], confidence_quantile
        )
        cur_high = high_confidence_mask(
            current_confidence[frame_index], confidence_quantile
        )
        valid_depth = (
            np.isfinite(previous_depth[frame_index])
            & np.isfinite(current_depth[frame_index])
            & (previous_depth[frame_index] > 0)
            & (current_depth[frame_index] > 0)
        )
        for relation in relations:
            diagnostics["direct_anchor_candidate_count"] += 1
            core = (
                (prev_frame.anchor_labels == relation.src_id)
                & (cur_frame.anchor_labels == relation.tgt_id)
                & prev_high
                & cur_high
                & valid_depth
            )
            pixel_count = int(np.count_nonzero(core))
            if pixel_count < anchor_min_pixels:
                diagnostics["rejected_small_core_count"] += 1
                continue
            scale = float(
                irls(
                    current_depth[frame_index],
                    previous_depth[frame_index],
                    core,
                )
            )
            if not np.isfinite(scale) or scale <= 0:
                diagnostics["rejected_invalid_scale_count"] += 1
                continue
            anchors.append(
                DirectAnchor(
                    current_frame=frame_index,
                    current_anchor_id=relation.tgt_id,
                    current_segment_id=int(
                        current_tracks.segment_ids[frame_index][relation.tgt_id]
                    ),
                    previous_segment_id=int(
                        previous_tracks.segment_ids[frame_index][relation.src_id]
                    ),
                    scale=scale,
                    pixel_count=pixel_count,
                )
            )
            diagnostics["direct_anchor_count"] += 1
    return tuple(anchors), dict(diagnostics)
