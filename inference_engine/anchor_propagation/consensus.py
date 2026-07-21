"""Robust one-dimensional scale aggregation and hierarchical filling."""

from collections import defaultdict

import numpy as np


STATUS_NO_ANCHOR = 0
STATUS_DIRECT = 1
STATUS_LEAF_FILL = 2
STATUS_PARENT_FILL = 3
STATUS_CONFLICT = 4


def complete_link_groups(scales, threshold):
    scales = np.asarray(scales, dtype=np.float64)
    if scales.size == 0:
        return ()
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales must be finite and positive")
    order = np.argsort(np.log(scales), kind="stable")
    logs = np.log(scales[order])
    groups = []
    start = 0
    for index in range(1, logs.size + 1):
        if index == logs.size or logs[index] - logs[start] > threshold:
            groups.append(order[start:index])
            start = index
    return tuple(groups)


def aggregate_segment_scales(direct_anchors, threshold):
    """Median within one provenance, then reject conflicting provenances."""
    by_current_previous = defaultdict(list)
    for anchor in direct_anchors:
        by_current_previous[
            (anchor.current_segment_id, anchor.previous_segment_id)
        ].append(anchor.scale)
    provenance_medians = defaultdict(list)
    for (current_segment, _), scales in by_current_previous.items():
        provenance_medians[current_segment].append(float(np.median(scales)))

    segment_scales = {}
    conflict_segments = set()
    for segment_id, scales in provenance_medians.items():
        groups = complete_link_groups(scales, threshold)
        if len(groups) == 1:
            segment_scales[segment_id] = float(np.median(scales))
        else:
            conflict_segments.add(segment_id)
    return segment_scales, conflict_segments


def _unique_consensus(values, threshold):
    if not values:
        return None, False
    groups = complete_link_groups(values, threshold)
    if len(groups) != 1:
        return None, True
    return float(np.median(values)), False


def build_scale_mask(
    frames,
    anchor_tracks,
    segment_scales,
    conflict_segments,
    threshold,
):
    masks = []
    statuses = []
    diagnostics = defaultdict(int)
    for frame_index, frame in enumerate(frames):
        anchor_count = int(frame.anchor_labels.max()) + 1
        scales = np.ones(anchor_count, dtype=np.float64)
        state = np.full(anchor_count, STATUS_NO_ANCHOR, dtype=np.uint8)
        segment_ids = anchor_tracks.segment_ids[frame_index]
        for anchor_id, segment_id in enumerate(segment_ids.tolist()):
            if segment_id in conflict_segments:
                state[anchor_id] = STATUS_CONFLICT
                diagnostics["track_conflict_count"] += 1
            elif segment_id in segment_scales:
                scales[anchor_id] = segment_scales[segment_id]
                state[anchor_id] = STATUS_DIRECT

        leaf_representatives = {}
        leaf_conflicts = set()
        for leaf_id in range(int(frame.leaf_labels.max()) + 1):
            anchor_ids = np.flatnonzero(frame.anchor_to_leaf == leaf_id)
            direct_ids = anchor_ids[state[anchor_ids] == STATUS_DIRECT]
            direct_values = scales[direct_ids].tolist()
            consensus, conflict = _unique_consensus(direct_values, threshold)
            if conflict:
                leaf_conflicts.add(leaf_id)
                diagnostics["leaf_conflict_count"] += 1
                continue
            if consensus is None:
                continue
            leaf_representatives[leaf_id] = consensus
            fill_ids = anchor_ids[state[anchor_ids] == STATUS_NO_ANCHOR]
            scales[fill_ids] = consensus
            state[fill_ids] = STATUS_LEAF_FILL
            scales[direct_ids] = consensus
            diagnostics["leaf_consensus_count"] += 1

        parent_count = int(frame.parent_labels.max()) + 1
        for parent_id in range(parent_count):
            leaf_ids = np.flatnonzero(frame.leaf_to_parent == parent_id)
            if leaf_ids.size <= 1:
                continue
            values = [
                leaf_representatives[int(leaf_id)]
                for leaf_id in leaf_ids
                if int(leaf_id) in leaf_representatives
            ]
            consensus, conflict = _unique_consensus(values, threshold)
            if conflict or any(int(leaf_id) in leaf_conflicts for leaf_id in leaf_ids):
                diagnostics["parent_conflict_count"] += 1
                continue
            if consensus is None:
                continue
            parent_anchor_ids = np.concatenate(
                [
                    np.flatnonzero(frame.anchor_to_leaf == int(leaf_id))
                    for leaf_id in leaf_ids
                ]
            )
            eligible = parent_anchor_ids[
                state[parent_anchor_ids] != STATUS_CONFLICT
            ]
            was_missing = state[eligible] == STATUS_NO_ANCHOR
            state[eligible[was_missing]] = STATUS_PARENT_FILL
            scales[eligible] = consensus
            diagnostics["parent_consensus_count"] += 1

        mask = scales[frame.anchor_labels]
        if np.any(~np.isfinite(mask)) or np.any(mask <= 0):
            raise ValueError("HART produced a non-finite or non-positive scale")
        masks.append(mask.astype(np.float32, copy=False))
        statuses.append(state[frame.anchor_labels])

    mask = np.stack(masks)
    status_maps = np.stack(statuses)
    total = max(status_maps.size, 1)
    diagnostics.update(
        direct_pixel_ratio=float(np.count_nonzero(status_maps == STATUS_DIRECT) / total),
        filled_pixel_ratio=float(
            np.count_nonzero(
                (status_maps == STATUS_LEAF_FILL)
                | (status_maps == STATUS_PARENT_FILL)
            )
            / total
        ),
        conflict_pixel_ratio=float(
            np.count_nonzero(status_maps == STATUS_CONFLICT) / total
        ),
        no_anchor_pixel_ratio=float(
            np.count_nonzero(status_maps == STATUS_NO_ANCHOR) / total
        ),
        regional_scale_min=float(mask.min()),
        regional_scale_median=float(np.median(mask)),
        regional_scale_max=float(mask.max()),
    )
    return mask, status_maps, dict(diagnostics)
