"""Sparse label correspondence and deterministic temporal track construction."""

from collections import defaultdict

import numpy as np

from .types import PairRelation, TemporalEdge, TrackWindow


def sparse_pair_relations(src_labels, tgt_labels):
    src = np.asarray(src_labels, dtype=np.int64)
    tgt = np.asarray(tgt_labels, dtype=np.int64)
    if src.ndim != 2 or src.shape != tgt.shape:
        raise ValueError("src_labels and tgt_labels must share shape (H, W)")
    if src.size == 0:
        return ()
    if src.min() < 0 or tgt.min() < 0:
        raise ValueError("labels must be non-negative")
    num_src = int(src.max()) + 1
    num_tgt = int(tgt.max()) + 1
    src_area = np.bincount(src.reshape(-1), minlength=num_src)
    tgt_area = np.bincount(tgt.reshape(-1), minlength=num_tgt)
    codes, intersections = np.unique(
        (src * num_tgt + tgt).reshape(-1), return_counts=True
    )
    relations = []
    for code, intersection in zip(codes, intersections, strict=True):
        src_id = int(code // num_tgt)
        tgt_id = int(code % num_tgt)
        intersection = int(intersection)
        union = int(src_area[src_id] + tgt_area[tgt_id] - intersection)
        relations.append(
            PairRelation(
                src_id=src_id,
                tgt_id=tgt_id,
                intersection=intersection,
                iou=float(intersection / max(union, 1)),
                src_coverage=float(intersection / max(int(src_area[src_id]), 1)),
                tgt_coverage=float(intersection / max(int(tgt_area[tgt_id]), 1)),
            )
        )
    return tuple(relations)


def candidate_relations(src_labels, tgt_labels, threshold=0.3, allowed_pairs=None):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    return tuple(
        relation
        for relation in sparse_pair_relations(src_labels, tgt_labels)
        if relation.correspondence_score >= threshold
        and (
            allowed_pairs is None
            or (relation.src_id, relation.tgt_id) in allowed_pairs
        )
    )


def _primary_key(relation):
    return (
        relation.tgt_coverage,
        relation.iou,
        relation.intersection,
        -relation.src_id,
    )


def classify_edges(relations):
    """Choose one primary source for every target and retain all others."""
    by_target = defaultdict(list)
    for relation in relations:
        by_target[relation.tgt_id].append(relation)
    primary_by_target = {
        tgt_id: max(candidates, key=_primary_key)
        for tgt_id, candidates in by_target.items()
    }
    primary = tuple(
        TemporalEdge(
            relation.src_id, relation.tgt_id, relation, is_primary=True
        )
        for relation in sorted(primary_by_target.values(), key=lambda item: item.tgt_id)
    )
    primary_pairs = {(edge.src_id, edge.tgt_id) for edge in primary}
    secondary = tuple(
        TemporalEdge(
            relation.src_id, relation.tgt_id, relation, is_primary=False
        )
        for relation in sorted(
            relations, key=lambda item: (item.tgt_id, item.src_id)
        )
        if (relation.src_id, relation.tgt_id) not in primary_pairs
    )
    return primary, secondary


def build_track_window(label_maps, threshold=0.3, allowed_pairs_by_step=None):
    label_maps = tuple(np.asarray(labels, dtype=np.intp) for labels in label_maps)
    if not label_maps:
        return TrackWindow((), (), (), {})
    shape = label_maps[0].shape
    if any(labels.ndim != 2 or labels.shape != shape for labels in label_maps):
        raise ValueError("all label maps must share shape (H, W)")

    first_count = int(label_maps[0].max()) + 1
    segment_ids = [np.arange(first_count, dtype=np.int64)]
    lineage_parents = {segment_id: None for segment_id in range(first_count)}
    next_segment = first_count
    primary_steps = []
    secondary_steps = []

    for step, (src_labels, tgt_labels) in enumerate(
        zip(label_maps[:-1], label_maps[1:], strict=True)
    ):
        allowed = (
            None
            if allowed_pairs_by_step is None
            else allowed_pairs_by_step[step]
        )
        relations = candidate_relations(
            src_labels, tgt_labels, threshold=threshold, allowed_pairs=allowed
        )
        primary, secondary = classify_edges(relations)
        primary_steps.append(primary)
        secondary_steps.append(secondary)
        tgt_count = int(tgt_labels.max()) + 1
        tgt_segments = np.empty(tgt_count, dtype=np.int64)
        primary_by_target = {edge.tgt_id: edge for edge in primary}
        children_by_source = defaultdict(list)
        for edge in primary:
            children_by_source[edge.src_id].append(edge)
        continuation = {}
        for src_id, edges in children_by_source.items():
            continuation[src_id] = max(
                edges,
                key=lambda edge: (
                    edge.relation.tgt_coverage,
                    edge.relation.iou,
                    edge.relation.intersection,
                    -edge.tgt_id,
                ),
            ).tgt_id

        for tgt_id in range(tgt_count):
            edge = primary_by_target.get(tgt_id)
            if edge is None:
                tgt_segments[tgt_id] = next_segment
                lineage_parents[next_segment] = None
                next_segment += 1
                continue
            src_segment = int(segment_ids[-1][edge.src_id])
            if continuation[edge.src_id] == tgt_id:
                tgt_segments[tgt_id] = src_segment
            else:
                tgt_segments[tgt_id] = next_segment
                lineage_parents[next_segment] = src_segment
                next_segment += 1
        segment_ids.append(tgt_segments)

    return TrackWindow(
        segment_ids=tuple(segment_ids),
        primary_edges=tuple(primary_steps),
        secondary_edges=tuple(secondary_steps),
        lineage_parents=lineage_parents,
    )


def leaf_allowed_anchor_pairs(src_frame, tgt_frame, threshold=0.3):
    leaf_relations = candidate_relations(
        src_frame.leaf_labels, tgt_frame.leaf_labels, threshold=threshold
    )
    allowed_leaf = {(relation.src_id, relation.tgt_id) for relation in leaf_relations}
    allowed_anchor = set()
    for relation in sparse_pair_relations(
        src_frame.anchor_labels, tgt_frame.anchor_labels
    ):
        src_leaf = int(src_frame.anchor_to_leaf[relation.src_id])
        tgt_leaf = int(tgt_frame.anchor_to_leaf[relation.tgt_id])
        if (src_leaf, tgt_leaf) in allowed_leaf:
            allowed_anchor.add((relation.src_id, relation.tgt_id))
    return allowed_anchor, leaf_relations


def build_hierarchical_tracks(frames, threshold=0.3):
    frames = tuple(frames)
    leaf_maps = tuple(frame.leaf_labels for frame in frames)
    leaf_tracks = build_track_window(leaf_maps, threshold=threshold)
    allowed_by_step = []
    for src_frame, tgt_frame in zip(frames[:-1], frames[1:], strict=True):
        allowed, _ = leaf_allowed_anchor_pairs(
            src_frame, tgt_frame, threshold=threshold
        )
        allowed_by_step.append(allowed)
    anchor_tracks = build_track_window(
        tuple(frame.anchor_labels for frame in frames),
        threshold=threshold,
        allowed_pairs_by_step=tuple(allowed_by_step),
    )
    return leaf_tracks, anchor_tracks
