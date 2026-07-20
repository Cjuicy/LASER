"""Adapt every LASER segmentation mode to a strict HART hierarchy."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import numpy as np

from inference_engine.utils.depth import segment_depth_felzenszwalb_rag_stages
from inference_engine.utils.geometry_segmentation import (
    segment_geometry_felzenszwalb_rag_baseline_params_stages,
    segment_geometry_felzenszwalb_rag_stages,
)
from inference_engine.utils.layer_atomic_geometry import (
    segment_point_map_layer_atomic_stages,
)
from inference_engine.utils.lsa import (
    NORMAL_METHODS,
    SEGMENT_MODES,
    get_felzenszwalb_params,
)

from .types import SegmentationFrame, SegmentationWindow


def compact_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must have shape (H, W)")
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.intp, copy=False)


def compact_intersection_labels(initial_labels, leaf_labels):
    """Split initial cells at final-leaf boundaries and compact pair labels."""
    initial_labels = np.asarray(initial_labels)
    leaf_labels = np.asarray(leaf_labels)
    if initial_labels.shape != leaf_labels.shape or initial_labels.ndim != 2:
        raise ValueError("initial_labels and leaf_labels must share shape (H, W)")
    pairs = np.stack(
        (initial_labels.reshape(-1), leaf_labels.reshape(-1)), axis=-1
    )
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return inverse.reshape(initial_labels.shape).astype(np.intp, copy=False)


def label_parent_lookup(child_labels, parent_labels):
    child_labels = compact_labels(child_labels)
    parent_labels = compact_labels(parent_labels)
    if child_labels.shape != parent_labels.shape:
        raise ValueError("child_labels and parent_labels must have the same shape")
    lookup = np.empty(int(child_labels.max()) + 1, dtype=np.intp)
    for child_id in range(lookup.size):
        parents = np.unique(parent_labels[child_labels == child_id])
        if parents.size != 1:
            raise ValueError(
                f"child label {child_id} crosses {parents.size} parent labels"
            )
        lookup[child_id] = parents[0]
    return lookup


def validate_segmentation_frame(frame):
    shape = frame.leaf_labels.shape
    for name in ("leaf_labels", "parent_labels", "anchor_labels"):
        labels = np.asarray(getattr(frame, name))
        if labels.ndim != 2 or labels.shape != shape:
            raise ValueError(f"{name} must have the common shape (H, W)")
        if labels.size and (labels.min() != 0 or labels.max() + 1 != np.unique(labels).size):
            raise ValueError(f"{name} must contain compact non-negative labels")
    expected_leaf_parent = label_parent_lookup(
        frame.leaf_labels, frame.parent_labels
    )
    expected_anchor_leaf = label_parent_lookup(
        frame.anchor_labels, frame.leaf_labels
    )
    if not np.array_equal(frame.leaf_to_parent, expected_leaf_parent):
        raise ValueError("leaf_to_parent does not match label maps")
    if not np.array_equal(frame.anchor_to_leaf, expected_anchor_leaf):
        raise ValueError("anchor_to_leaf does not match label maps")
    return frame


def _frame_from_layers(initial, leaf, parent=None, diagnostics=None):
    leaf = compact_labels(leaf)
    parent = leaf.copy() if parent is None else compact_labels(parent)
    anchor = compact_intersection_labels(initial, leaf)
    frame = SegmentationFrame(
        leaf_labels=leaf,
        parent_labels=parent,
        anchor_labels=anchor,
        leaf_to_parent=label_parent_lookup(leaf, parent),
        anchor_to_leaf=label_parent_lookup(anchor, leaf),
        split_diagnostics=diagnostics,
    )
    return validate_segmentation_frame(frame)


def _parallel_frames(count, operation, n_jobs=None):
    if count == 0:
        return []
    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, count)
    if n_jobs == 1:
        return [operation(index) for index in range(count)]
    results = [None] * count
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(operation, index): index for index in range(count)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def build_segmentation_window(
    point_maps,
    *,
    conf_map=None,
    top_conf_percentile=None,
    segment_mode="depth",
    depth_merge_thresh=0.1,
    normal_method="cross",
    geometry_seg_profile="baseline_params",
    rgb_images=None,
    split_score_thresh=0.10,
    split_aux_confirmation=True,
    n_jobs=None,
):
    point_maps = np.asarray(point_maps)
    if point_maps.ndim != 4 or point_maps.shape[-1] != 3:
        raise ValueError("point_maps must have shape (N, H, W, 3)")
    if segment_mode not in SEGMENT_MODES:
        raise ValueError(
            f"Unknown segment_mode: {segment_mode!r}; expected one of {SEGMENT_MODES}."
        )
    if segment_mode in ("geometry", "layer_atomic_split") and normal_method not in NORMAL_METHODS:
        raise ValueError(
            f"Unknown normal_method: {normal_method!r}; expected one of {NORMAL_METHODS}."
        )
    params = get_felzenszwalb_params(segment_mode, geometry_seg_profile)
    count = point_maps.shape[0]

    def operation(index):
        if segment_mode == "depth":
            initial, merged, _ = segment_depth_felzenszwalb_rag_stages(
                point_maps[index, ..., -1],
                depth_merge_thresh,
                conf_map,
                top_conf_percentile,
                **params,
                batch_idx=index,
            )
            return _frame_from_layers(initial, merged)
        if segment_mode == "geometry":
            stage_op = (
                segment_geometry_felzenszwalb_rag_baseline_params_stages
                if geometry_seg_profile == "baseline_params"
                else segment_geometry_felzenszwalb_rag_stages
            )
            stages = stage_op(
                point_maps[index, ..., -1],
                conf_map=conf_map,
                point_map=point_maps,
                top_conf_percentile=top_conf_percentile,
                depth_merge_thresh=depth_merge_thresh,
                normal_method=normal_method,
                **params,
                batch_idx=index,
            )
            return _frame_from_layers(
                stages.initial_labels, stages.merged_labels
            )
        stages = segment_point_map_layer_atomic_stages(
            point_maps[index],
            depth_merge_thresh,
            split=segment_mode == "layer_atomic_split",
            rgb_images=rgb_images,
            normal_method=normal_method,
            split_score_thresh=split_score_thresh,
            split_aux_confirmation=split_aux_confirmation,
            conf_map=conf_map,
            top_conf_percentile=top_conf_percentile,
            **params,
            batch_idx=index,
        )
        diagnostics = (
            None
            if stages.split_diagnostics is None
            else stages.split_diagnostics.as_dict()
        )
        return _frame_from_layers(
            stages.initial_labels,
            stages.refined_labels,
            parent=stages.merged_labels,
            diagnostics=diagnostics,
        )

    frames = tuple(_parallel_frames(count, operation, n_jobs=n_jobs))
    return SegmentationWindow(frames=frames, segment_mode=segment_mode)
