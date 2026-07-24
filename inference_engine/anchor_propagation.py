from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from inference_engine.utils.depth import assign_overlap_window_depth_scale


def align_adjacent_windows_depth_segments(
    src_depth,
    tgt_depth,
    src_sp_graphs,
    tgt_sp_graphs,
    overlap,
    corr_iou_thresh=0.4,
):
    def _propagate_scale_cache(parent, child, edge_wt):
        if len(parent.cache["scale"]) > 0:
            iou_wts = np.asarray(parent.cache["iou"])
            prop_scale = np.dot(
                np.asarray(parent.cache["scale"]),
                iou_wts / np.sum(iou_wts),
            )
            child.cache["iou"].append(edge_wt)
            child.cache["scale"].append(prop_scale)

    def _get_scale_mask(mask, cache):
        mask = mask.astype(np.float32)
        if len(cache["scale"]) > 0:
            iou_wts = np.asarray(cache["iou"])
            mu_scale = np.dot(
                np.asarray(cache["scale"]),
                iou_wts / np.sum(iou_wts),
            )
        else:
            mu_scale = 1.0
        return mask * mu_scale

    src_depth_overlap = src_depth[-overlap:]
    tgt_depth_overlap = tgt_depth[:overlap]
    src_sp_graphs_overlap = src_sp_graphs[-overlap:]
    tgt_sp_graphs_overlap = tgt_sp_graphs[:overlap]

    for sp_graph in src_sp_graphs_overlap:
        for vertex in sp_graph:
            vertex.remove_all_edges()

    assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=corr_iou_thresh,
    )
    for target_graph_layer in tgt_sp_graphs:
        for vertex in target_graph_layer:
            vertex.propagate_data_once(_propagate_scale_cache)

    mask_sequence = []
    for sp_graph in tgt_sp_graphs:
        mask_frame = sp_graph[0].data_cache_op(_get_scale_mask)
        for vertex in sp_graph[1:]:
            mask_frame += vertex.data_cache_op(_get_scale_mask)
        mask_sequence.append(mask_frame)

    return np.stack(mask_sequence)


@dataclass(frozen=True)
class AnchorPropagator:
    correspondence_iou_threshold: float = 0.4

    def __post_init__(self) -> None:
        threshold = self.correspondence_iou_threshold
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "correspondence_iou_threshold must be in [0, 1]"
            )

    def propagate(
        self,
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap: int,
    ) -> torch.Tensor:
        scale = align_adjacent_windows_depth_segments(
            source_points[..., -1],
            target_points[..., -1],
            source_graphs,
            target_graphs,
            overlap,
            corr_iou_thresh=self.correspondence_iou_threshold,
        )
        return torch.from_numpy(scale[..., None])
