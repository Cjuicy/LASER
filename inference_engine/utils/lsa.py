import torch
import numpy as np

from .depth import (
    assign_overlap_window_depth_scale,
    match_segmentation_seq,
    segment_depth_felzenszwalb_rag,
)
from .batch_threading import batched_image_op_wrapper
from .geometry_segmentation import (
    segment_geometry_felzenszwalb_rag,
    segment_geometry_felzenszwalb_rag_baseline_params,
)
from .layer_atomic_geometry import (
    segment_point_map_layer_atomic,
    segment_point_map_layer_atomic_split,
)


SEGMENT_MODES = ("depth", "geometry", "layer_atomic", "layer_atomic_split")
NORMAL_METHODS = ("cross", "sobel")
FELZENSZWALB_BASELINE_PARAMS = {
    "seg_scale": 300,
    "seg_sigma": 1.1,
    "seg_min_size": 500,
}
GEOMETRY_SEGMENTATION_PROFILES = {
    "baseline_params": (
        segment_geometry_felzenszwalb_rag_baseline_params,
        FELZENSZWALB_BASELINE_PARAMS,
    ),
    "legacy": (
        segment_geometry_felzenszwalb_rag,
        {"seg_scale": 200, "seg_sigma": 1.0, "seg_min_size": 300},
    ),
}


def get_felzenszwalb_params(segment_mode, geometry_seg_profile="baseline_params"):
    if segment_mode not in SEGMENT_MODES:
        raise ValueError(
            f"Unknown segment_mode: {segment_mode!r}; expected one of {SEGMENT_MODES}."
        )
    if segment_mode == "geometry":
        try:
            _, params = GEOMETRY_SEGMENTATION_PROFILES[geometry_seg_profile]
        except KeyError:
            raise ValueError(
                "Unknown geometry_seg_profile: "
                f"{geometry_seg_profile!r}; expected one of "
                f"{tuple(GEOMETRY_SEGMENTATION_PROFILES)}."
            ) from None
        return dict(params)
    return dict(FELZENSZWALB_BASELINE_PARAMS)


def refine_depth_segments(
        src_pcd,
        tgt_pcd,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4,
        diagnostic_sink=None,
        diagnostic_context=None,
):
    """
    src_pcd: previous window pcd
    tgt_pcd: current window pcd
    src_sp_graphs: previous window superpixel graph
    overlap: window overlap size
    conf_mask: confidence mask
    depth_merge_thresh: percentage confident depth range to be considered as smooth change
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding
    """
    src_depth = src_pcd[..., -1]
    tgt_depth = tgt_pcd[..., -1]

    tgt_scale_mask = align_adjacent_windows_depth_segments(
        src_depth,
        tgt_depth,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh,
        diagnostic_sink=diagnostic_sink,
        diagnostic_context=diagnostic_context,
    )

    return torch.from_numpy(tgt_scale_mask[..., None])


def make_sp_graph(
        point_maps,
        depth_merge_thresh=0.1,
        conf_map=None,
        top_conf_percentile=None,
        corr_iou_thresh=0.3,
        segment_mode=None,
        normal_method="cross",
        geometry_seg_profile="baseline_params",
        rgb_images=None,
        split_score_thresh=0.10,
        split_aux_confirmation=True,
        diagnostic_sink=None,
        diagnostic_context=None,
):
    legacy_layer_atomic_call = segment_mode is None
    if legacy_layer_atomic_call:
        segment_mode = "layer_atomic"
    params = get_felzenszwalb_params(segment_mode, geometry_seg_profile)
    common_kwargs = {
        "depth_merge_thresh": depth_merge_thresh,
        "conf_map": conf_map,
        "top_conf_percentile": top_conf_percentile,
    }
    if not legacy_layer_atomic_call:
        common_kwargs.update(params)

    if segment_mode == "depth":
        images = np.asarray(point_maps)[..., -1]
        segmentation_op = segment_depth_felzenszwalb_rag
    elif segment_mode == "geometry":
        if normal_method not in NORMAL_METHODS:
            raise ValueError(
                f"Unknown normal_method: {normal_method!r}; expected one of {NORMAL_METHODS}."
            )
        images = np.asarray(point_maps)[..., -1]
        segmentation_op = GEOMETRY_SEGMENTATION_PROFILES[geometry_seg_profile][0]
        common_kwargs.update(
            point_map=point_maps,
            normal_method=normal_method,
        )
    elif segment_mode == "layer_atomic":
        images = point_maps
        segmentation_op = segment_point_map_layer_atomic
    else:
        if normal_method not in NORMAL_METHODS:
            raise ValueError(
                f"Unknown normal_method: {normal_method!r}; expected one of {NORMAL_METHODS}."
            )
        images = point_maps
        segmentation_op = segment_point_map_layer_atomic_split
        common_kwargs.update(
            rgb_images=rgb_images,
            normal_method=normal_method,
            split_score_thresh=split_score_thresh,
            split_aux_confirmation=split_aux_confirmation,
        )

    labels = batched_image_op_wrapper(
        images,
        segmentation_op,
        **common_kwargs,
    )
    if diagnostic_sink is not None:
        from inference_engine.diagnostics.segmentation import trace_segmentation_frame

        point_maps_array = np.asarray(point_maps)
        conf_array = None if conf_map is None else np.asarray(conf_map)
        rgb_array = None if rgb_images is None else np.asarray(rgb_images)
        for local_index, formal_labels in enumerate(labels):
            rgb_frame = None
            if segment_mode == "layer_atomic_split" and rgb_array is not None:
                rgb_frame = rgb_array[local_index] if rgb_array.ndim == 4 else rgb_array
                if rgb_frame.shape == (3, *point_maps_array.shape[1:3]):
                    rgb_frame = np.moveaxis(rgb_frame, 0, -1)
            trace_kwargs = {
                "segment_mode": segment_mode,
                "depth_merge_thresh": depth_merge_thresh,
                "conf_map": None if conf_array is None else conf_array[local_index],
                "top_conf_percentile": top_conf_percentile,
                "normal_method": normal_method,
                **params,
            }
            if segment_mode == "layer_atomic_split":
                trace_kwargs.update(
                    rgb_image=rgb_frame,
                    split_score_thresh=split_score_thresh,
                    split_aux_confirmation=split_aux_confirmation,
                )
            trace = trace_segmentation_frame(
                point_maps_array[local_index],
                formal_labels,
                **trace_kwargs,
            )
            arrays = dict(trace["arrays"])
            if trace["merge_trace"] is not None:
                pair_fields = (
                    "left_atom", "right_atom", "same_coarse", "boundary_edges",
                    "boundary_gap_mean", "boundary_gap_median", "boundary_gap_p90",
                    "boundary_gap_p95", "normalized_gap", "limit",
                    "threshold_margin", "accepted", "normal_angle_deg",
                    "boundary_local_normalized_gap_median",
                    "whole_vs_local_scale_mismatch",
                )
                arrays["merge_pair_table"] = np.asarray(
                    [
                        [
                            np.nan if pair.get(field) is None else pair.get(field)
                            for field in pair_fields
                        ]
                        for pair in trace["merge_trace"].pair_table
                    ],
                    dtype=np.float64,
                )
                arrays["merge_events"] = np.asarray(
                    [
                        [
                            event["left_atom"], event["right_atom"],
                            event["component_atoms_after"], event["component_pixels_after"],
                            event["merge_depth_after"],
                        ]
                        for event in trace["merge_trace"].events
                    ],
                    dtype=np.float64,
                )
            diagnostic_sink.emit_segmentation(
                diagnostic_context,
                local_index,
                trace["metrics"],
                arrays,
            )
    graphs = match_segmentation_seq(labels, iou_thresh=corr_iou_thresh)
    if diagnostic_sink is not None:
        from inference_engine.diagnostics.temporal import trace_temporal_graph

        temporal_trace = trace_temporal_graph(graphs)
        diagnostic_sink.emit_temporal(
            diagnostic_context,
            temporal_trace["metrics"],
            temporal_trace["arrays"],
        )
    return graphs


def align_adjacent_windows_depth_segments(
        src_depth,  # N, H, W
        tgt_depth,  # N, H, W
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4,
        diagnostic_sink=None,
        diagnostic_context=None,
):
    """
    src_depth: previous window depth map
    tgt_depth: current window depth map
    src_sp_graphs: previous window superpixel graph (nested list of Vertex)
    tgt_sp_graphs: current window superpixel graph
    overlap: window overlap size
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding

    Return:
        depth_scale_mask: N, H, W for current window pcd
    """

    def _propagate_scale_cache(parent, child, edge_wt):
        if len(parent.cache['scale']) > 0:
            iou_wts = np.asarray(parent.cache['iou'])
            prop_scale = np.dot(np.asarray(parent.cache['scale']), iou_wts / np.sum(iou_wts))
            child.cache['iou'].append(edge_wt)
            child.cache['scale'].append(prop_scale)

    def _get_scale_mask(mask, cache):
        mask = mask.astype(np.float32)
        if len(cache['scale']) > 0:
            iou_wts = np.asarray(cache['iou'])
            mu_scale = np.dot(np.asarray(cache['scale']), iou_wts / np.sum(iou_wts))
        else:
            mu_scale = 1.0
        return mask * mu_scale

    src_depth_overlap = src_depth[-overlap:]
    tgt_depth_overlap = tgt_depth[:overlap]
    src_sp_graphs_overlap = src_sp_graphs[-overlap:]
    tgt_sp_graphs_overlap = tgt_sp_graphs[:overlap]

    for sp_graph in src_sp_graphs_overlap:
        for v in sp_graph:
            v.remove_all_edges()

    # sptial scale initilaization
    assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=corr_iou_thresh
    )
    direct_cache = None
    if diagnostic_sink is not None:
        direct_cache = diagnostic_sink.snapshot_direct_anchors(
            diagnostic_context,
            tgt_sp_graphs,
        )
    # temporal scale propagation
    for tgt_graph_layer in tgt_sp_graphs:
        for v in tgt_graph_layer:
            v.propagate_data_once(_propagate_scale_cache)

    mask_seq = []
    for sp_graph in tgt_sp_graphs:
        mask_frame = sp_graph[0].data_cache_op(_get_scale_mask)
        for v in sp_graph[1:]:
            mask_frame += v.data_cache_op(_get_scale_mask)
        mask_seq.append(mask_frame)

    scale_mask = np.stack(mask_seq)
    if diagnostic_sink is not None:
        from inference_engine.diagnostics.scale import summarize_scale_observations

        scale_summary = summarize_scale_observations(
            tgt_sp_graphs,
            direct_cache=direct_cache,
        )
        diagnostic_sink.emit_scale(
            diagnostic_context,
            scale_summary["metrics"],
            scale_summary["arrays"],
        )
    return scale_mask
