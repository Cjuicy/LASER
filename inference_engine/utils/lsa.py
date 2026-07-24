import numpy as np

from .depth import (
    match_segmentation_seq,
    segment_depth_felzenszwalb_rag,
)
from .batch_threading import batched_image_op_wrapper
from .geometry_segmentation import (
    segment_geometry_felzenszwalb_rag,
    segment_geometry_felzenszwalb_rag_baseline_params,
)
from .layer_atomic_geometry import segment_point_map_atomic


SEGMENT_MODES = ("depth", "geometry", "layer_atomic")
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


def make_sp_graph(
        point_maps,
        depth_merge_thresh=0.1,
        conf_map=None,
        top_conf_percentile=None,
        corr_iou_thresh=0.3,
        segment_mode=None,
        normal_method="cross",
        geometry_seg_profile="baseline_params",
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
    else:
        images = point_maps
        segmentation_op = segment_point_map_atomic

    labels = batched_image_op_wrapper(
        images,
        segmentation_op,
        **common_kwargs,
    )
    return match_segmentation_seq(labels, iou_thresh=corr_iou_thresh)
