from __future__ import annotations

import numpy as np

from inference_engine.segmentation.base import (
    SegmentationResult,
    compact_labels,
    validate_strategy_inputs,
)
from inference_engine.utils.layer_atomic_geometry import (
    segment_point_map_atomic,
)
from pipeline.config import SegmentationConfig, SegmentationMethod


class AtomicSegmentationStrategy:
    name = SegmentationMethod.ATOMIC

    def __init__(self, config: SegmentationConfig) -> None:
        self.confidence_keep_ratio = config.confidence_keep_ratio
        self.depth_merge_threshold = config.depth_merge_threshold
        self.seg_scale = config.felzenszwalb.scale
        self.seg_sigma = config.felzenszwalb.sigma
        self.seg_min_size = config.felzenszwalb.min_size
        self.normal_method = config.geometry.normal_method
        self.split_mode = config.atomic.split_mode
        self.split_score_threshold = config.atomic.split_score_threshold

    def segment(
        self,
        point_maps: np.ndarray,
        confidence: np.ndarray | None,
        images: np.ndarray | None,
    ) -> list[SegmentationResult]:
        points, confidence_array = validate_strategy_inputs(
            point_maps,
            confidence,
        )
        image_array = None if images is None else np.asarray(images)
        if image_array is not None and (
            image_array.ndim != 4
            or image_array.shape[0] != points.shape[0]
        ):
            raise ValueError(
                "images must have shape (N, H, W, 3) or (N, 3, H, W)"
            )

        results = []
        for frame_index, point_map in enumerate(points):
            frame_confidence = (
                None
                if confidence_array is None
                else confidence_array[frame_index]
            )
            frame_image = (
                None if image_array is None else image_array[frame_index]
            )
            labels, split_diagnostics = segment_point_map_atomic(
                point_map,
                depth_merge_thresh=self.depth_merge_threshold,
                rgb_images=frame_image,
                normal_method=self.normal_method,
                split_score_threshold=self.split_score_threshold,
                split_mode=self.split_mode,
                conf_map=frame_confidence,
                confidence_keep_ratio=self.confidence_keep_ratio,
                seg_scale=self.seg_scale,
                seg_sigma=self.seg_sigma,
                seg_min_size=self.seg_min_size,
            )
            labels = compact_labels(labels)
            diagnostics = split_diagnostics.as_dict()
            diagnostics.update(
                {
                    "method": self.name.value,
                    "region_count": int(np.unique(labels).size),
                }
            )
            results.append(
                SegmentationResult(
                    labels=labels,
                    diagnostics=diagnostics,
                )
            )
        return results
