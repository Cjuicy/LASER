from __future__ import annotations

import numpy as np

from inference_engine.segmentation.base import (
    SegmentationResult,
    compact_labels,
    validate_strategy_inputs,
)
from inference_engine.utils.geometry_segmentation import (
    segment_geometry_felzenszwalb_rag,
)
from pipeline.config import SegmentationConfig, SegmentationMethod


class GeometrySegmentationStrategy:
    name = SegmentationMethod.GEOMETRY

    def __init__(self, config: SegmentationConfig) -> None:
        self.confidence_keep_ratio = config.confidence_keep_ratio
        self.depth_merge_threshold = config.depth_merge_threshold
        self.seg_scale = config.felzenszwalb.scale
        self.seg_sigma = config.felzenszwalb.sigma
        self.seg_min_size = config.felzenszwalb.min_size
        self.normal_method = config.geometry.normal_method
        self.normal_threshold_degrees = (
            config.geometry.normal_threshold_degrees
        )

    def segment(
        self,
        point_maps: np.ndarray,
        confidence: np.ndarray | None,
        images: np.ndarray | None,
    ) -> list[SegmentationResult]:
        del images
        points, confidence_array = validate_strategy_inputs(
            point_maps,
            confidence,
        )
        results = []
        for frame_index, point_map in enumerate(points):
            frame_confidence = (
                None
                if confidence_array is None
                else confidence_array[frame_index]
            )
            labels = segment_geometry_felzenszwalb_rag(
                point_map[..., 2],
                conf_map=frame_confidence,
                point_map=point_map,
                confidence_keep_ratio=self.confidence_keep_ratio,
                depth_merge_thresh=self.depth_merge_threshold,
                normal_thresh_deg=self.normal_threshold_degrees,
                seg_scale=self.seg_scale,
                seg_sigma=self.seg_sigma,
                seg_min_size=self.seg_min_size,
                normal_method=self.normal_method,
            )
            labels = compact_labels(labels)
            results.append(
                SegmentationResult(
                    labels=labels,
                    diagnostics={
                        "method": self.name.value,
                        "region_count": int(np.unique(labels).size),
                        "normal_method": self.normal_method,
                        "normal_threshold_degrees": (
                            self.normal_threshold_degrees
                        ),
                    },
                )
            )
        return results

