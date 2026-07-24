from __future__ import annotations

import numpy as np

from inference_engine.segmentation.base import (
    SegmentationResult,
    compact_labels,
    validate_strategy_inputs,
)
from inference_engine.utils.depth import segment_depth_felzenszwalb_rag
from pipeline.config import SegmentationConfig, SegmentationMethod


class DepthSegmentationStrategy:
    name = SegmentationMethod.DEPTH

    def __init__(self, config: SegmentationConfig) -> None:
        self.confidence_keep_ratio = config.confidence_keep_ratio
        self.depth_merge_threshold = config.depth_merge_threshold
        self.seg_scale = config.felzenszwalb.scale
        self.seg_sigma = config.felzenszwalb.sigma
        self.seg_min_size = config.felzenszwalb.min_size

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
            labels = segment_depth_felzenszwalb_rag(
                point_map[..., 2],
                depth_merge_thresh=self.depth_merge_threshold,
                conf_map=frame_confidence,
                confidence_keep_ratio=self.confidence_keep_ratio,
                seg_scale=self.seg_scale,
                seg_sigma=self.seg_sigma,
                seg_min_size=self.seg_min_size,
            )
            labels = compact_labels(labels)
            results.append(
                SegmentationResult(
                    labels=labels,
                    diagnostics={
                        "method": self.name.value,
                        "region_count": int(np.unique(labels).size),
                    },
                )
            )
        return results

