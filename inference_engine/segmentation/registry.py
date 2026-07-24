from __future__ import annotations

from inference_engine.segmentation.base import SegmentationStrategy
from inference_engine.segmentation.depth import DepthSegmentationStrategy
from inference_engine.segmentation.geometry import GeometrySegmentationStrategy
from pipeline.config import SegmentationConfig, SegmentationMethod


def build_segmentation_strategy(
    config: SegmentationConfig,
) -> SegmentationStrategy:
    if config.method is SegmentationMethod.DEPTH:
        return DepthSegmentationStrategy(config)
    if config.method is SegmentationMethod.GEOMETRY:
        return GeometrySegmentationStrategy(config)
    if config.method is SegmentationMethod.ATOMIC:
        raise NotImplementedError("atomic segmentation is not registered yet")
    raise ValueError(f"unsupported segmentation method: {config.method!r}")

