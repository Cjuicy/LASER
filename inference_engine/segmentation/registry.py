from __future__ import annotations

from inference_engine.segmentation.atomic import AtomicSegmentationStrategy
from inference_engine.segmentation.base import SegmentationStrategy
from inference_engine.segmentation.depth import DepthSegmentationStrategy
from inference_engine.segmentation.geometry import GeometrySegmentationStrategy
from pipeline.config import SegmentationConfig, SegmentationMethod


STRATEGY_FACTORIES = {
    SegmentationMethod.DEPTH: DepthSegmentationStrategy,
    SegmentationMethod.GEOMETRY: GeometrySegmentationStrategy,
    SegmentationMethod.ATOMIC: AtomicSegmentationStrategy,
}


def build_segmentation_strategy(
    config: SegmentationConfig,
) -> SegmentationStrategy:
    try:
        factory = STRATEGY_FACTORIES[config.method]
    except KeyError:
        raise ValueError(
            f"unsupported segmentation method: {config.method!r}"
        ) from None
    return factory(config)
