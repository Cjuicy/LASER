from .base import (
    SegmentationResult,
    SegmentationStrategy,
    build_temporal_graphs,
)
from .confidence import select_numpy_top_confidence_mask


def build_segmentation_strategy(config):
    from .registry import build_segmentation_strategy as build

    return build(config)


__all__ = [
    "SegmentationResult",
    "SegmentationStrategy",
    "build_segmentation_strategy",
    "build_temporal_graphs",
    "select_numpy_top_confidence_mask",
]

