from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from pipeline.config import SegmentationMethod


DiagnosticValue = float | int | bool | str


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    diagnostics: Mapping[str, DiagnosticValue]

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 2:
            raise ValueError("segmentation labels must be two-dimensional")
        unique = np.unique(labels)
        np.testing.assert_array_equal(unique, np.arange(unique.size))


class SegmentationStrategy(Protocol):
    name: SegmentationMethod

    def segment(
        self,
        point_maps: np.ndarray,
        confidence: np.ndarray | None,
        images: np.ndarray | None,
    ) -> list[SegmentationResult]:
        raise NotImplementedError


def compact_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("segmentation labels must be two-dimensional")
    _, compact = np.unique(labels, return_inverse=True)
    return compact.reshape(labels.shape).astype(np.intp, copy=False)


def validate_strategy_inputs(
    point_maps: np.ndarray,
    confidence: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(point_maps)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("point_maps must have shape (N, H, W, 3)")
    if points.shape[0] < 1 or points.shape[1] < 1 or points.shape[2] < 1:
        raise ValueError("point_maps dimensions must be non-empty")

    if confidence is None:
        return points, None
    confidence_array = np.asarray(confidence)
    if confidence_array.shape != points.shape[:-1]:
        raise ValueError(
            "confidence must have shape (N, H, W) matching point_maps"
        )
    return points, confidence_array


def build_temporal_graphs(
    results: list[SegmentationResult],
    temporal_iou_threshold: float,
):
    from inference_engine.utils.depth import match_segmentation_seq

    labels = [result.labels for result in results]
    return match_segmentation_seq(
        labels,
        iou_thresh=temporal_iou_threshold,
    )

