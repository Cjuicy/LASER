from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.segmentation.base import SegmentationStrategy
from pipeline.artifacts import ReconstructionArtifact
from pipeline.config import (
    AnchorPropagationConfig,
    ReconstructionMode,
    RegistrationConfig,
    SegmentationConfig,
    WindowConfig,
)
from reconstruction.prediction_stream import WindowPrediction


@dataclass(frozen=True)
class ReconstructionContext:
    predictions: Iterable[WindowPrediction]
    frame_ids: tuple[int, ...]
    segmentation_strategy: SegmentationStrategy
    anchor_propagator: AnchorPropagator
    segmentation_config: SegmentationConfig
    anchor_config: AnchorPropagationConfig
    registration_config: RegistrationConfig
    window_config: WindowConfig
    reconstruction_mode: ReconstructionMode


class ReconstructionModeRunner(Protocol):
    def run(self, context: ReconstructionContext) -> ReconstructionArtifact:
        raise NotImplementedError
