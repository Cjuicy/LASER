from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping, Protocol

import torch

from inference_engine.anchor_propagation import AnchorPropagator
from inference_engine.segmentation.base import SegmentationStrategy
from inference_engine.segmentation.window_reference import WindowReferenceRefinement
from pipeline.artifacts import (
    ReconstructionArtifact,
    StagedReconstructionArtifacts,
)
from pipeline.config import (
    AnchorPropagationConfig,
    ReconstructionMode,
    RegistrationConfig,
    SegmentationConfig,
    WindowConfig,
)
from pipeline.manifest import ImageManifest
from reconstruction.prediction_stream import WindowPrediction


@dataclass(frozen=True)
class ReconstructionContext:
    predictions: Iterable[WindowPrediction]
    frame_ids: tuple[int, ...]
    segmentation_strategy: SegmentationStrategy
    window_reference_refiner: WindowReferenceRefinement
    anchor_propagator: AnchorPropagator
    segmentation_config: SegmentationConfig
    anchor_config: AnchorPropagationConfig
    registration_config: RegistrationConfig
    window_config: WindowConfig
    reconstruction_mode: ReconstructionMode
    image_manifest: ImageManifest | None = None
    images: torch.Tensor | None = None


@dataclass(frozen=True)
class ReconstructionTensors:
    local_points: torch.Tensor
    global_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    mode_scalars: Mapping[str, int | float]


class ReconstructionModeRunner(Protocol):
    def run(
        self,
        context: ReconstructionContext,
    ) -> ReconstructionArtifact | StagedReconstructionArtifacts:
        raise NotImplementedError
