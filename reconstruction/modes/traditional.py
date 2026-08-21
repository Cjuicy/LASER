from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.segmentation import build_temporal_graphs
from inference_engine.utils.geometry import apply_sim3_to_pose
from loop_closure.detection import LoopDetector
from loop_closure.evidence import LoopEvidenceProvider
from loop_closure.methods.traditional import (
    TraditionalLoopProcessor,
    identity_sim3,
)
from loop_closure.types import Sim3, validate_sim3
from loop_closure.utils.sim3loop import Sim3LoopOptimizer
from pipeline.artifacts import ReconstructionArtifact, ReconstructionDiagnostics
from pipeline.config import OptimizerConfig, ReconstructionMode
from reconstruction.modes.base import ReconstructionContext
from reconstruction.shared import (
    as_numpy,
    mutual_confidence_mask,
    segment_and_refine_window,
)


@dataclass(frozen=True)
class TraditionalWindowState:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_labels: tuple[np.ndarray, ...]
    anchor_scale_mask: torch.Tensor | None
    relative_sim3: Sim3
    segmentation_diagnostics: tuple[Mapping[str, object], ...]


class TraditionalReconstructionMode:
    def __init__(
        self,
        *,
        detector: LoopDetector,
        evidence: LoopEvidenceProvider,
        optimizer_config: OptimizerConfig,
        optimizer: Sim3LoopOptimizer | None = None,
        processor: TraditionalLoopProcessor | None = None,
        register_adjacent: Callable = register_adjacent_windows,
        apply_pose_sim3: Callable = apply_sim3_to_pose,
        build_graphs: Callable = build_temporal_graphs,
        segment_window: Callable = segment_and_refine_window,
    ) -> None:
        self.detector = detector
        self.evidence = evidence
        self.processor = processor or TraditionalLoopProcessor(
            optimizer_config,
            optimizer=optimizer,
            apply_pose_sim3=apply_pose_sim3,
        )
        self._register_adjacent = register_adjacent
        self._build_graphs = build_graphs
        self._segment_window = segment_window
        self.trace: tuple[TraditionalWindowState, ...] = ()

    def run(self, context: ReconstructionContext) -> ReconstructionArtifact:
        if context.reconstruction_mode is not ReconstructionMode.TRADITIONAL:
            raise ValueError(
                "TraditionalReconstructionMode requires traditional context"
            )
        if context.image_manifest is None or context.images is None:
            raise ValueError("traditional context requires manifest and images")
        if context.images.shape[0] != len(context.image_manifest):
            raise ValueError("traditional manifest and image lengths differ")

        overlap = context.window_config.overlap
        states: list[TraditionalWindowState] = []
        previous_graph = None
        prediction_key = None
        segmentation_summaries: list[Mapping[str, object]] = []

        for expected_index, prediction in enumerate(context.predictions):
            spec = prediction.spec
            if spec.index != expected_index:
                raise ValueError(
                    "traditional predictions must be in WindowSpec order"
                )
            if prediction_key is None:
                prediction_key = prediction.prediction_key
            elif prediction.prediction_key != prediction_key:
                raise ValueError(
                    "traditional prediction windows use different cache keys"
                )

            local_points = prediction.local_points
            camera_poses = prediction.camera_poses
            confidence = prediction.confidence
            if states:
                previous = states[-1]
                mask = mutual_confidence_mask(
                    previous.confidence,
                    confidence,
                    overlap,
                    context.registration_config.confidence_keep_ratio,
                    context="traditional sequential registration",
                )
                relative_sim3 = self._register_adjacent(
                    previous.local_points[-overlap:],
                    local_points[:overlap],
                    previous.camera_poses[-overlap:],
                    camera_poses[:overlap],
                    mask,
                )
                validate_sim3(
                    relative_sim3,
                    context="traditional relative Sim(3)",
                )
            else:
                relative_sim3 = identity_sim3(local_points.device)

            results = self._segment_window(
                strategy=context.segmentation_strategy,
                refiner=context.window_reference_refiner,
                point_maps=local_points,
                camera_poses=camera_poses,
                confidence=confidence,
                images=prediction.images,
                reference_intrinsic=prediction.reference_intrinsic,
            )
            graph = self._build_graphs(
                results,
                context.segmentation_config.temporal_iou_threshold,
            )
            labels = tuple(result.labels.copy() for result in results)
            diagnostics = tuple(dict(result.diagnostics) for result in results)
            segmentation_summaries.extend(
                {
                    "window_index": spec.index,
                    "frame_index": spec.frame_start + offset,
                    **item,
                }
                for offset, item in enumerate(diagnostics)
            )

            anchor_scale_mask = None
            if states and context.anchor_config.enabled:
                if previous_graph is None:
                    raise RuntimeError(
                        "traditional predecessor segmentation graph is unavailable"
                    )
                anchor_scale_mask = context.anchor_propagator.propagate(
                    as_numpy(states[-1].local_points),
                    as_numpy(local_points),
                    previous_graph,
                    graph,
                    overlap,
                ).to(device=local_points.device, dtype=local_points.dtype)
                expected_shape = (*local_points.shape[:-1], 1)
                if anchor_scale_mask.shape != expected_shape:
                    raise ValueError(
                        "traditional anchor scale mask has invalid shape"
                    )
                if not torch.isfinite(anchor_scale_mask).all():
                    raise ValueError(
                        "traditional anchor scale mask must be finite"
                    )

            states.append(
                TraditionalWindowState(
                    window_index=spec.index,
                    frame_start=spec.frame_start,
                    frame_end=spec.frame_end,
                    local_points=local_points,
                    camera_poses=camera_poses,
                    confidence=confidence,
                    segmentation_labels=labels,
                    anchor_scale_mask=anchor_scale_mask,
                    relative_sim3=relative_sim3,
                    segmentation_diagnostics=diagnostics,
                )
            )
            previous_graph = graph

        if not states or prediction_key is None:
            raise ValueError("traditional requires at least one prediction window")
        self.trace = tuple(states)

        candidates = self.detector.detect(
            context.image_manifest,
            context.images,
        )
        constraints = self.processor.build_constraints(
            self.trace,
            candidates,
            self.evidence,
        )
        solution = self.processor.optimize(self.trace, constraints)
        aggregate = self.processor.aggregate(self.trace, solution)
        local_points = aggregate.local_points.detach().cpu()
        global_points = aggregate.global_points.detach().cpu()
        camera_poses = aggregate.camera_poses.detach().cpu()
        confidence = aggregate.confidence.detach().cpu()
        if local_points.shape[0] != len(context.frame_ids):
            raise ValueError("traditional aggregation did not cover every frame once")

        return ReconstructionArtifact(
            schema_version=1,
            frame_ids=context.frame_ids,
            local_points=local_points,
            global_points=global_points,
            camera_poses=camera_poses,
            confidence=confidence,
            segmentation_method=context.segmentation_strategy.name,
            reconstruction_mode=ReconstructionMode.TRADITIONAL,
            prediction_key=prediction_key,
            diagnostics=ReconstructionDiagnostics(
                stage_timings_ms={},
                segmentation_summaries=tuple(segmentation_summaries),
                candidate_count=len(candidates),
                constraint_count=len(constraints),
                mode_scalars=aggregate.mode_scalars,
            ),
        )


__all__ = [
    "TraditionalLoopProcessor",
    "TraditionalReconstructionMode",
    "TraditionalWindowState",
]
