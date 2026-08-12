from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.segmentation import build_temporal_graphs
from inference_engine.utils.geometry import (
    accumulate_sim3,
    apply_sim3_to_pose,
    closed_form_inverse_sim3,
)
from loop_closure.detection import LoopDetector
from loop_closure.evidence import LoopEvidenceProvider
from loop_closure.methods.corrected import (
    CorrectedLoopProcessor,
    identity_sim3_like,
)
from loop_closure.types import Sim3, validate_sim3
from loop_closure.utils.sim3loop import Sim3LoopOptimizer
from pipeline.artifacts import ReconstructionArtifact, ReconstructionDiagnostics
from pipeline.config import OptimizerConfig, ReconstructionMode
from reconstruction.modes.base import ReconstructionContext
from reconstruction.shared import as_numpy, mutual_confidence_mask


@dataclass(frozen=True)
class CorrectedWindowState:
    window_index: int
    frame_start: int
    frame_end: int
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_labels: tuple[np.ndarray, ...]
    anchor_scale_mask: torch.Tensor | None
    sim3_abs: Sim3
    sim3_edge: Sim3 | None
    segmentation_diagnostics: tuple[Mapping[str, object], ...]


class CorrectedReconstructionMode:
    def __init__(
        self,
        *,
        detector: LoopDetector,
        evidence: LoopEvidenceProvider,
        optimizer_config: OptimizerConfig,
        optimizer: Sim3LoopOptimizer | None = None,
        processor: CorrectedLoopProcessor | None = None,
        register_adjacent: Callable = register_adjacent_windows,
        apply_pose_sim3: Callable = apply_sim3_to_pose,
        build_graphs: Callable = build_temporal_graphs,
    ) -> None:
        self.detector = detector
        self.evidence = evidence
        self.processor = processor or CorrectedLoopProcessor(
            optimizer_config,
            optimizer=optimizer,
            apply_pose_sim3=apply_pose_sim3,
        )
        self._register_adjacent = register_adjacent
        self._apply_pose_sim3 = apply_pose_sim3
        self._build_graphs = build_graphs
        self.trace: tuple[CorrectedWindowState, ...] = ()

    def run(self, context: ReconstructionContext) -> ReconstructionArtifact:
        if context.reconstruction_mode is not ReconstructionMode.CORRECTED:
            raise ValueError(
                "CorrectedReconstructionMode requires corrected context"
            )
        if context.image_manifest is None or context.images is None:
            raise ValueError("corrected context requires manifest and images")
        if context.images.shape[0] != len(context.image_manifest):
            raise ValueError("corrected manifest and image lengths differ")

        overlap = context.window_config.overlap
        states: list[CorrectedWindowState] = []
        previous_graph = None
        prediction_key = None
        segmentation_summaries: list[Mapping[str, object]] = []

        for expected_index, prediction in enumerate(context.predictions):
            spec = prediction.spec
            if spec.index != expected_index:
                raise ValueError("corrected predictions must be in WindowSpec order")
            if prediction_key is None:
                prediction_key = prediction.prediction_key
            elif prediction.prediction_key != prediction_key:
                raise ValueError(
                    "corrected prediction windows use different cache keys"
                )

            local_points = prediction.local_points
            camera_poses = prediction.camera_poses
            confidence = prediction.confidence
            anchor_scale_mask = None
            if states:
                previous = states[-1]
                mask = mutual_confidence_mask(
                    previous.confidence,
                    confidence,
                    overlap,
                    context.registration_config.confidence_keep_ratio,
                    context="corrected sequential registration",
                )
                sim3_abs = self._register_adjacent(
                    previous.local_points[-overlap:],
                    local_points[:overlap],
                    previous.camera_poses[-overlap:],
                    camera_poses[:overlap],
                    mask,
                )
                validate_sim3(sim3_abs, context="corrected absolute Sim(3)")
                sim3_edge = accumulate_sim3(
                    closed_form_inverse_sim3(*previous.sim3_abs),
                    sim3_abs,
                )
                validate_sim3(
                    sim3_edge,
                    context="corrected sequential Sim(3) edge",
                )
                scale, rotation, translation = sim3_abs
                local_points = (
                    torch.as_tensor(
                        scale,
                        device=local_points.device,
                        dtype=local_points.dtype,
                    )
                    * local_points
                )
                camera_poses = self._apply_pose_sim3(
                    camera_poses,
                    scale,
                    rotation.to(camera_poses),
                    translation.to(camera_poses),
                )
            else:
                sim3_abs = identity_sim3_like()
                sim3_edge = None

            results = context.segmentation_strategy.segment(
                as_numpy(local_points),
                as_numpy(confidence),
                as_numpy(prediction.images),
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

            if states and context.anchor_config.enabled:
                if previous_graph is None:
                    raise RuntimeError(
                        "corrected predecessor segmentation graph is unavailable"
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
                    raise ValueError("corrected anchor scale mask has invalid shape")
                if not torch.isfinite(anchor_scale_mask).all():
                    raise ValueError("corrected anchor scale mask must be finite")
                local_points = anchor_scale_mask * local_points

            states.append(
                CorrectedWindowState(
                    window_index=spec.index,
                    frame_start=spec.frame_start,
                    frame_end=spec.frame_end,
                    local_points=local_points,
                    camera_poses=camera_poses,
                    confidence=confidence,
                    segmentation_labels=labels,
                    anchor_scale_mask=anchor_scale_mask,
                    sim3_abs=sim3_abs,
                    sim3_edge=sim3_edge,
                    segmentation_diagnostics=diagnostics,
                )
            )
            previous_graph = graph

        if not states or prediction_key is None:
            raise ValueError("corrected requires at least one prediction window")
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
            raise ValueError("corrected aggregation did not cover every frame once")

        return ReconstructionArtifact(
            schema_version=1,
            frame_ids=context.frame_ids,
            local_points=local_points,
            global_points=global_points,
            camera_poses=camera_poses,
            confidence=confidence,
            segmentation_method=context.segmentation_strategy.name,
            reconstruction_mode=ReconstructionMode.CORRECTED,
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
    "CorrectedLoopProcessor",
    "CorrectedReconstructionMode",
    "CorrectedWindowState",
]
