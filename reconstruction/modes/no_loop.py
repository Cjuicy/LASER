from __future__ import annotations

from collections.abc import Callable

import torch

from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.segmentation import build_temporal_graphs
from inference_engine.utils.geometry import apply_sim3_to_pose, homogenize_points
from pipeline.artifacts import (
    ReconstructionArtifact,
    ReconstructionDiagnostics,
)
from pipeline.config import ReconstructionMode
from reconstruction.modes.base import ReconstructionContext
from reconstruction.shared import (
    as_numpy,
    mutual_confidence_mask,
    reference_intrinsic,
    unproject_with_reference,
)


class NoLoopReconstructionMode:
    def __init__(
        self,
        *,
        register_adjacent: Callable = register_adjacent_windows,
        apply_pose_sim3: Callable = apply_sim3_to_pose,
        build_graphs: Callable = build_temporal_graphs,
    ) -> None:
        self._register_adjacent = register_adjacent
        self._apply_pose_sim3 = apply_pose_sim3
        self._build_graphs = build_graphs

    def run(self, context: ReconstructionContext) -> ReconstructionArtifact:
        if context.reconstruction_mode is not ReconstructionMode.NO_LOOP:
            raise ValueError("NoLoopReconstructionMode requires no_loop context")
        overlap = context.window_config.overlap
        previous = None
        previous_graph = None
        intrinsic = None
        previous_frame_end = None
        point_chunks = []
        pose_chunks = []
        confidence_chunks = []
        segmentation_summaries = []
        prediction_key = None
        window_count = 0

        for expected_index, prediction in enumerate(context.predictions):
            spec = prediction.spec
            if spec.index != expected_index:
                raise ValueError("no_loop predictions must be in WindowSpec order")
            if prediction_key is None:
                prediction_key = prediction.prediction_key
            elif prediction.prediction_key != prediction_key:
                raise ValueError(
                    "no_loop prediction windows use different cache keys"
                )
            window_count += 1
            local_points = prediction.local_points
            camera_poses = prediction.camera_poses
            confidence = prediction.confidence
            if previous is None:
                intrinsic = reference_intrinsic(local_points)
                local_points = unproject_with_reference(local_points, intrinsic)
            else:
                if intrinsic is None:
                    raise RuntimeError("no_loop reference intrinsic is unavailable")
                local_points = unproject_with_reference(local_points, intrinsic)
                mask = mutual_confidence_mask(
                    previous["confidence"],
                    confidence,
                    overlap,
                    context.registration_config.confidence_keep_ratio,
                )
                scale, rotation, translation = self._register_adjacent(
                    previous["local_points"][-overlap:],
                    local_points[:overlap],
                    previous["camera_poses"][-overlap:],
                    camera_poses[:overlap],
                    mask,
                )
                scale_tensor = torch.as_tensor(
                    scale,
                    device=local_points.device,
                    dtype=local_points.dtype,
                )
                if scale_tensor.numel() != 1 or not torch.isfinite(scale_tensor).all():
                    raise ValueError("no_loop adjacent Sim(3) scale is invalid")
                local_points = scale_tensor * local_points
                camera_poses = self._apply_pose_sim3(
                    camera_poses,
                    scale,
                    rotation,
                    translation,
                )

            results = context.segmentation_strategy.segment(
                as_numpy(local_points),
                as_numpy(confidence),
                as_numpy(prediction.images),
            )
            graph = self._build_graphs(
                results,
                context.segmentation_config.temporal_iou_threshold,
            )
            segmentation_summaries.extend(
                {
                    "window_index": spec.index,
                    "frame_index": spec.frame_start + offset,
                    **dict(result.diagnostics),
                }
                for offset, result in enumerate(results)
            )

            if previous is not None and context.anchor_config.enabled:
                scale_mask = context.anchor_propagator.propagate(
                    as_numpy(previous["local_points"]),
                    as_numpy(local_points),
                    previous_graph,
                    graph,
                    overlap,
                ).to(device=local_points.device, dtype=local_points.dtype)
                if scale_mask.shape != (*local_points.shape[:-1], 1):
                    raise ValueError("no_loop anchor scale mask has invalid shape")
                if not torch.isfinite(scale_mask).all():
                    raise ValueError("no_loop anchor scale mask must be finite")
                local_points = local_points * scale_mask

            trim = 0
            if previous_frame_end is not None:
                trim = max(0, previous_frame_end - spec.frame_start)
            point_chunks.append(local_points[trim:])
            pose_chunks.append(camera_poses[trim:])
            confidence_chunks.append(confidence[trim:])
            previous_frame_end = max(spec.frame_end, previous_frame_end or spec.frame_end)
            previous = {
                "local_points": local_points,
                "camera_poses": camera_poses,
                "confidence": confidence,
            }
            previous_graph = graph

        if window_count == 0:
            raise ValueError("no_loop requires at least one prediction window")
        local_points = torch.cat(point_chunks).detach().cpu()
        camera_poses = torch.cat(pose_chunks).detach().cpu()
        confidence = torch.cat(confidence_chunks).detach().cpu()
        if local_points.shape[0] != len(context.frame_ids):
            raise ValueError("no_loop aggregation did not cover every frame once")
        global_points = torch.einsum(
            "nij,nhwj->nhwi",
            camera_poses,
            homogenize_points(local_points),
        )[..., :3]
        return ReconstructionArtifact(
            schema_version=1,
            frame_ids=context.frame_ids,
            local_points=local_points,
            global_points=global_points.detach().cpu(),
            camera_poses=camera_poses,
            confidence=confidence,
            segmentation_method=context.segmentation_strategy.name,
            reconstruction_mode=ReconstructionMode.NO_LOOP,
            prediction_key=prediction_key,
            diagnostics=ReconstructionDiagnostics(
                stage_timings_ms={},
                segmentation_summaries=tuple(segmentation_summaries),
                candidate_count=0,
                constraint_count=0,
                mode_scalars={"window_count": window_count},
            ),
        )
