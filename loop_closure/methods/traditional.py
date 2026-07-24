from __future__ import annotations

import time
from typing import Callable, Sequence

import torch

from inference_engine.inference_utils import (
    estimate_pseudo_depth_and_intrinsics,
    register_adjacent_windows,
    unproject_depth_to_local_points,
)
from inference_engine.streaming_window_engine import (
    STOP_SIGNAL,
    StreamingWindowEngine,
)
from inference_engine.utils.geometry import (
    accumulate_sim3,
    apply_sim3_to_pose,
    homogenize_points,
)
from loop_closure.utils.sim3loop import Sim3LoopOptimizer
from pipeline.config import LoopMethod, OptimizerConfig

from . import shared
from .base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopCandidate,
    LoopConstraint,
    LoopSolution,
    ReconstructionResult,
    Sim3,
    WindowCache,
    validate_sim3,
)


def _identity_sim3(device: str = "cpu") -> Sim3:
    return (
        1.0,
        torch.eye(3, device=device),
        torch.zeros(3, device=device),
    )


def compute_sim3_ab(transform_a: Sim3, transform_b: Sim3) -> Sim3:
    """Preserve the baseline common-frame relative Sim(3) formula."""

    scale_a, rotation_a, translation_a = transform_a
    scale_b, rotation_b, translation_b = transform_b
    scale_ab = scale_b / scale_a
    rotation_ab = rotation_b @ rotation_a.T
    translation_ab = (
        translation_b
        - scale_ab * (rotation_ab @ translation_a)
    )
    return scale_ab, rotation_ab, translation_ab


class TraditionalWindowEngine(StreamingWindowEngine):
    """Baseline worker that defers all transform and anchor application."""

    def _save_cache(self):
        if not isinstance(self.prev_window_cache, WindowCache):
            raise TypeError("traditional cache must be a WindowCache")
        torch.save(
            self.prev_window_cache.to_payload(),
            self.temp_cache_dir / f"window_cache_{self.cache_id}.pt",
        )
        self.cache_id += 1

    def _registration_worker(self):
        ref_intrinsic = None
        target_graph = None

        while True:
            item = self.registration_queue.get()
            if item is STOP_SIGNAL:
                return

            working_window, inference_duration = item
            started = time.perf_counter()
            for key in tuple(working_window):
                if isinstance(working_window[key], torch.Tensor):
                    working_window[key] = working_window[key].squeeze(0)

            target_mask = shared.select_top_confidence_mask(
                working_window["conf"][: self.overlap],
                self.registration_confidence_keep_ratio,
            )

            if self.prev_window_cache is None:
                _, intrinsic = estimate_pseudo_depth_and_intrinsics(
                    working_window["local_points"]
                )
                ref_intrinsic = intrinsic[0]
                working_window["local_points"] = (
                    unproject_depth_to_local_points(
                        working_window["local_points"][..., -1],
                        ref_intrinsic,
                    )
                )
                relative_sim3 = _identity_sim3(self.process_device)
                anchor_scale_mask = None
                if self.anchor_enabled:
                    target_graph = self._build_segment_graph(
                        working_window["local_points"],
                        working_window["conf"],
                        working_window.get("images"),
                    )
            else:
                working_window["local_points"] = (
                    unproject_depth_to_local_points(
                        working_window["local_points"][..., -1],
                        ref_intrinsic,
                    )
                )
                previous_mask = shared.select_top_confidence_mask(
                    self.prev_window_cache.confidence[-self.overlap :],
                    self.registration_confidence_keep_ratio,
                )
                mutual_mask = shared.intersect_confidence_masks(
                    previous_mask,
                    target_mask,
                    context="traditional sequential registration",
                )
                relative_sim3 = register_adjacent_windows(
                    self.prev_window_cache.local_points[-self.overlap :],
                    working_window["local_points"][: self.overlap],
                    self.prev_window_cache.camera_poses[-self.overlap :],
                    working_window["camera_poses"][: self.overlap],
                    mutual_mask,
                )
                validate_sim3(
                    relative_sim3,
                    context="traditional relative Sim(3)",
                )

                anchor_scale_mask = None
                if self.anchor_enabled:
                    target_points = (
                        working_window["local_points"].detach().cpu().numpy()
                    )
                    target_graph = self._build_segment_graph(
                        working_window["local_points"],
                        working_window["conf"],
                        working_window.get("images"),
                    )
                    anchor_scale_mask = self.anchor_propagator.propagate(
                        self.prev_window_cache.local_points.detach()
                        .cpu()
                        .numpy(),
                        target_points,
                        self.anchor_sp_graph,
                        target_graph,
                        self.overlap,
                    ).to(
                        device=working_window["local_points"].device,
                        dtype=working_window["local_points"].dtype,
                    )

            frame_count = int(working_window["local_points"].shape[0])
            frame_start = self.cache_id * (self.window_size - self.overlap)
            labels = (
                ()
                if self.last_segmentation_results is None
                else tuple(
                    result.labels.copy()
                    for result in self.last_segmentation_results
                )
            )
            diagnostics = (
                ()
                if self.last_segmentation_results is None
                else tuple(
                    dict(result.diagnostics)
                    for result in self.last_segmentation_results
                )
            )
            cache = WindowCache(
                schema_version=WINDOW_CACHE_SCHEMA_VERSION,
                loop_method=LoopMethod.TRADITIONAL,
                window_index=self.cache_id,
                frame_start=frame_start,
                frame_end=frame_start + frame_count,
                local_points=working_window["local_points"],
                camera_poses=working_window["camera_poses"],
                confidence=working_window["conf"],
                segmentation_labels=labels,
                anchor_scale_mask=anchor_scale_mask,
                loop_state={
                    "tag": LoopMethod.TRADITIONAL.value,
                    "relative_sim3": relative_sim3,
                    "anchor_scale_applied": False,
                },
                segmentation_diagnostics=diagnostics,
            )
            self._update_cache(cache, target_graph)
            self._save_cache()
            registration_duration = time.perf_counter() - started
            self.latencies.append(
                inference_duration + registration_duration
            )


ConstraintEstimator = Callable[
    [WindowCache, WindowCache, LoopCandidate, float],
    tuple[Sim3, Sim3],
]


class TraditionalLoopClosureStrategy:
    name = LoopMethod.TRADITIONAL

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        registration_confidence_keep_ratio: float,
        *,
        optimizer: Sim3LoopOptimizer | None = None,
        constraint_estimator: ConstraintEstimator | None = None,
    ) -> None:
        self.optimizer_config = optimizer_config
        self.registration_confidence_keep_ratio = float(
            registration_confidence_keep_ratio
        )
        if not 0.0 < self.registration_confidence_keep_ratio <= 1.0:
            raise ValueError(
                "registration_confidence_keep_ratio must be in (0, 1]"
            )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else Sim3LoopOptimizer(optimizer_config, device="cpu")
        )
        self.constraint_estimator = constraint_estimator

    def create_window_engine(self, **dependencies) -> TraditionalWindowEngine:
        return TraditionalWindowEngine(**dependencies)

    @staticmethod
    def _cache_for_frame(
        caches: Sequence[WindowCache],
        frame: int,
    ) -> WindowCache:
        matches = [
            cache
            for cache in caches
            if cache.frame_start <= frame < cache.frame_end
        ]
        if not matches:
            raise ValueError(f"loop candidate frame {frame} is not cached")
        return max(matches, key=lambda cache: cache.frame_start)

    def _estimate_direct_pair(
        self,
        cache_a: WindowCache,
        cache_b: WindowCache,
        candidate: LoopCandidate,
        keep_ratio: float,
    ) -> tuple[Sim3, Sim3]:
        index_a = candidate.frame_a - cache_a.frame_start
        index_b = candidate.frame_b - cache_b.frame_start
        points_a = cache_a.local_points[index_a : index_a + 1]
        points_b = cache_b.local_points[index_b : index_b + 1]
        poses_a = cache_a.camera_poses[index_a : index_a + 1]
        poses_b = cache_b.camera_poses[index_b : index_b + 1]
        mask_a = shared.select_top_confidence_mask(
            cache_a.confidence[index_a : index_a + 1],
            keep_ratio,
        )
        mask_b = shared.select_top_confidence_mask(
            cache_b.confidence[index_b : index_b + 1],
            keep_ratio,
        )
        mutual = shared.intersect_confidence_masks(
            mask_a,
            mask_b,
            context="traditional loop candidate",
        )
        transform_b = register_adjacent_windows(
            points_a,
            points_b,
            poses_a,
            poses_b,
            mutual,
        )
        return _identity_sim3(str(points_a.device)), transform_b

    def build_constraints(
        self,
        caches: Sequence[WindowCache],
        candidates: tuple[LoopCandidate, ...],
    ) -> list[LoopConstraint]:
        constraints = []
        estimator = self.constraint_estimator or self._estimate_direct_pair
        for candidate in candidates:
            cache_a = self._cache_for_frame(caches, candidate.frame_a)
            cache_b = self._cache_for_frame(caches, candidate.frame_b)
            if cache_a.window_index == cache_b.window_index:
                continue
            transform_a, transform_b = estimator(
                cache_a,
                cache_b,
                candidate,
                self.registration_confidence_keep_ratio,
            )
            measurement = compute_sim3_ab(transform_a, transform_b)
            validate_sim3(
                measurement,
                context="traditional loop measurement",
            )
            constraints.append(
                LoopConstraint(
                    window_a=cache_a.window_index,
                    window_b=cache_b.window_index,
                    measurement=measurement,
                    candidate=candidate,
                )
            )
        return constraints

    def optimize(
        self,
        caches: Sequence[WindowCache],
        constraints: Sequence[LoopConstraint],
    ) -> LoopSolution:
        original = tuple(
            cache.loop_state["relative_sim3"] for cache in caches
        )
        for index, transform in enumerate(original):
            validate_sim3(
                transform,
                context=f"traditional cache transform {index}",
            )
        if not constraints:
            return LoopSolution(
                optimized_transforms=original,
                constraints=(),
                used_no_loop_path=True,
            )

        optimizer_constraints = [
            (
                constraint.window_a,
                constraint.window_b,
                constraint.measurement,
            )
            for constraint in constraints
        ]
        optimized_tail = self.optimizer.optimize(
            list(original[1:]),
            optimizer_constraints,
        )
        optimized = (original[0], *tuple(optimized_tail))
        if len(optimized) != len(caches):
            raise ValueError(
                "traditional optimizer transform count does not match caches"
            )
        return LoopSolution(
            optimized_transforms=optimized,
            constraints=tuple(constraints),
            used_no_loop_path=False,
        )

    def aggregate(
        self,
        caches: Sequence[WindowCache],
        solution: LoopSolution,
    ) -> ReconstructionResult:
        if len(caches) != len(solution.optimized_transforms):
            raise ValueError(
                "traditional solution transform count does not match caches"
            )
        if not caches:
            raise ValueError("traditional aggregation requires caches")

        reference = _identity_sim3("cpu")
        point_chunks = []
        pose_chunks = []
        confidence_chunks = []
        previous_frame_end = None

        for cache, relative in zip(
            caches,
            solution.optimized_transforms,
            strict=True,
        ):
            absolute = accumulate_sim3(reference, relative)
            scale, rotation, translation = absolute
            local_points = cache.local_points.clone()
            camera_poses = cache.camera_poses.clone()
            confidence = cache.confidence.clone()
            if cache.anchor_scale_mask is not None:
                local_points = (
                    torch.as_tensor(
                        reference[0],
                        device=local_points.device,
                        dtype=local_points.dtype,
                    )
                    * cache.anchor_scale_mask.to(local_points)
                    * local_points
                )
            else:
                local_points = (
                    torch.as_tensor(
                        scale,
                        device=local_points.device,
                        dtype=local_points.dtype,
                    )
                    * local_points
                )
            camera_poses = apply_sim3_to_pose(
                camera_poses,
                scale,
                rotation.to(camera_poses),
                translation.to(camera_poses),
            )

            trim = 0
            if previous_frame_end is not None:
                trim = max(0, previous_frame_end - cache.frame_start)
            point_chunks.append(local_points[trim:])
            pose_chunks.append(camera_poses[trim:])
            confidence_chunks.append(confidence[trim:])
            previous_frame_end = max(
                cache.frame_end,
                previous_frame_end or cache.frame_end,
            )
            reference = absolute

        local_points = torch.cat(point_chunks, dim=0)
        camera_poses = torch.cat(pose_chunks, dim=0)
        confidence = torch.cat(confidence_chunks, dim=0)
        points = torch.einsum(
            "nij,nhwj->nhwi",
            camera_poses,
            homogenize_points(local_points),
        )[..., :3]
        return ReconstructionResult(
            payload={
                "local_points": local_points,
                "camera_poses": camera_poses,
                "confidence": confidence,
                "points": points,
            },
            summary={
                "loop_method": self.name.value,
                "window_count": len(caches),
                "constraint_count": len(solution.constraints),
                "used_no_loop_path": solution.used_no_loop_path,
            },
        )
