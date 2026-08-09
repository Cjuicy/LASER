from __future__ import annotations

import logging
import math
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
    closed_form_inverse_sim3,
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


logger = logging.getLogger(__name__)


def _identity_sim3_like(reference: Sim3 | None = None) -> Sim3:
    if reference is None:
        return 1.0, torch.eye(3), torch.zeros(3)
    _, rotation, translation = reference
    return (
        1.0,
        torch.eye(
            rotation.shape[-1],
            dtype=rotation.dtype,
            device=rotation.device,
        ),
        torch.zeros_like(translation),
    )


def build_local_loop_constraint(
    sim3_abs_a: Sim3,
    sim3_abs_b: Sim3,
    global_alignment_a: Sim3,
    global_alignment_b: Sim3,
) -> Sim3:
    for context, transform in (
        ("sim3_abs_a", sim3_abs_a),
        ("sim3_abs_b", sim3_abs_b),
        ("global_alignment_a", global_alignment_a),
        ("global_alignment_b", global_alignment_b),
    ):
        validate_sim3(transform, context=context)

    global_correction = accumulate_sim3(
        global_alignment_b,
        closed_form_inverse_sim3(*global_alignment_a),
    )
    constraint_ab = accumulate_sim3(
        closed_form_inverse_sim3(*sim3_abs_b),
        accumulate_sim3(global_correction, sim3_abs_a),
    )
    validate_sim3(
        constraint_ab,
        context="corrected local loop constraint",
    )
    return constraint_ab


class CorrectedWindowEngine(StreamingWindowEngine):
    """Immediate-application worker from the corrected pipeline."""

    def _save_cache(self):
        if not isinstance(self.prev_window_cache, WindowCache):
            raise TypeError("corrected cache must be a WindowCache")
        torch.save(
            self.prev_window_cache.to_payload(),
            self.temp_cache_dir / f"window_cache_{self.cache_id}.pt",
        )
        self.cache_id += 1

    def _registration_worker(self):
        target_graph = None

        while True:
            item = self.registration_queue.get()
            if item is STOP_SIGNAL:
                return
            spec, working_window, inference_duration = item
            if spec.index != self.cache_id:
                raise ValueError(
                    "corrected windows must be registered in "
                    "WindowSpec order"
                )
            started = time.perf_counter()
            for key in tuple(working_window):
                if isinstance(working_window[key], torch.Tensor):
                    working_window[key] = working_window[key].squeeze(0)

            target_mask = shared.select_top_confidence_mask(
                working_window["conf"][: self.overlap],
                self.registration_confidence_keep_ratio,
            )
            anchor_scale_mask = None

            if self.prev_window_cache is None:
                sim3_abs = _identity_sim3_like()
                loop_state = {
                    "tag": LoopMethod.CORRECTED.value,
                    "sim3_abs": sim3_abs,
                    "anchor_scale_applied": True,
                }
                if self.anchor_enabled:
                    target_graph = self._build_segment_graph(
                        working_window["local_points"],
                        working_window["conf"],
                        working_window.get("images"),
                    )
            else:
                previous_mask = shared.select_top_confidence_mask(
                    self.prev_window_cache.confidence[-self.overlap :],
                    self.registration_confidence_keep_ratio,
                )
                mutual_mask = shared.intersect_confidence_masks(
                    previous_mask,
                    target_mask,
                    context="corrected sequential registration",
                )
                sim3_abs = register_adjacent_windows(
                    self.prev_window_cache.local_points[-self.overlap :],
                    working_window["local_points"][: self.overlap],
                    self.prev_window_cache.camera_poses[-self.overlap :],
                    working_window["camera_poses"][: self.overlap],
                    mutual_mask,
                )
                validate_sim3(
                    sim3_abs,
                    context="corrected absolute Sim(3)",
                )
                previous_abs = self.prev_window_cache.loop_state[
                    "sim3_abs"
                ]
                sim3_edge = accumulate_sim3(
                    closed_form_inverse_sim3(*previous_abs),
                    sim3_abs,
                )
                validate_sim3(
                    sim3_edge,
                    context="corrected sequential Sim(3) edge",
                )

                scale, rotation, translation = sim3_abs
                working_window["local_points"] = (
                    torch.as_tensor(
                        scale,
                        device=working_window["local_points"].device,
                        dtype=working_window["local_points"].dtype,
                    )
                    * working_window["local_points"]
                )
                working_window["camera_poses"] = apply_sim3_to_pose(
                    working_window["camera_poses"],
                    scale,
                    rotation.to(working_window["camera_poses"]),
                    translation.to(working_window["camera_poses"]),
                )

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
                    working_window["local_points"] = (
                        anchor_scale_mask * working_window["local_points"]
                    )
                loop_state = {
                    "tag": LoopMethod.CORRECTED.value,
                    "sim3_abs": sim3_abs,
                    "sim3_edge": sim3_edge,
                    "anchor_scale_applied": True,
                }

            frame_count = int(working_window["local_points"].shape[0])
            if frame_count != spec.frame_count:
                raise ValueError(
                    "corrected prediction frame count does not match "
                    "WindowSpec"
                )
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
                loop_method=LoopMethod.CORRECTED,
                prediction_key=self.prediction_key,
                model_name=self.model_name,
                checkpoint_digest=self.checkpoint_digest,
                window_index=spec.index,
                frame_start=spec.frame_start,
                frame_end=spec.frame_end,
                local_points=working_window["local_points"],
                camera_poses=working_window["camera_poses"],
                confidence=working_window["conf"],
                segmentation_labels=labels,
                anchor_scale_mask=anchor_scale_mask,
                loop_state=loop_state,
                segmentation_diagnostics=diagnostics,
            )
            self._update_cache(cache, target_graph)
            self._save_cache()
            self.latencies.append(
                inference_duration + time.perf_counter() - started
            )


ConstraintEstimator = Callable[
    [WindowCache, WindowCache, LoopCandidate, float],
    tuple[Sim3, Sim3],
]


class CorrectedLoopClosureStrategy:
    name = LoopMethod.CORRECTED

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

    def create_window_engine(self, **dependencies) -> CorrectedWindowEngine:
        return CorrectedWindowEngine(**dependencies)

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

    def build_constraints(
        self,
        caches: Sequence[WindowCache],
        candidates: tuple[LoopCandidate, ...],
        *,
        constraint_estimator: ConstraintEstimator | None = None,
    ) -> list[LoopConstraint]:
        constraints = []
        seen_pairs = set()
        estimator = constraint_estimator or self.constraint_estimator
        for candidate in candidates:
            try:
                cache_a = self._cache_for_frame(caches, candidate.frame_a)
                cache_b = self._cache_for_frame(caches, candidate.frame_b)
            except ValueError as error:
                logger.warning(
                    "Skipping %s loop candidate frames=%s->%s: %s",
                    self.name.value,
                    candidate.frame_a,
                    candidate.frame_b,
                    error,
                )
                continue
            pair = (cache_a.window_index, cache_b.window_index)
            if pair[0] == pair[1] or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if estimator is None:
                raise ValueError(
                    f"{self.name.value} loop constraints require "
                    "a joint constraint estimator"
                )
            try:
                alignment_a, alignment_b = estimator(
                    cache_a,
                    cache_b,
                    candidate,
                    self.registration_confidence_keep_ratio,
                )
                measurement = build_local_loop_constraint(
                    cache_a.loop_state["sim3_abs"],
                    cache_b.loop_state["sim3_abs"],
                    alignment_a,
                    alignment_b,
                )
                validate_sim3(
                    measurement,
                    context="corrected loop measurement",
                )
            except ValueError as error:
                logger.warning(
                    "Skipping %s loop candidate frames=%s->%s: %s",
                    self.name.value,
                    candidate.frame_a,
                    candidate.frame_b,
                    error,
                )
                continue
            constraints.append(
                LoopConstraint(
                    window_a=pair[0],
                    window_b=pair[1],
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
        original_absolute = tuple(
            cache.loop_state["sim3_abs"] for cache in caches
        )
        if not constraints:
            return LoopSolution(
                optimized_transforms=original_absolute,
                constraints=(),
                used_no_loop_path=True,
            )
        edges = [
            cache.loop_state["sim3_edge"] for cache in caches[1:]
        ]
        optimizer_constraints = [
            (
                constraint.window_a,
                constraint.window_b,
                constraint.measurement,
            )
            for constraint in constraints
        ]
        optimized_edges = self.optimizer.optimize(
            edges,
            optimizer_constraints,
        )
        if len(optimized_edges) != len(edges):
            raise ValueError(
                "corrected optimizer edge count does not match "
                "sequential edge count"
            )
        optimized_absolute = [
            _identity_sim3_like(original_absolute[0])
        ]
        for edge in optimized_edges:
            optimized_absolute.append(
                accumulate_sim3(optimized_absolute[-1], edge)
            )
        return LoopSolution(
            optimized_transforms=tuple(optimized_absolute),
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
                "corrected solution/cache count mismatch"
            )
        if not caches:
            raise ValueError("corrected aggregation requires caches")

        local_chunks = []
        pose_chunks = []
        confidence_chunks = []
        log_scale_deltas = []
        previous_frame_end = None

        for cache, optimized_abs in zip(
            caches,
            solution.optimized_transforms,
            strict=True,
        ):
            original_abs = cache.loop_state["sim3_abs"]
            delta = accumulate_sim3(
                optimized_abs,
                closed_form_inverse_sim3(*original_abs),
            )
            validate_sim3(delta, context="corrected optimization delta")
            scale, rotation, translation = delta
            scale_value = float(torch.as_tensor(scale).item())
            log_scale_deltas.append(abs(math.log(scale_value)))

            local_points = (
                torch.as_tensor(
                    scale,
                    device=cache.local_points.device,
                    dtype=cache.local_points.dtype,
                )
                * cache.local_points.clone()
            )
            camera_poses = apply_sim3_to_pose(
                cache.camera_poses.clone(),
                scale,
                rotation.to(cache.camera_poses),
                translation.to(cache.camera_poses),
            )
            confidence = cache.confidence.clone()

            trim = 0
            if previous_frame_end is not None:
                trim = max(0, previous_frame_end - cache.frame_start)
            local_chunks.append(local_points[trim:])
            pose_chunks.append(camera_poses[trim:])
            confidence_chunks.append(confidence[trim:])
            previous_frame_end = max(
                cache.frame_end,
                previous_frame_end or cache.frame_end,
            )

        local_points = torch.cat(local_chunks, dim=0)
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
                "max_abs_log_scale_delta": max(log_scale_deltas),
                "mean_abs_log_scale_delta": (
                    sum(log_scale_deltas) / len(log_scale_deltas)
                ),
            },
        )
