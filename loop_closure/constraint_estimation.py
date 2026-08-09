from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from inference_engine.inference_utils import register_adjacent_windows
from inference_engine.models.lazy import (
    LazyModelHandle,
    ModelForwardKind,
)
from inference_engine.utils.registration_confidence import (
    intersect_confidence_masks,
    select_top_confidence_mask,
    validate_confidence_keep_ratio,
)
from pipeline.manifest import ImageManifest

from .methods.base import LoopCandidate, Sim3, WindowCache, validate_sim3


def centered_frame_range(
    frame_start: int,
    frame_end: int,
    center: int,
    chunk_size: int,
) -> tuple[int, int]:
    if frame_start < 0 or frame_end <= frame_start:
        raise ValueError("candidate cache frame range is invalid")
    if not frame_start <= center < frame_end:
        raise ValueError("candidate frame is outside its selected cache")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("loop constraint chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("loop constraint chunk_size must be at least 1")
    size = min(chunk_size, frame_end - frame_start)
    start = center - size // 2
    start = max(frame_start, min(start, frame_end - size))
    return start, start + size


class JointAlignmentEstimator:
    def __init__(
        self,
        model: LazyModelHandle,
        images: torch.Tensor,
        manifest: ImageManifest,
        chunk_size: int,
        confidence_keep_ratio: float,
    ) -> None:
        if not isinstance(model, LazyModelHandle):
            raise ValueError(
                "joint alignment model must be a LazyModelHandle"
            )
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(
                "joint alignment images must have shape (frames, C, H, W)"
            )
        if not isinstance(manifest, ImageManifest):
            raise ValueError(
                "joint alignment manifest must be an ImageManifest"
            )
        if images.shape[0] != len(manifest):
            raise ValueError(
                "joint alignment image and manifest lengths must match"
            )
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise ValueError("loop constraint chunk_size must be an integer")
        if chunk_size < 1:
            raise ValueError("loop constraint chunk_size must be at least 1")

        self.model = model
        self.images = images
        self.manifest = manifest
        self.chunk_size = chunk_size
        self.confidence_keep_ratio = validate_confidence_keep_ratio(
            confidence_keep_ratio
        )

    def __call__(
        self,
        cache_a: WindowCache,
        cache_b: WindowCache,
        candidate: LoopCandidate,
        keep_ratio: float,
    ) -> tuple[Sim3, Sim3]:
        ratio = validate_confidence_keep_ratio(keep_ratio)
        if not math.isclose(
            ratio,
            self.confidence_keep_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "strategy keep ratio does not match joint estimator configuration"
            )
        range_a = centered_frame_range(
            cache_a.frame_start,
            cache_a.frame_end,
            candidate.frame_a,
            self.chunk_size,
        )
        range_b = centered_frame_range(
            cache_b.frame_start,
            cache_b.frame_end,
            candidate.frame_b,
            self.chunk_size,
        )
        images_a = self.images[slice(*range_a)]
        images_b = self.images[slice(*range_b)]
        joint_images = torch.cat((images_a, images_b), dim=0)
        prediction = self._predict(joint_images)
        side_a_count = range_a[1] - range_a[0]
        joint_a = self._prediction_side(prediction, 0, side_a_count)
        joint_b = self._prediction_side(
            prediction,
            side_a_count,
            side_a_count + range_b[1] - range_b[0],
        )
        alignment_a = self._align_side(cache_a, range_a, joint_a, "side A")
        alignment_b = self._align_side(cache_b, range_b, joint_b, "side B")
        return alignment_a, alignment_b

    def _predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.model.predict(
            images,
            kind=ModelForwardKind.JOINT,
        )
        if not isinstance(prediction, Mapping):
            raise ValueError("joint alignment prediction must be a mapping")

        required = ("local_points", "camera_poses", "conf")
        result = {}
        for key in required:
            value = prediction.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    "joint alignment prediction is missing tensor "
                    f"{key!r}"
                )
            if value.ndim < 1 or value.shape[0] != 1:
                raise ValueError(
                    f"joint alignment prediction {key!r} must have "
                    "one model batch"
                )
            result[key] = value.squeeze(0).detach().cpu()
        return result

    @staticmethod
    def _prediction_side(
        prediction: Mapping[str, torch.Tensor],
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        side = {key: value[start:end] for key, value in prediction.items()}
        if any(value.shape[0] != end - start for value in side.values()):
            raise ValueError("joint alignment prediction has too few frames")
        return side

    def _align_side(
        self,
        cache: WindowCache,
        frame_range: tuple[int, int],
        joint: Mapping[str, torch.Tensor],
        side_name: str,
    ) -> Sim3:
        cache_start = frame_range[0] - cache.frame_start
        cache_end = frame_range[1] - cache.frame_start
        cached_points = cache.local_points[cache_start:cache_end].detach().cpu()
        cached_poses = cache.camera_poses[cache_start:cache_end].detach().cpu()
        cached_confidence = cache.confidence[cache_start:cache_end].detach().cpu()
        cached_mask = select_top_confidence_mask(
            cached_confidence,
            self.confidence_keep_ratio,
        )
        joint_mask = select_top_confidence_mask(
            joint["conf"],
            self.confidence_keep_ratio,
        )
        mask = intersect_confidence_masks(
            cached_mask,
            joint_mask,
            context=f"joint {side_name} alignment",
        )
        alignment = register_adjacent_windows(
            cached_points,
            joint["local_points"],
            cached_poses,
            joint["camera_poses"],
            mask,
        )
        validate_sim3(alignment, context=f"joint {side_name} alignment")
        return alignment
