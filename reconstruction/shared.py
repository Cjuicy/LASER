from __future__ import annotations

import torch

from inference_engine.inference_utils import (
    estimate_pseudo_depth_and_intrinsics,
    unproject_depth_to_local_points,
)
from inference_engine.utils.registration_confidence import (
    intersect_confidence_masks,
    select_top_confidence_mask,
)


def as_numpy(value: torch.Tensor):
    return value.detach().cpu().numpy()


def reference_intrinsic(local_points: torch.Tensor) -> torch.Tensor:
    _, intrinsics = estimate_pseudo_depth_and_intrinsics(local_points)
    if (
        not isinstance(intrinsics, torch.Tensor)
        or intrinsics.ndim != 3
        or intrinsics.shape[0] < 1
        or intrinsics.shape[1:] != (3, 3)
        or not torch.isfinite(intrinsics).all()
    ):
        raise ValueError("first window intrinsic estimator returned invalid intrinsics")
    return intrinsics[0]


def unproject_with_reference(
    local_points: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    return unproject_depth_to_local_points(local_points[..., 2], intrinsic)


def mutual_confidence_mask(
    previous_confidence: torch.Tensor,
    current_confidence: torch.Tensor,
    overlap: int,
    keep_ratio: float,
) -> torch.Tensor:
    previous = select_top_confidence_mask(
        previous_confidence[-overlap:],
        keep_ratio,
    )
    current = select_top_confidence_mask(
        current_confidence[:overlap],
        keep_ratio,
    )
    return intersect_confidence_masks(
        previous,
        current,
        context="no_loop sequential registration",
    )
