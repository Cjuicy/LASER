import math

import torch


def validate_confidence_keep_ratio(keep_ratio: float) -> float:
    ratio = float(keep_ratio)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("confidence keep ratio must be in (0, 1]")
    return ratio


def select_top_confidence_mask(
    confidence: torch.Tensor,
    keep_ratio: float = 0.3,
) -> torch.Tensor:
    ratio = validate_confidence_keep_ratio(keep_ratio)
    finite = torch.isfinite(confidence)
    finite_values = confidence[finite]
    if finite_values.numel() == 0:
        raise ValueError("confidence contains no finite values")

    quantile_values = (
        finite_values
        if finite_values.dtype in (torch.float32, torch.float64)
        else finite_values.float()
    )
    threshold = torch.quantile(
        quantile_values,
        1.0 - ratio,
        interpolation="higher",
    )
    return finite & (confidence >= threshold)


def intersect_confidence_masks(
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    context: str = "registration",
) -> torch.Tensor:
    if source_mask.shape != target_mask.shape:
        raise ValueError(
            f"{context} confidence mask shapes do not match: "
            f"{tuple(source_mask.shape)} != {tuple(target_mask.shape)}"
        )

    mutual_mask = source_mask & target_mask
    if not torch.any(mutual_mask):
        raise ValueError(
            f"{context} has no shared high-confidence pixels"
        )
    return mutual_mask
