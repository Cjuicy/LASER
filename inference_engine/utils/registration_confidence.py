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

    threshold = torch.quantile(
        finite_values,
        1.0 - ratio,
        interpolation="nearest",
    )
    return finite & (confidence >= threshold)
