from __future__ import annotations

import numpy as np

from pipeline.config import ConfidenceQuantileMethod


def select_numpy_top_confidence_mask(
    confidence: np.ndarray,
    keep_ratio: float,
    method: ConfidenceQuantileMethod | str = ConfidenceQuantileMethod.HIGHER,
) -> np.ndarray:
    values = np.asarray(confidence)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        raise ValueError("confidence contains no finite values")
    if not np.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError("confidence keep_ratio must be in (0, 1]")
    try:
        quantile_method = ConfidenceQuantileMethod(method)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence quantile method must be higher or nearest") from exc
    threshold = np.quantile(
        finite_values,
        1.0 - keep_ratio,
        method=quantile_method.value,
    )
    return finite & (values >= threshold)
