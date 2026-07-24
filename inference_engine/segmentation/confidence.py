from __future__ import annotations

import numpy as np


def select_numpy_top_confidence_mask(
    confidence: np.ndarray,
    keep_ratio: float,
) -> np.ndarray:
    values = np.asarray(confidence)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        raise ValueError("confidence contains no finite values")
    if not np.isfinite(keep_ratio) or not 0.0 < keep_ratio <= 1.0:
        raise ValueError("confidence keep_ratio must be in (0, 1]")
    threshold = np.quantile(
        finite_values,
        1.0 - keep_ratio,
        method="higher",
    )
    return finite & (values >= threshold)
