from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationStages:
    initial_labels: np.ndarray
    merged_labels: np.ndarray
    confidence_threshold: float
    high_confidence_mask: np.ndarray


def confidence_selection(conf, quantile, method=None):
    conf = np.asarray(conf)
    if quantile is None:
        return float("nan"), np.ones(conf.shape, dtype=bool)

    kwargs = {} if method is None else {"method": method}
    threshold = float(np.quantile(conf.reshape(-1), quantile, **kwargs))
    return threshold, np.isfinite(conf) & (conf >= threshold)
