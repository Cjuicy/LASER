from pathlib import Path
import sys

import numpy as np
import pytest
from skimage.segmentation import felzenszwalb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
inference_engine_module = sys.modules.get("inference_engine")
if inference_engine_module is not None and not hasattr(inference_engine_module, "__path__"):
    del sys.modules["inference_engine"]

from inference_engine.utils._segmentation_cy import merge_regions
from inference_engine.utils.depth import (
    segment_depth_felzenszwalb_rag,
    segment_depth_felzenszwalb_rag_stages,
)


@pytest.mark.parametrize("use_confidence", [False, True], ids=["all-depth", "confident-depth"])
def test_stages_preserve_legacy_coarse_labels_and_threshold(use_confidence):
    yy, xx = np.mgrid[:40, :60]
    depth = (1.0 + 0.002 * xx + 0.004 * yy).astype(np.float64)
    depth[:, 30:] += 0.8

    if use_confidence:
        frame_conf = (3 * xx + yy).astype(np.float64)
        conf_map = np.stack([np.flipud(frame_conf), frame_conf])
        confidence_keep_ratio = 0.25
        batch_idx = 1
        conf_thresh = np.quantile(
            frame_conf.reshape(-1),
            1.0 - confidence_keep_ratio,
            method="nearest",
        )
        conf_depth = depth[frame_conf >= conf_thresh]
    else:
        conf_map = None
        confidence_keep_ratio = None
        batch_idx = None
        conf_depth = depth

    expected_initial = felzenszwalb(
        depth,
        scale=300,
        sigma=1.1,
        min_size=20,
    )
    expected_threshold = np.float64(0.1 * (conf_depth.max() - conf_depth.min()))
    expected_coarse = merge_regions(expected_initial, depth, expected_threshold)

    initial, coarse, threshold = segment_depth_felzenszwalb_rag_stages(
        depth,
        depth_merge_thresh=0.1,
        conf_map=conf_map,
        confidence_keep_ratio=confidence_keep_ratio,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=20,
        batch_idx=batch_idx,
    )
    legacy = segment_depth_felzenszwalb_rag(
        depth,
        depth_merge_thresh=0.1,
        conf_map=conf_map,
        confidence_keep_ratio=confidence_keep_ratio,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=20,
        batch_idx=batch_idx,
    )

    np.testing.assert_array_equal(initial, expected_initial)
    np.testing.assert_array_equal(coarse, expected_coarse)
    np.testing.assert_array_equal(legacy, expected_coarse)
    assert initial.shape == depth.shape
    assert threshold == expected_threshold
