from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
inference_engine_module = sys.modules.get("inference_engine")
if inference_engine_module is not None and not hasattr(inference_engine_module, "__path__"):
    del sys.modules["inference_engine"]

from inference_engine.utils import lsa
from inference_engine.utils.layer_atomic_geometry import (
    segment_point_map_layer_atomic,
)


def test_make_sp_graph_dispatches_complete_point_maps_to_layer_atomic(monkeypatch):
    point_maps = np.arange(2 * 3 * 4 * 3, dtype=np.float64).reshape(2, 3, 4, 3)
    conf_map = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    labels = np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4)
    expected_graph = object()
    batch_calls = []
    match_calls = []

    def fake_batched_image_op_wrapper(images, operation, **kwargs):
        batch_calls.append((images, operation, kwargs))
        return labels

    def fake_match_segmentation_seq(received_labels, *, iou_thresh):
        match_calls.append((received_labels, iou_thresh))
        return expected_graph

    monkeypatch.setattr(lsa, "batched_image_op_wrapper", fake_batched_image_op_wrapper)
    monkeypatch.setattr(lsa, "match_segmentation_seq", fake_match_segmentation_seq)

    result = lsa.make_sp_graph(
        point_maps,
        depth_merge_thresh=0.125,
        conf_map=conf_map,
        top_conf_percentile=0.75,
        corr_iou_thresh=0.35,
    )

    assert len(batch_calls) == 1
    received_points, operation, kwargs = batch_calls[0]
    assert received_points is point_maps
    assert operation is segment_point_map_layer_atomic
    assert kwargs.keys() == {
        "depth_merge_thresh",
        "conf_map",
        "top_conf_percentile",
    }
    assert kwargs["depth_merge_thresh"] == 0.125
    assert kwargs["conf_map"] is conf_map
    assert kwargs["top_conf_percentile"] == 0.75
    assert len(match_calls) == 1
    received_labels, iou_thresh = match_calls[0]
    assert received_labels is labels
    assert iou_thresh == 0.35
    assert result is expected_graph
