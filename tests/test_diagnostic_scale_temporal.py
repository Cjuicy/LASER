import copy

import numpy as np
from pi3.utils.graph import Vertex

from inference_engine.diagnostics.scale import summarize_scale_observations
from inference_engine.diagnostics.temporal import summarize_temporal_graph, trace_temporal_graph
from inference_engine.utils.lsa import align_adjacent_windows_depth_segments


def _vertex(mask, scales=(), ious=()):
    return Vertex(data=np.asarray(mask, bool), default_cache={"scale": list(scales), "iou": list(ious)})


def test_scale_summary_classifies_sources_and_dispersion():
    a = _vertex([[1, 1], [0, 0]], (2.0, 2.2), (1.0, 1.0))
    b = _vertex([[0, 0], [1, 0]], (1.5,), (.5,))
    c = _vertex([[0, 0], [0, 1]])
    direct = [[{"scale": [2.0, 2.2], "iou": [1.0, 1.0]}, {"scale": [], "iou": []}, {"scale": [], "iou": []}]]

    result = summarize_scale_observations([[a, b, c]], direct_cache=direct)

    assert result["metrics"]["source_segment_count"] == {"direct": 1, "propagated": 1, "fallback": 1}
    assert result["metrics"]["source_pixel_count"] == {"direct": 2, "propagated": 1, "fallback": 1}
    assert result["metrics"]["scale_log_mad_quantiles"]["p50"] is not None
    assert result["arrays"]["source_map"].tolist() == [[[1, 1], [2, 0]]]
    assert result["arrays"]["scale_map"][0, 1, 1] == 1.0
    assert result["metrics"]["max_propagation_hops"] is None


def test_temporal_summary_captures_iou_churn_and_split_merge_events():
    a = _vertex([[1, 1], [0, 0]])
    b = _vertex([[0, 0], [1, 1]])
    c = _vertex([[1, 0], [0, 0]])
    d = _vertex([[0, 1], [1, 1]])
    a.add_edges([c, d], [.8, .4])
    b.add_edge(d, .7)

    summary = summarize_temporal_graph([[a, b], [c, d]])

    assert summary["edge_count"] == 3
    assert summary["one_to_many_count"] == 1
    assert summary["many_to_one_count"] == 1
    assert summary["matched_area_ratio"] == 1.0
    assert summary["weighted_mean_iou"] == np.mean([.8, .4, .7])
    assert summary["max_segment_lifetime"] == 2
    dense = trace_temporal_graph([[a, b], [c, d]])
    assert dense["arrays"]["temporal_outgoing_degree_map"].shape == (2, 2, 2)
    assert dense["arrays"]["temporal_incoming_degree_map"][1].max() == 2


def test_alignment_observer_does_not_change_scale_mask_or_caches():
    src_depth = np.ones((1, 2, 2), dtype=float)
    tgt_depth = np.full((1, 2, 2), 2.0)
    src = [[_vertex(np.ones((2, 2), bool))]]
    tgt = [[_vertex(np.ones((2, 2), bool))]]
    src_observed = copy.deepcopy(src)
    tgt_observed = copy.deepcopy(tgt)

    class Recorder:
        def __init__(self):
            self.scale = []
        def snapshot_direct_anchors(self, context, graphs):
            return [[{"scale": list(v.cache["scale"]), "iou": list(v.cache["iou"])} for v in layer] for layer in graphs]
        def emit_scale(self, context, metrics, arrays=None):
            self.scale.append((metrics, arrays))

    baseline = align_adjacent_windows_depth_segments(src_depth, tgt_depth, src, tgt, 1)
    recorder = Recorder()
    observed = align_adjacent_windows_depth_segments(
        src_depth, tgt_depth, src_observed, tgt_observed, 1,
        diagnostic_sink=recorder, diagnostic_context=object(),
    )

    np.testing.assert_allclose(observed, baseline)
    np.testing.assert_allclose(tgt_observed[0][0].cache["scale"], tgt[0][0].cache["scale"])
    assert len(recorder.scale) == 1
