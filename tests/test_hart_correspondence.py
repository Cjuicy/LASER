import numpy as np

from inference_engine.anchor_propagation.correspondence import (
    build_track_window,
    candidate_relations,
    classify_edges,
    sparse_pair_relations,
)


def _relation(relations, src, tgt):
    return next(item for item in relations if (item.src_id, item.tgt_id) == (src, tgt))


def test_sparse_relations_report_iou_and_both_containments():
    src = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]])
    tgt = np.asarray([[0, 2, 2, 1], [0, 2, 2, 1]])

    relation = _relation(sparse_pair_relations(src, tgt), 0, 0)

    assert relation.intersection == 2
    assert relation.iou == 0.5
    assert relation.src_coverage == 0.5
    assert relation.tgt_coverage == 1.0


def test_one_sided_containment_keeps_small_split_child():
    src = np.zeros((2, 10), dtype=np.intp)
    tgt = np.zeros_like(src)
    tgt[:, -2:] = 1

    relations = candidate_relations(src, tgt, threshold=0.3)

    assert {(item.src_id, item.tgt_id) for item in relations} == {(0, 0), (0, 1)}


def test_split_creates_lineage_segment_and_merge_retains_secondary_edge():
    first = np.zeros((2, 4), dtype=np.intp)
    split = np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.intp)
    merged = np.zeros_like(first)

    tracks = build_track_window((first, split, merged), threshold=0.3)

    assert tracks.segment_ids[1][0] == tracks.segment_ids[0][0]
    child_segment = int(tracks.segment_ids[1][1])
    assert child_segment != tracks.segment_ids[0][0]
    assert tracks.lineage_parents[child_segment] == tracks.segment_ids[0][0]
    assert len(tracks.primary_edges[1]) == 1
    assert len(tracks.secondary_edges[1]) == 1


def test_primary_tie_break_is_deterministic_by_smallest_source_id():
    src = np.asarray([[0, 0, 1, 1]])
    tgt = np.zeros_like(src)
    primary, secondary = classify_edges(candidate_relations(src, tgt, threshold=0.3))

    assert [(edge.src_id, edge.tgt_id) for edge in primary] == [(0, 0)]
    assert [(edge.src_id, edge.tgt_id) for edge in secondary] == [(1, 0)]
