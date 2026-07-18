from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine.utils import layer_atomic_geometry as lag


def test_split_entry_runs_after_merge_and_passes_only_geometry_rgb_and_scales(
    monkeypatch,
):
    points = np.zeros((8, 10, 3), dtype=np.float32)
    rgb_batch = np.zeros((2, 3, 8, 10), dtype=np.float32)
    initial = np.zeros((8, 10), dtype=np.intp)
    coarse = np.zeros_like(initial)
    merged = np.zeros_like(initial)
    atom_scales = np.asarray([0.25], dtype=np.float64)
    calls = []

    monkeypatch.setattr(
        lag,
        "segment_depth_felzenszwalb_rag_stages",
        lambda *args: (initial, coarse, 0.1),
    )
    monkeypatch.setattr(
        lag,
        "_merge_layer_atoms_with_metadata",
        lambda *args: lag.AtomMergeResult(merged, initial, atom_scales),
    )

    def fake_refine(point_map, rgb_image, auto_labels, atom_labels, scales, **kwargs):
        calls.append((point_map, rgb_image, auto_labels, atom_labels, scales, kwargs))
        return auto_labels.copy(), lag.SplitDiagnostics()

    monkeypatch.setattr(lag, "refine_auto_regions", fake_refine)

    result = lag.segment_point_map_layer_atomic_split(
        points,
        depth_merge_thresh=0.1,
        rgb_images=rgb_batch,
        conf_map=np.ones((2, 8, 10), dtype=np.float32),
        top_conf_percentile=0.5,
        seg_min_size=20,
        split_aux_confirmation=False,
        batch_idx=1,
    )

    assert result.shape == merged.shape
    assert len(calls) == 1
    _, rgb, received_auto, received_atoms, received_scales, kwargs = calls[0]
    assert rgb.shape == (8, 10, 3)
    assert received_auto is merged
    assert received_atoms is initial
    assert received_scales is atom_scales
    assert set(kwargs) == {
        "seg_min_size",
        "normal_method",
        "split_score_thresh",
        "split_aux_confirmation",
    }
    assert kwargs["split_aux_confirmation"] is False


def test_old_public_merger_matches_metadata_labels():
    point_map = np.zeros((4, 6, 3), dtype=np.float64)
    point_map[..., 0] = np.arange(6)
    initial = np.tile(np.asarray([0, 0, 1, 1, 2, 2]), (4, 1))
    coarse = np.zeros_like(initial)

    public = lag.merge_layer_atoms(point_map, initial, coarse, 0.1)
    metadata = lag._merge_layer_atoms_with_metadata(
        point_map, initial, coarse, 0.1
    )

    np.testing.assert_array_equal(public, metadata.labels)


def test_rgb_selection_requires_batch_index_for_batched_input():
    rgb = np.zeros((2, 3, 8, 10), dtype=np.float32)

    try:
        lag._select_rgb_frame(rgb, None, 8, 10)
    except ValueError as exc:
        assert "batch_idx" in str(exc)
    else:
        raise AssertionError("batched RGB must require batch_idx")
