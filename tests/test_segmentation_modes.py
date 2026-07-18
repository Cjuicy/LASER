from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_engine.utils import lsa


def _capture_route(monkeypatch):
    calls = {}

    def fake_batch(images, op_func=None, **kwargs):
        calls["images"] = images
        calls["op_func"] = op_func
        calls["kwargs"] = kwargs
        return np.zeros(images.shape[:3], dtype=np.intp)

    def fake_match(labels, iou_thresh):
        calls["labels"] = labels
        calls["iou_thresh"] = iou_thresh
        return "shared_graph"

    monkeypatch.setattr(lsa, "batched_image_op_wrapper", fake_batch)
    monkeypatch.setattr(lsa, "match_segmentation_seq", fake_match)
    return calls


def _inputs():
    point_maps = np.arange(2 * 4 * 5 * 3, dtype=np.float32).reshape(2, 4, 5, 3)
    conf_map = np.linspace(0.0, 1.0, 2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    return point_maps, conf_map


def test_depth_mode_routes_scalar_depth_with_aligned_parameters(monkeypatch):
    calls = _capture_route(monkeypatch)
    point_maps, conf_map = _inputs()

    graph = lsa.make_sp_graph(
        point_maps,
        conf_map=conf_map,
        top_conf_percentile=0.7,
        corr_iou_thresh=0.25,
        segment_mode="depth",
        normal_method="sobel",
        geometry_seg_profile="legacy",
    )

    assert graph == "shared_graph"
    np.testing.assert_array_equal(calls["images"], point_maps[..., -1])
    assert calls["op_func"] is lsa.segment_depth_felzenszwalb_rag
    assert calls["kwargs"] == {
        "depth_merge_thresh": 0.1,
        "conf_map": conf_map,
        "top_conf_percentile": 0.7,
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 500,
    }
    assert calls["iou_thresh"] == 0.25


@pytest.mark.parametrize(
    ("profile", "expected_op", "expected_params"),
    [
        (
            "baseline_params",
            "segment_geometry_felzenszwalb_rag_baseline_params",
            {"seg_scale": 300, "seg_sigma": 1.1, "seg_min_size": 500},
        ),
        (
            "legacy",
            "segment_geometry_felzenszwalb_rag",
            {"seg_scale": 200, "seg_sigma": 1.0, "seg_min_size": 300},
        ),
    ],
)
def test_geometry_mode_routes_depth_and_point_maps_with_profile_parameters(
    monkeypatch,
    profile,
    expected_op,
    expected_params,
):
    calls = _capture_route(monkeypatch)
    point_maps, conf_map = _inputs()

    graph = lsa.make_sp_graph(
        point_maps,
        depth_merge_thresh=0.2,
        conf_map=conf_map,
        top_conf_percentile=0.6,
        segment_mode="geometry",
        normal_method="sobel",
        geometry_seg_profile=profile,
    )

    assert graph == "shared_graph"
    np.testing.assert_array_equal(calls["images"], point_maps[..., -1])
    assert calls["op_func"] is getattr(lsa, expected_op)
    assert calls["kwargs"] == {
        "depth_merge_thresh": 0.2,
        "conf_map": conf_map,
        "top_conf_percentile": 0.6,
        "point_map": point_maps,
        "normal_method": "sobel",
        **expected_params,
    }


def test_layer_atomic_mode_routes_full_point_maps_with_aligned_parameters(monkeypatch):
    calls = _capture_route(monkeypatch)
    point_maps, conf_map = _inputs()

    graph = lsa.make_sp_graph(
        point_maps,
        depth_merge_thresh=0.3,
        conf_map=conf_map,
        top_conf_percentile=0.8,
        segment_mode="layer_atomic",
        normal_method="sobel",
        geometry_seg_profile="legacy",
    )

    assert graph == "shared_graph"
    np.testing.assert_array_equal(calls["images"], point_maps)
    assert calls["op_func"] is lsa.segment_point_map_layer_atomic
    assert calls["kwargs"] == {
        "depth_merge_thresh": 0.3,
        "conf_map": conf_map,
        "top_conf_percentile": 0.8,
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 500,
    }


def test_layer_atomic_split_routes_rgb_and_single_split_configuration(monkeypatch):
    calls = _capture_route(monkeypatch)
    point_maps, conf_map = _inputs()
    rgb_images = np.zeros((2, 3, 4, 5), dtype=np.float32)

    graph = lsa.make_sp_graph(
        point_maps,
        depth_merge_thresh=0.3,
        conf_map=conf_map,
        top_conf_percentile=0.8,
        segment_mode="layer_atomic_split",
        normal_method="sobel",
        rgb_images=rgb_images,
        split_score_thresh=0.17,
        split_aux_confirmation=False,
    )

    assert graph == "shared_graph"
    np.testing.assert_array_equal(calls["images"], point_maps)
    assert calls["op_func"] is lsa.segment_point_map_layer_atomic_split
    assert calls["kwargs"] == {
        "depth_merge_thresh": 0.3,
        "conf_map": conf_map,
        "top_conf_percentile": 0.8,
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 500,
        "rgb_images": rgb_images,
        "normal_method": "sobel",
        "split_score_thresh": 0.17,
        "split_aux_confirmation": False,
    }


def test_router_rejects_unknown_mode_before_batching():
    point_maps, _ = _inputs()

    with pytest.raises(ValueError, match="segment_mode"):
        lsa.make_sp_graph(point_maps, segment_mode="unknown")


def test_geometry_router_rejects_unknown_profile_before_batching():
    point_maps, _ = _inputs()

    with pytest.raises(ValueError, match="geometry_seg_profile"):
        lsa.make_sp_graph(
            point_maps,
            segment_mode="geometry",
            geometry_seg_profile="unknown",
        )


def test_geometry_router_rejects_unknown_normal_method_before_batching():
    point_maps, _ = _inputs()

    with pytest.raises(ValueError, match="normal_method"):
        lsa.make_sp_graph(
            point_maps,
            segment_mode="geometry",
            normal_method="unknown",
        )


def test_split_router_rejects_unknown_normal_method_before_batching():
    point_maps, _ = _inputs()

    with pytest.raises(ValueError, match="normal_method"):
        lsa.make_sp_graph(
            point_maps,
            segment_mode="layer_atomic_split",
            normal_method="unknown",
        )


def test_effective_felzenszwalb_parameters_match_mode_and_profile():
    assert lsa.get_felzenszwalb_params("depth") == {
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 500,
    }
    assert lsa.get_felzenszwalb_params("layer_atomic") == {
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 500,
    }
    assert lsa.get_felzenszwalb_params("layer_atomic_split") == {
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 500,
    }
    assert lsa.get_felzenszwalb_params("geometry", "baseline_params") == {
        "seg_scale": 300,
        "seg_sigma": 1.1,
        "seg_min_size": 500,
    }
    assert lsa.get_felzenszwalb_params("geometry", "legacy") == {
        "seg_scale": 200,
        "seg_sigma": 1.0,
        "seg_min_size": 300,
    }
