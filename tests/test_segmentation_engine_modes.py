import numpy as np
import pytest
import torch

from inference_engine import streaming_window_engine as swe_module
from inference_engine.streaming_window_engine import StreamingWindowEngine
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC


def _engine(tmp_path, **kwargs):
    return StreamingWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        benchmark_latency=False,
        **kwargs,
    )


def test_engine_defaults_to_depth_and_aligned_geometry_configuration(tmp_path):
    engine = _engine(tmp_path, depth_refine=False)

    assert engine.segment_mode == "depth"
    assert engine.normal_method == "cross"
    assert engine.geometry_seg_profile == "baseline_params"


@pytest.mark.parametrize("mode", ["depth", "geometry", "layer_atomic"])
def test_engine_accepts_all_explicit_modes_when_refinement_is_enabled(tmp_path, mode):
    engine = _engine(tmp_path / mode, segment_mode=mode, depth_refine=True)

    assert engine.segment_mode == mode


@pytest.mark.parametrize("mode", ["geometry", "layer_atomic"])
def test_non_depth_mode_requires_depth_refinement(tmp_path, mode):
    with pytest.raises(ValueError, match="depth_refine"):
        _engine(tmp_path / mode, segment_mode=mode, depth_refine=False)


def test_engine_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="segment_mode"):
        _engine(tmp_path, segment_mode="unknown", depth_refine=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("normal_method", "unknown", "normal_method"),
        ("geometry_seg_profile", "unknown", "geometry_seg_profile"),
    ],
)
def test_geometry_mode_rejects_unknown_geometry_options(tmp_path, field, value, message):
    with pytest.raises(ValueError, match=message):
        _engine(
            tmp_path,
            segment_mode="geometry",
            depth_refine=True,
            **{field: value},
        )


def test_parent_segment_graph_helper_forwards_effective_configuration(monkeypatch, tmp_path):
    calls = []
    expected_graph = object()

    def fake_make_sp_graph(point_maps, **kwargs):
        calls.append((point_maps, kwargs))
        return expected_graph

    monkeypatch.setattr(swe_module, "make_sp_graph", fake_make_sp_graph)
    engine = _engine(
        tmp_path,
        segment_mode="geometry",
        normal_method="sobel",
        geometry_seg_profile="legacy",
        depth_refine=True,
    )
    local_points = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
    conf = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    graph = engine._build_segment_graph(local_points, conf)

    assert graph is expected_graph
    assert len(calls) == 1
    point_maps, kwargs = calls[0]
    np.testing.assert_array_equal(point_maps, local_points.numpy())
    np.testing.assert_array_equal(kwargs.pop("conf_map"), conf.numpy())
    assert kwargs == {
        "top_conf_percentile": engine.top_conf_percentile,
        "segment_mode": "geometry",
        "normal_method": "sobel",
        "geometry_seg_profile": "legacy",
    }


def test_loop_closure_engine_forwards_parent_segmentation_configuration(tmp_path):
    engine = StreamingWindowEngineLC(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        cache_root=str(tmp_path),
        depth_refine=True,
        segment_mode="layer_atomic",
        normal_method="sobel",
        geometry_seg_profile="legacy",
    )

    assert engine.segment_mode == "layer_atomic"
    assert engine.normal_method == "sobel"
    assert engine.geometry_seg_profile == "legacy"


@pytest.mark.parametrize(
    ("mode", "profile", "expected"),
    [
        ("depth", "baseline_params", "scale=300, sigma=1.1, min_size=500"),
        ("geometry", "baseline_params", "scale=300, sigma=1.1, min_size=500"),
        ("geometry", "legacy", "scale=200, sigma=1.0, min_size=300"),
        ("layer_atomic", "baseline_params", "scale=300, sigma=1.1, min_size=500"),
    ],
)
def test_engine_prints_effective_segmentation_configuration(
    tmp_path,
    capsys,
    mode,
    profile,
    expected,
):
    _engine(
        tmp_path / f"{mode}-{profile}",
        segment_mode=mode,
        geometry_seg_profile=profile,
        depth_refine=True,
    )

    output = capsys.readouterr().out
    assert f"mode={mode}" in output
    assert expected in output
    if mode == "geometry":
        assert f"profile={profile}" in output
        assert "normal=cross" in output
