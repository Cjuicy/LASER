from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StubPi3:
    pass


pi3_model_module = types.ModuleType("pi3.models.pi3")
pi3_model_module.Pi3 = StubPi3
sys.modules["pi3.models.pi3"] = pi3_model_module

inference_engine_module = types.ModuleType("inference_engine")
inference_engine_module.StreamingWindowEngineLC = object
sys.modules["inference_engine"] = inference_engine_module

load_fn_module = types.ModuleType("vggt.utils.load_fn")
load_fn_module.load_and_preprocess_images = lambda paths: paths
sys.modules["vggt.utils.load_fn"] = load_fn_module

save_func_module = types.ModuleType("eval.save_func")
save_func_module.save_for_viser = lambda *args, **kwargs: None
sys.modules["eval.save_func"] = save_func_module

loop_closure_module = types.ModuleType("loop_closure.loop_closure")
loop_closure_module.LoopClosureEngine = object
sys.modules["loop_closure.loop_closure"] = loop_closure_module

config_utils_module = types.ModuleType("loop_closure.utils.config_utils")
config_utils_module.load_config = lambda path: {}
sys.modules["loop_closure.utils.config_utils"] = config_utils_module

torch.cuda.get_device_capability = lambda: (8, 0)

import demo_lc


def test_model_checkpoint_defaults_to_local_safetensors():
    args = demo_lc.get_args_parser().parse_args([])

    assert args.model_ckpt == "weights/model.safetensors"


def test_model_checkpoint_can_be_overridden():
    args = demo_lc.get_args_parser().parse_args([
        "--model_ckpt",
        "custom/model.pt",
    ])

    assert args.model_ckpt == "custom/model.pt"


def test_segmentation_options_default_to_depth_baseline():
    args = demo_lc.get_args_parser().parse_args([])

    assert args.segment_mode == "depth"
    assert args.normal_method == "cross"
    assert args.geometry_seg_profile == "baseline_params"


def test_loop_registration_confidence_defaults_to_top_thirty_percent():
    args = demo_lc.get_args_parser().parse_args([])

    assert args.registration_top_confidence_ratio == 0.3


@pytest.mark.parametrize("segment_mode", ["depth", "geometry", "layer_atomic"])
def test_parser_accepts_all_segmentation_modes(segment_mode):
    args = demo_lc.get_args_parser().parse_args([
        "--segment_mode",
        segment_mode,
    ])

    assert args.segment_mode == segment_mode


def test_parser_accepts_geometry_options():
    args = demo_lc.get_args_parser().parse_args([
        "--normal_method",
        "sobel",
        "--geometry_seg_profile",
        "legacy",
    ])

    assert args.normal_method == "sobel"
    assert args.geometry_seg_profile == "legacy"


def test_load_model_rejects_missing_checkpoint_before_model_construction(
    tmp_path,
    monkeypatch,
):
    missing_checkpoint = tmp_path / "missing.safetensors"
    args = demo_lc.get_args_parser().parse_args([
        "--model_ckpt",
        str(missing_checkpoint),
    ])
    monkeypatch.setattr(
        demo_lc,
        "Pi3",
        lambda: pytest.fail("Pi3 must not be constructed"),
    )

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        demo_lc.load_model(args)


def test_load_model_forwards_segmentation_options(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    args = demo_lc.get_args_parser().parse_args([
        "--model_ckpt",
        str(checkpoint),
        "--depth_refine",
        "--segment_mode",
        "geometry",
        "--normal_method",
        "sobel",
        "--geometry_seg_profile",
        "legacy",
    ])
    captured = {}

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, checkpoint, strict):
            return "loaded"

    def fake_engine(model, **kwargs):
        captured.update(kwargs)
        return "engine"

    monkeypatch.setattr(demo_lc, "Pi3", FakeModel)
    monkeypatch.setattr(demo_lc.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(demo_lc, "StreamingWindowEngineLC", fake_engine)

    model, engine = demo_lc.load_model(args)

    assert isinstance(model, FakeModel)
    assert engine == "engine"
    assert captured["segment_mode"] == "geometry"
    assert captured["normal_method"] == "sobel"
    assert captured["geometry_seg_profile"] == "legacy"
    assert captured["registration_top_confidence_ratio"] == 0.3


def test_natural_sort_key_orders_embedded_numbers_numerically():
    names = ["frame10.jpg", "Frame2.jpg", "frame1.jpg"]

    assert sorted(names, key=demo_lc.natural_sort_key) == [
        "frame1.jpg",
        "Frame2.jpg",
        "frame10.jpg",
    ]


def test_discover_images_filters_sorts_then_samples(tmp_path):
    for name in [
        "frame10.jpg",
        "frame2.PNG",
        "frame1.jpeg",
        "notes.txt",
        "frame3.jpg",
    ]:
        (tmp_path / name).touch()

    image_names = demo_lc.discover_images(str(tmp_path), sample_interval=2)

    assert [Path(path).name for path in image_names] == [
        "frame1.jpeg",
        "frame3.jpg",
    ]


def test_build_loop_closure_engine_passes_canonical_manifest(
    monkeypatch,
    tmp_path,
):
    calls = []

    def recording_engine(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(demo_lc, "LoopClosureEngine", recording_engine)
    args = demo_lc.get_args_parser().parse_args([
        "--data_path",
        str(tmp_path),
    ])
    image_paths = [str(tmp_path / "frame1.png")]

    demo_lc.build_loop_closure_engine(
        {},
        args,
        object(),
        image_paths,
        tmp_path / "cache_lc",
    )

    assert calls[0][1]["image_paths"] is image_paths
    assert calls[0][1]["registration_top_confidence_ratio"] == 0.3
