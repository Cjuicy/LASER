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
from utils.image_paths import (
    discover_images as shared_discover_images,
    natural_sort_key as shared_natural_sort_key,
)


def test_model_checkpoint_defaults_to_local_safetensors():
    args = demo_lc.get_args_parser().parse_args([])

    assert args.model_ckpt == "weights/model.safetensors"


def test_loop_defaults_enable_layer_atomic_refinement_and_standard_config():
    args = demo_lc.get_args_parser().parse_args([])

    assert args.config_path == "configs/loop_config.yaml"
    assert args.depth_refine is True


def test_no_depth_refine_disables_layer_atomic_refinement():
    args = demo_lc.get_args_parser().parse_args(["--no-depth-refine"])

    assert args.depth_refine is False


def test_select_runtime_dtype_uses_float32_without_cuda(monkeypatch):
    monkeypatch.setattr(
        demo_lc.torch.cuda,
        "get_device_capability",
        lambda: pytest.fail("CPU selection must not query CUDA capability"),
    )

    assert demo_lc.select_runtime_dtype("cpu") is torch.float32


@pytest.mark.parametrize(
    ("major", "expected"),
    [(8, torch.bfloat16), (7, torch.float16)],
)
def test_select_runtime_dtype_matches_cuda_capability(
    monkeypatch,
    major,
    expected,
):
    monkeypatch.setattr(
        demo_lc.torch.cuda,
        "get_device_capability",
        lambda: (major, 0),
    )

    assert demo_lc.select_runtime_dtype("cuda") is expected


def test_model_checkpoint_can_be_overridden():
    args = demo_lc.get_args_parser().parse_args([
        "--model_ckpt",
        "custom/model.pt",
    ])

    assert args.model_ckpt == "custom/model.pt"


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


def test_demo_lc_reexports_shared_image_helpers():
    assert demo_lc.discover_images is shared_discover_images
    assert demo_lc.natural_sort_key is shared_natural_sort_key


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


def valid_cloud_args_and_config(tmp_path):
    data_path = tmp_path / "image_2"
    data_path.mkdir()
    for name in ["000000.png", "000001.png"]:
        (data_path / name).touch()

    model_ckpt = tmp_path / "model.safetensors"
    config_path = tmp_path / "loop_config.yaml"
    salad = tmp_path / "dino_salad.ckpt"
    dino = tmp_path / "dinov2.pth"
    for path in [model_ckpt, config_path, salad, dino]:
        path.touch()

    args = demo_lc.get_args_parser().parse_args([
        "--data_path",
        str(data_path),
        "--model_ckpt",
        str(model_ckpt),
        "--config_path",
        str(config_path),
        "--window_size",
        "2",
        "--overlap",
        "1",
    ])
    config = {
        "Weights": {
            "SALAD": str(salad),
            "DINO": str(dino),
        },
        "Model": {"loop_enable": True},
    }
    return args, config


def test_load_and_validate_inputs_returns_canonical_images(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)

    loaded_config, image_paths = demo_lc.load_and_validate_inputs(args)

    assert loaded_config is config
    assert [Path(path).name for path in image_paths] == [
        "000000.png",
        "000001.png",
    ]


def test_load_and_validate_inputs_requires_cuda(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)

    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        demo_lc.load_and_validate_inputs(args)


@pytest.mark.parametrize(
    ("window_size", "overlap"),
    [(5, 5), (4, 5), (5, 0)],
)
def test_load_and_validate_inputs_rejects_invalid_windowing(
    tmp_path,
    monkeypatch,
    window_size,
    overlap,
):
    args, config = valid_cloud_args_and_config(tmp_path)
    args.window_size = window_size
    args.overlap = overlap
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)

    with pytest.raises(ValueError, match="window_size must be greater than overlap"):
        demo_lc.load_and_validate_inputs(args)


def test_load_and_validate_inputs_requires_enabled_loop_mode(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    config["Model"]["loop_enable"] = False
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)

    with pytest.raises(ValueError, match="Model.loop_enable must be true"):
        demo_lc.load_and_validate_inputs(args)


def test_load_and_validate_inputs_reports_missing_loop_weight(tmp_path, monkeypatch):
    args, config = valid_cloud_args_and_config(tmp_path)
    Path(config["Weights"]["SALAD"]).unlink()
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)

    with pytest.raises(FileNotFoundError, match="SALAD checkpoint not found"):
        demo_lc.load_and_validate_inputs(args)


def test_load_and_validate_inputs_requires_more_images_than_overlap(
    tmp_path,
    monkeypatch,
):
    args, config = valid_cloud_args_and_config(tmp_path)
    args.window_size = 3
    args.overlap = 2
    monkeypatch.setattr(demo_lc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(demo_lc, "load_config", lambda path: config)

    with pytest.raises(ValueError, match="Image count must be greater than overlap"):
        demo_lc.load_and_validate_inputs(args)


def test_require_complete_cache_files_sorts_numeric_ids(tmp_path):
    for name in [
        "window_cache_10.pt",
        "window_cache_2.pt",
        "window_cache_1.pt",
    ]:
        (tmp_path / name).touch()

    cache_files = demo_lc.require_complete_cache_files(
        tmp_path,
        expected_count=3,
    )

    assert [path.name for path in cache_files] == [
        "window_cache_1.pt",
        "window_cache_2.pt",
        "window_cache_10.pt",
    ]


def test_require_complete_cache_files_rejects_incomplete_output(tmp_path):
    (tmp_path / "window_cache_0.pt").touch()

    with pytest.raises(RuntimeError, match="Window cache count mismatch"):
        demo_lc.require_complete_cache_files(tmp_path, expected_count=2)


def test_run_dynamic_scene_builds_windows_once_and_logs_configuration(
    monkeypatch,
    tmp_path,
    capsys,
):
    image_dir = tmp_path / "image_2"
    image_names = [
        str(image_dir / "000000.png"),
        str(image_dir / "000001.png"),
    ]
    windows = [[image_names[0]], [image_names[1]]]
    window_calls = []
    run_calls = []

    class FakeModel:
        def img_sliding_window(self, received_images):
            window_calls.append(received_images)
            return windows

    monkeypatch.setattr(demo_lc, "model", FakeModel(), raising=False)
    monkeypatch.setattr(
        demo_lc,
        "run_model",
        lambda received_windows: run_calls.append(received_windows),
    )
    args = demo_lc.get_args_parser().parse_args([
        "--data_path",
        str(image_dir),
    ])

    scene_name, expected_windows = demo_lc.run_dynamic_scene(args, image_names)

    assert scene_name == "image_2"
    assert expected_windows == 2
    assert window_calls == [image_names]
    assert run_calls == [windows]
    output = capsys.readouterr().out
    assert "Loop closure: enabled" in output
    assert "Layer-atomic depth refinement: enabled" in output
    assert "first=000000.png, last=000001.png" in output
    assert "Windows: 2 (window_size=10, overlap=5)" in output
