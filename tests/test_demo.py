from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StubPi3:
    pass


pi3_model_module = types.ModuleType("pi3.models.pi3")
pi3_model_module.Pi3 = StubPi3
sys.modules["pi3.models.pi3"] = pi3_model_module

inference_engine_module = types.ModuleType("inference_engine")
inference_engine_module.StreamingWindowEngine = object
sys.modules["inference_engine"] = inference_engine_module

load_fn_module = types.ModuleType("utils.load_fn")
load_fn_module.load_and_preprocess_images = lambda paths: paths
sys.modules["utils.load_fn"] = load_fn_module

save_func_module = types.ModuleType("eval.save_func")
save_func_module.save_for_viser = lambda *args, **kwargs: None
sys.modules["eval.save_func"] = save_func_module

import demo


def test_model_checkpoint_defaults_to_local_safetensors():
    args = demo.get_args_parser().parse_args([])

    assert args.model_ckpt == "weights/model.safetensors"


def test_model_checkpoint_can_be_overridden():
    args = demo.get_args_parser().parse_args([
        "--model_ckpt",
        "custom/model.pt",
    ])

    assert args.model_ckpt == "custom/model.pt"


def test_load_model_rejects_missing_checkpoint_before_model_construction(
    tmp_path,
    monkeypatch,
):
    missing_checkpoint = tmp_path / "missing.safetensors"
    args = demo.get_args_parser().parse_args([
        "--model_ckpt",
        str(missing_checkpoint),
    ])
    monkeypatch.setattr(
        demo,
        "Pi3",
        lambda: pytest.fail("Pi3 must not be constructed"),
    )

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        demo.load_model(args)


def test_natural_sort_key_orders_embedded_numbers_numerically():
    names = ["frame10.jpg", "Frame2.jpg", "frame1.jpg"]

    assert sorted(names, key=demo.natural_sort_key) == [
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

    image_names = demo.discover_images(str(tmp_path), sample_interval=2)

    assert [Path(path).name for path in image_names] == [
        "frame1.jpeg",
        "frame3.jpg",
    ]
