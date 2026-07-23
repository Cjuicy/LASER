from pathlib import Path
import importlib.util
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_closure.loop_model import LoopDetector
from utils.image_paths import discover_images, natural_sort_key


def loop_config():
    return {
        "Weights": {"SALAD": "weights/dino_salad.ckpt"},
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 32,
                "similarity_threshold": 0.7,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            }
        },
    }


def test_natural_sort_key_orders_numeric_frame_names():
    names = ["frame10.jpg", "Frame2.jpg", "frame1.jpg"]

    assert sorted(names, key=natural_sort_key) == [
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
        "frame3.JPG",
    ]:
        (tmp_path / name).touch()

    image_paths = discover_images(tmp_path, sample_interval=2)

    assert [Path(path).name for path in image_paths] == [
        "frame1.jpeg",
        "frame3.JPG",
    ]


def test_discover_images_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        discover_images(tmp_path / "missing")


@pytest.mark.parametrize("sample_interval", [0, -1])
def test_discover_images_rejects_non_positive_interval(
    tmp_path,
    sample_interval,
):
    with pytest.raises(ValueError, match="sample_interval must be at least 1"):
        discover_images(tmp_path, sample_interval=sample_interval)


def test_loop_detector_uses_explicit_manifest_without_rescanning(
    monkeypatch,
    tmp_path,
):
    image_paths = [
        str(tmp_path / "frame2.png"),
        str(tmp_path / "frame10.png"),
    ]

    def fail_discovery(*args, **kwargs):
        raise AssertionError("explicit manifest must not rescan")

    monkeypatch.setattr(
        "loop_closure.loop_model.discover_images",
        fail_discovery,
        raising=False,
    )
    detector = LoopDetector(
        image_dir=tmp_path,
        sample_interval=3,
        config=loop_config(),
        image_paths=image_paths,
    )

    assert detector.get_image_paths() is image_paths


def test_loop_detector_fallback_uses_shared_discovery(monkeypatch, tmp_path):
    expected = [str(tmp_path / "frame1.png")]
    calls = []

    def fake_discover_images(data_path, sample_interval):
        calls.append((Path(data_path), sample_interval))
        return expected

    monkeypatch.setattr(
        "loop_closure.loop_model.discover_images",
        fake_discover_images,
        raising=False,
    )
    detector = LoopDetector(
        image_dir=tmp_path,
        sample_interval=4,
        config=loop_config(),
    )

    assert detector.get_image_paths() is expected
    assert calls == [(tmp_path, 4)]


def load_loop_engine_module(monkeypatch, detector_calls):
    fake_loop_model = types.ModuleType("loop_closure.loop_model")

    class RecordingDetector:
        def __init__(self, **kwargs):
            detector_calls.append(kwargs)

    fake_loop_model.LoopDetector = RecordingDetector
    monkeypatch.setitem(sys.modules, "loop_closure.loop_model", fake_loop_model)

    fake_optimizer_module = types.ModuleType("loop_closure.utils.sim3loop")

    class IdentityOptimizer:
        def __init__(self, config):
            self.config = config

    fake_optimizer_module.Sim3LoopOptimizer = IdentityOptimizer
    monkeypatch.setitem(
        sys.modules,
        "loop_closure.utils.sim3loop",
        fake_optimizer_module,
    )

    monkeypatch.setitem(
        sys.modules,
        "loop_closure.utils.sim3utils",
        types.ModuleType("loop_closure.utils.sim3utils"),
    )

    fake_load = types.ModuleType("utils.load_fn")
    fake_load.load_and_preprocess_images = lambda paths: paths
    monkeypatch.setitem(sys.modules, "utils.load_fn", fake_load)

    fake_pi3 = types.ModuleType("pi3.models.pi3")
    fake_pi3.Pi3 = object
    monkeypatch.setitem(sys.modules, "pi3.models.pi3", fake_pi3)

    fake_engine = types.ModuleType("inference_engine")
    fake_engine.__path__ = []
    fake_engine.StreamingWindowEngine = object
    fake_engine.StreamingWindowEngineLC = object
    monkeypatch.setitem(sys.modules, "inference_engine", fake_engine)

    fake_inference = types.ModuleType("inference_engine.inference_utils")
    fake_inference.register_adjacent_windows = lambda *args: None
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.inference_utils",
        fake_inference,
    )

    fake_engine_utils = types.ModuleType("inference_engine.utils")
    fake_engine_utils.__path__ = []
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils",
        fake_engine_utils,
    )

    fake_confidence = types.ModuleType(
        "inference_engine.utils.registration_confidence"
    )
    fake_confidence.select_top_confidence_mask = (
        lambda confidence, keep_ratio: torch.ones_like(
            confidence,
            dtype=torch.bool,
        )
    )
    fake_confidence.validate_confidence_keep_ratio = float

    def fake_intersect_confidence_masks(
        source_mask,
        target_mask,
        *,
        context="registration",
    ):
        mutual_mask = source_mask & target_mask
        if not torch.any(mutual_mask):
            raise ValueError(
                f"{context} has no shared high-confidence pixels"
            )
        return mutual_mask

    fake_confidence.intersect_confidence_masks = (
        fake_intersect_confidence_masks
    )
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils.registration_confidence",
        fake_confidence,
    )

    fake_geometry = types.ModuleType("inference_engine.utils.geometry")
    fake_geometry.accumulate_sim3 = lambda first, second: second
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils.geometry",
        fake_geometry,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))

    module_name = "loop_closure._manifest_test_module"
    source_path = (
        Path(__file__).resolve().parents[1]
        / "loop_closure"
        / "loop_closure.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "loop_closure"
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def engine_config():
    config = loop_config()
    config["Model"] = {"loop_chunk_size": 20}
    config["Loop"]["SIM3_Optimizer"] = {
        "lang_version": "python",
        "max_iterations": 30,
        "lambda_init": "1e-6",
    }
    return config


def test_loop_engine_passes_explicit_manifest_to_detector(
    monkeypatch,
    tmp_path,
):
    detector_calls = []
    module = load_loop_engine_module(monkeypatch, detector_calls)
    image_paths = [
        str(tmp_path / "frame2.png"),
        str(tmp_path / "frame10.png"),
    ]

    engine = module.LoopClosureEngine(
        engine_config(),
        tmp_path,
        tmp_path / "output",
        object(),
        10,
        5,
        sample_interval=3,
        image_paths=image_paths,
    )

    assert engine.img_list is image_paths
    assert detector_calls[0]["image_paths"] is image_paths


def test_loop_engine_defaults_registration_to_top_thirty_percent(
    monkeypatch,
    tmp_path,
):
    detector_calls = []
    module = load_loop_engine_module(monkeypatch, detector_calls)

    engine = module.LoopClosureEngine(
        engine_config(),
        tmp_path,
        tmp_path / "output",
        object(),
        10,
        5,
        image_paths=[str(tmp_path / "frame1.png")],
    )

    assert engine.registration_top_confidence_ratio == 0.3
    assert detector_calls[0]["image_paths"] == [
        str(tmp_path / "frame1.png"),
    ]
