from pathlib import Path
import importlib.util
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from loop_closure import loop_model


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


def test_loop_detector_uses_explicit_manifest_without_rescanning(
    monkeypatch,
    tmp_path,
):
    image_paths = [
        str(tmp_path / "frame2.png"),
        str(tmp_path / "frame10.png"),
    ]
    monkeypatch.setattr(
        loop_model,
        "discover_images",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("explicit manifest must not rescan")
        ),
        raising=False,
    )

    detector = loop_model.LoopDetector(
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
        loop_model,
        "discover_images",
        fake_discover_images,
        raising=False,
    )
    detector = loop_model.LoopDetector(
        image_dir=tmp_path,
        sample_interval=4,
        config=loop_config(),
    )

    assert detector.get_image_paths() is expected
    assert calls == [(tmp_path, 4)]


def load_engine_module(monkeypatch, detector_calls):
    fake_loop_model = types.ModuleType("loop_closure.loop_model")

    class RecordingDetector:
        def __init__(self, **kwargs):
            detector_calls.append(kwargs)

    fake_loop_model.LoopDetector = RecordingDetector
    monkeypatch.setitem(sys.modules, "loop_closure.loop_model", fake_loop_model)

    fake_optimizer_module = types.ModuleType("loop_closure.utils.sim3loop")

    class IdentityOptimizer:
        def __init__(self, config):
            self.calls = []

        def optimize(self, transforms, constraints):
            self.calls.append((transforms, constraints))
            return transforms

    fake_optimizer_module.Sim3LoopOptimizer = IdentityOptimizer
    monkeypatch.setitem(
        sys.modules,
        "loop_closure.utils.sim3loop",
        fake_optimizer_module,
    )

    fake_utils = types.ModuleType("loop_closure.utils.sim3utils")
    fake_utils.process_loop_list = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "loop_closure.utils.sim3utils", fake_utils)

    fake_load = types.ModuleType("utils.load_fn")
    fake_load.load_and_preprocess_images = lambda paths: paths
    monkeypatch.setitem(sys.modules, "utils.load_fn", fake_load)

    fake_pi3 = types.ModuleType("pi3.models.pi3")
    fake_pi3.Pi3 = object
    monkeypatch.setitem(sys.modules, "pi3.models.pi3", fake_pi3)

    fake_engine = types.ModuleType("inference_engine")
    fake_engine.StreamingWindowEngine = object
    monkeypatch.setitem(sys.modules, "inference_engine", fake_engine)

    fake_inference = types.ModuleType("inference_engine.inference_utils")
    fake_inference.register_adjacent_windows = lambda *args: None
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.inference_utils",
        fake_inference,
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


def test_loop_engine_passes_explicit_manifest_to_detector(monkeypatch, tmp_path):
    detector_calls = []
    module = load_engine_module(monkeypatch, detector_calls)
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


def test_zero_loop_constraints_preserve_existing_transforms(monkeypatch, tmp_path):
    detector_calls = []
    module = load_engine_module(monkeypatch, detector_calls)
    image_paths = [str(tmp_path / f"frame{i}.png") for i in range(16)]
    engine = module.LoopClosureEngine(
        engine_config(),
        tmp_path,
        tmp_path / "output",
        object(),
        10,
        5,
        image_paths=image_paths,
    )
    original = [(1.0, torch.eye(3), torch.zeros(3))]
    engine.sim3_list = original

    engine.process_loops([])

    assert engine.sim3_list is original
    assert engine.loop_optimizer.calls == [(original, [])]
