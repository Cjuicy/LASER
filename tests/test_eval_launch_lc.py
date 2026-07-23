from pathlib import Path
import importlib.util
import sys
import types

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_eval_launch(monkeypatch):
    loop_engine_instances = []
    aggregate_calls = []
    cache_by_path = {
        "window_cache_0.pt": {
            "local_points": torch.ones((2, 1, 1, 3)),
            "camera_poses": torch.eye(4).repeat(2, 1, 1),
            "conf": torch.ones((2, 1, 1)),
            "sim3_abs": (1.0, torch.eye(3), torch.zeros(3)),
        },
        "window_cache_1.pt": {
            "local_points": torch.ones((3, 1, 1, 3)),
            "camera_poses": torch.eye(4).repeat(3, 1, 1),
            "conf": torch.ones((3, 1, 1)),
            "sim3_abs": (2.0, torch.eye(3), torch.zeros(3)),
            "sim3_edge": (2.0, torch.eye(3), torch.zeros(3)),
        },
    }
    optimized_absolute = [
        (1.0, torch.eye(3), torch.zeros(3)),
        (3.0, torch.eye(3), torch.zeros(3)),
    ]

    fake_pose_eval = types.ModuleType("eval.pose_eval")
    fake_pose_eval.eval_pose_estimation = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "eval.pose_eval", fake_pose_eval)

    fake_depth_eval = types.ModuleType("eval.depth_eval")
    fake_depth_eval.eval_mono_depth_estimation = (
        lambda *args, **kwargs: None
    )
    monkeypatch.setitem(sys.modules, "eval.depth_eval", fake_depth_eval)

    fake_pi3 = types.ModuleType("pi3.models.pi3")
    fake_pi3.Pi3 = object
    monkeypatch.setitem(sys.modules, "pi3.models.pi3", fake_pi3)

    fake_inference_engine = types.ModuleType("inference_engine")
    fake_inference_engine.__path__ = []

    class FakeStreamingWindowEngine:
        @staticmethod
        def parse_cache_file(cache_path, overlap=0):
            cache = cache_by_path[Path(cache_path).name]
            return {
                key: value[overlap:]
                if isinstance(value, torch.Tensor)
                else value
                for key, value in cache.items()
            }

    class FakeStreamingWindowEngineLC(FakeStreamingWindowEngine):
        @staticmethod
        def aggregate_caches(caches, transforms):
            aggregate_calls.append((caches, transforms))
            return {"camera_poses": torch.eye(4)[None, None]}

        @staticmethod
        def _post_process_pred(prediction):
            return prediction

    fake_inference_engine.VanillaEngine = object
    fake_inference_engine.StreamingWindowEngine = (
        FakeStreamingWindowEngine
    )
    fake_inference_engine.StreamingWindowEngineLC = (
        FakeStreamingWindowEngineLC
    )
    monkeypatch.setitem(
        sys.modules,
        "inference_engine",
        fake_inference_engine,
    )

    fake_inference_utils = types.ModuleType("inference_engine.utils")
    fake_inference_utils.__path__ = []
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils",
        fake_inference_utils,
    )

    fake_cache_utils = types.ModuleType(
        "inference_engine.utils.cache_utils"
    )

    def prepare_caches_for_aggregation(predictions, overlap):
        return [
            {
                key: value[(overlap if index > 0 else 0):]
                if isinstance(value, torch.Tensor)
                else value
                for key, value in prediction.items()
            }
            for index, prediction in enumerate(predictions)
        ]

    fake_cache_utils.prepare_caches_for_aggregation = (
        prepare_caches_for_aggregation
    )
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils.cache_utils",
        fake_cache_utils,
    )

    fake_loop_module = types.ModuleType("loop_closure.loop_closure")

    class FakeLoopClosureEngine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.run_predictions = None
            loop_engine_instances.append(self)

        def run(self, predictions):
            self.run_predictions = predictions
            return optimized_absolute

    fake_loop_module.LoopClosureEngine = FakeLoopClosureEngine
    monkeypatch.setitem(
        sys.modules,
        "loop_closure.loop_closure",
        fake_loop_module,
    )

    fake_config = types.ModuleType("loop_closure.utils.config_utils")
    fake_config.load_config = lambda path: {"config": path}
    monkeypatch.setitem(
        sys.modules,
        "loop_closure.utils.config_utils",
        fake_config,
    )

    fake_misc = types.ModuleType("eval.misc")
    monkeypatch.setitem(sys.modules, "eval.misc", fake_misc)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda: (8, 0),
    )

    module_name = "_eval_launch_lc_test_module"
    source_path = Path(__file__).resolve().parents[1] / "eval_launch.py"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.glob,
        "glob",
        lambda pattern: [
            "window_cache_0.pt",
            "window_cache_1.pt",
        ],
    )
    monkeypatch.setattr(
        module.torch,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("LC evaluation must not rewrite caches")
        ),
    )
    return (
        module,
        loop_engine_instances,
        aggregate_calls,
        optimized_absolute,
    )


def test_streaming_lc_evaluation_uses_manifest_and_delta_aggregation(
    monkeypatch,
    tmp_path,
):
    (
        module,
        loop_engine_instances,
        aggregate_calls,
        optimized_absolute,
    ) = load_eval_launch(monkeypatch)
    image_paths = [
        str(tmp_path / "frame1.png"),
        str(tmp_path / "frame2.png"),
        str(tmp_path / "frame3.png"),
        str(tmp_path / "frame4.png"),
    ]

    class FakeModel:
        cache_dir = tmp_path
        temp_cache_dir = tmp_path / "run"
        delegate = object()
        window_size = 3
        overlap = 1
        registration_top_confidence_ratio = 0.3

        def img_sliding_window(self, images):
            return [images]

        def begin(self):
            pass

        def __call__(self, sample):
            pass

        def end(self):
            pass

    result = module.inference_streaming_model_lc(
        FakeModel(),
        torch.ones((4, 3, 1, 1)),
        tmp_path,
        image_paths=image_paths,
    )

    loop_engine = loop_engine_instances[0]
    assert loop_engine.kwargs["image_paths"] is image_paths
    assert (
        loop_engine.kwargs["registration_top_confidence_ratio"]
        == 0.3
    )
    assert len(loop_engine.run_predictions) == 2
    assert len(aggregate_calls) == 1
    parsed_caches, transforms = aggregate_calls[0]
    assert transforms is optimized_absolute
    assert parsed_caches[0]["local_points"].shape[0] == 2
    assert parsed_caches[1]["local_points"].shape[0] == 2
    assert result["camera_poses"].device.type == "cpu"


def load_pose_eval_module(monkeypatch):
    fake_load = types.ModuleType("utils.load_fn")
    fake_load.load_and_preprocess_images = lambda paths: paths
    monkeypatch.setitem(sys.modules, "utils.load_fn", fake_load)

    fake_geometry = types.ModuleType("utils.geometry")
    fake_geometry.closed_form_inverse_se3 = lambda poses: poses
    monkeypatch.setitem(sys.modules, "utils.geometry", fake_geometry)

    fake_save = types.ModuleType("eval.save_func")
    fake_save.get_tum_poses = lambda poses: poses
    fake_save.save_for_viser = lambda *args, **kwargs: None
    fake_save.get_se3_poses = lambda poses: poses
    monkeypatch.setitem(sys.modules, "eval.save_func", fake_save)

    fake_pose_enc = types.ModuleType("vggt.utils.pose_enc")
    fake_pose_enc.pose_encoding_to_extri_intri = (
        lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setitem(
        sys.modules,
        "vggt.utils.pose_enc",
        fake_pose_enc,
    )

    fake_vo_eval = types.ModuleType("eval.vo_eval")
    for name in (
        "load_traj",
        "eval_metrics",
        "plot_trajectory",
        "process_directory",
        "calculate_averages",
    ):
        setattr(fake_vo_eval, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "eval.vo_eval", fake_vo_eval)

    fake_misc = types.ModuleType("eval.misc")
    monkeypatch.setitem(sys.modules, "eval.misc", fake_misc)

    fake_metadata = types.ModuleType("eval.eval_metadata")
    fake_metadata.dataset_metadata = {}
    monkeypatch.setitem(
        sys.modules,
        "eval.eval_metadata",
        fake_metadata,
    )

    module_name = "eval._pose_eval_manifest_test_module"
    source_path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "pose_eval.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "eval"
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_pose_evaluation_forwards_exact_filelist_to_loop_model(
    monkeypatch,
):
    pose_eval = load_pose_eval_module(monkeypatch)

    image_paths = ["frame1.png", "frame2.png"]
    captured = {}

    class Args:
        model = "streaming_pi3_lc"

    class RecordingModel:
        def __call__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {"prediction": True}

    images = torch.ones((2, 3, 1, 1))
    result = pose_eval.run_model_inference(
        Args(),
        RecordingModel(),
        images,
        "/sequence",
        image_paths,
    )

    assert result == {"prediction": True}
    assert captured["args"] == (images, "/sequence")
    assert captured["kwargs"]["image_paths"] is image_paths
