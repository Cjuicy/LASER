import ast
from dataclasses import replace
from pathlib import Path
import sys
import types

from omegaconf import OmegaConf
import pytest
import torch

pose_eval = types.ModuleType("eval.pose_eval")
pose_eval.eval_pose_estimation = lambda *args, **kwargs: None
depth_eval = types.ModuleType("eval.depth_eval")
depth_eval.eval_mono_depth_estimation = lambda *args, **kwargs: None
sys.modules.setdefault("eval.pose_eval", pose_eval)
sys.modules.setdefault("eval.depth_eval", depth_eval)

import eval_launch
from loop_closure.methods.base import LoopCandidate, ReconstructionResult
from pipeline.config import load_pipeline_config
from pipeline.manifest import ImageManifest


class RecordingLoopStrategy:
    def __init__(self):
        self.constraint_estimators = []

    def build_constraints(
        self,
        caches,
        candidates,
        *,
        constraint_estimator=None,
    ):
        self.constraint_estimators.append(constraint_estimator)
        return []

    def optimize(self, caches, constraints):
        return object()

    def aggregate(self, caches, solution):
        frames = 2
        local_points = torch.zeros((frames, 1, 1, 3))
        local_points[..., 2] = 1.0
        return ReconstructionResult(
            payload={
                "local_points": local_points,
                "camera_poses": torch.eye(4).repeat(frames, 1, 1),
                "confidence": torch.ones((frames, 1, 1)),
            },
            summary={},
        )


class FakeEngine:
    def __init__(self, model_handle, config, loop_strategy):
        self.model_handle = model_handle
        self.pipeline_config = config
        self.loop_strategy = loop_strategy


def _load_pose_function(name):
    source_path = Path(__file__).resolve().parents[1] / "eval/pose_eval.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def test_pose_evaluation_passes_exact_paths_to_both_streaming_modes():
    dispatch = _load_pose_function("run_model_inference")
    images = object()
    image_paths = ["frame-0.png", "frame-1.png"]

    for model_name in ("streaming_pi3", "streaming_pi3_lc"):
        calls = []

        def model(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

        dispatch(
            types.SimpleNamespace(model=model_name),
            model,
            images,
            "/frames",
            image_paths,
        )

        assert calls == [
            ((images, "/frames"), {"image_paths": image_paths})
        ]


def test_pose_evaluation_keeps_streaming_images_on_cpu():
    source_path = Path(__file__).resolve().parents[1] / "eval/pose_eval.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_model_images"
    )
    transfers = []
    moved = object()

    class Images:
        def to(self, device):
            transfers.append(device)
            return moved

    images = Images()
    namespace = {
        "load_and_preprocess_images": lambda paths: images,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    load_images = namespace["load_model_images"]

    for model_name in ("streaming_pi3", "streaming_pi3_lc"):
        assert load_images(model_name, ["frame.png"], "cuda") is images
        assert transfers == []

    assert load_images("pi3", ["frame.png"], "cuda") is moved
    assert transfers == ["cuda"]


@pytest.mark.parametrize(
    "config_name",
    ("mv_recon_dense.yaml", "mv_recon_kf15.yaml", "mv_recon_outdoor.yaml"),
)
def test_shipped_mv_streaming_configs_define_local_checkpoint(config_name):
    repository_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(
        repository_root / "configs/evaluation" / config_name
    )
    assert config.pi3.checkpoint == "weights/model.safetensors"

    for relative_path in ("mv_recon/eval.py", "mv_recon/eval_outdoor.py"):
        source = (repository_root / relative_path).read_text(
            encoding="utf-8"
        )
        assert "cfg.pi3.checkpoint" in source


def test_eval_launch_does_not_import_pi3_directly():
    source = Path(eval_launch.__file__).read_text(encoding="utf-8")
    assert "from pi3.models.pi3 import Pi3" not in source


def test_legacy_streaming_entrypoints_do_not_use_obsolete_signatures():
    repository_root = Path(__file__).resolve().parents[1]
    sources = {
        relative_path: (
            repository_root / relative_path
        ).read_text(encoding="utf-8")
        for relative_path in (
            "mv_recon/eval.py",
            "mv_recon/eval_outdoor.py",
            "utils/interfaces.py",
        )
    }
    assert "build_default_window_engine(config, pi3)" not in (
        sources["mv_recon/eval.py"]
    )
    assert "build_default_window_engine(config, pi3)" not in (
        sources["mv_recon/eval_outdoor.py"]
    )
    assert (
        "run_windows(inference_engine, manifest, imgs, config)"
        not in sources["utils/interfaces.py"]
    )


def test_baseline_evaluation_uses_model_directly():
    model = object()
    assert eval_launch._select_inference_function("pi3", model) is model


def test_legacy_evaluation_builds_estimator_only_for_candidates(monkeypatch):
    config = load_pipeline_config("configs/pipeline/test.yaml").config
    config = replace(
        config,
        window=replace(config.window, size=2, overlap=1),
    )
    manifest = ImageManifest(
        paths=(Path("frame_00000000.png"), Path("frame_00000001.png"))
    )
    images = torch.zeros((2, 3, 2, 2))
    sentinel_estimator = object()
    estimator_instances = []

    def build_estimator(**kwargs):
        estimator_instances.append(sentinel_estimator)
        return sentinel_estimator

    monkeypatch.setattr(
        eval_launch,
        "run_windows",
        lambda model, manifest, images, specs, config: (object(),),
    )
    monkeypatch.setattr(
        eval_launch,
        "detect_loop_candidates",
        lambda config, manifest, output_path: (
            LoopCandidate(frame_a=1, frame_b=0, similarity=0.9),
        ),
    )
    monkeypatch.setattr(
        eval_launch,
        "JointAlignmentEstimator",
        build_estimator,
    )

    loop_strategy = RecordingLoopStrategy()
    model = FakeEngine(object(), config, loop_strategy)
    eval_launch._run_modular_evaluation(
        model,
        images,
        manifest,
        detect_loops=True,
    )

    assert estimator_instances == [sentinel_estimator]
    assert loop_strategy.constraint_estimators == [sentinel_estimator]

    no_loop_strategy = RecordingLoopStrategy()
    no_loop_model = FakeEngine(object(), config, no_loop_strategy)
    eval_launch._run_modular_evaluation(
        no_loop_model,
        images,
        manifest,
        detect_loops=False,
    )

    assert estimator_instances == [sentinel_estimator]
    assert no_loop_strategy.constraint_estimators == [None]
