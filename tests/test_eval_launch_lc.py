from pathlib import Path
import sys
import types

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
    def __init__(self, delegate, config, loop_strategy):
        self.delegate = delegate
        self.pipeline_config = config
        self.loop_strategy = loop_strategy


def test_legacy_evaluation_builds_estimator_only_for_candidates(monkeypatch):
    config = load_pipeline_config("configs/pipeline/test.yaml").config
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
        lambda model, manifest, images, config: (object(),),
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
        "JointPi3AlignmentEstimator",
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
