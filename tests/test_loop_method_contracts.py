from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopCandidate,
    LoopConstraint,
    LoopSolution,
    WindowCache,
)
from loop_closure.methods.corrected import CorrectedLoopClosureStrategy
from loop_closure.methods.traditional import TraditionalLoopClosureStrategy
from loop_closure.loop_model import LoopDetector
from loop_closure.utils.sim3loop import Sim3LoopOptimizer
from pipeline.config import LoopMethod, load_pipeline_config
from pipeline.manifest import ImageManifest


def _identity_sim3(scale=1.0):
    return scale, torch.eye(3), torch.zeros(3)


def window_cache_fixture(loop_method=LoopMethod.TRADITIONAL):
    return WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=loop_method,
        window_index=0,
        frame_start=0,
        frame_end=2,
        local_points=torch.zeros((2, 2, 3, 3)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        confidence=torch.ones((2, 2, 3)),
        segmentation_labels=(
            np.zeros((2, 3), dtype=np.intp),
            np.zeros((2, 3), dtype=np.intp),
        ),
        anchor_scale_mask=None,
        loop_state={"tag": loop_method.value},
    )


def test_loop_candidate_is_immutable_and_scored():
    candidate = LoopCandidate(frame_a=90, frame_b=10, similarity=0.81)
    assert candidate.frame_a == 90
    with pytest.raises(FrozenInstanceError):
        candidate.similarity = 0.5


@pytest.mark.parametrize(
    "kwargs",
    (
        {"frame_a": -1, "frame_b": 0, "similarity": 0.8},
        {"frame_a": 1, "frame_b": 1, "similarity": 0.8},
        {"frame_a": 2, "frame_b": 1, "similarity": float("nan")},
    ),
)
def test_loop_candidate_rejects_invalid_frame_range_or_score(kwargs):
    with pytest.raises(ValueError):
        LoopCandidate(**kwargs)


def test_cache_rejects_cross_method_loading():
    cache = window_cache_fixture(loop_method=LoopMethod.TRADITIONAL)
    with pytest.raises(ValueError, match="traditional.*corrected"):
        WindowCache.from_payload(
            cache.to_payload(),
            expected_method=LoopMethod.CORRECTED,
        )


def test_cache_rejects_schema_version_and_state_tag():
    cache = window_cache_fixture()
    wrong_version = cache.to_payload()
    wrong_version["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        WindowCache.from_payload(
            wrong_version,
            expected_method=LoopMethod.TRADITIONAL,
        )

    wrong_tag = cache.to_payload()
    wrong_tag["loop_state"] = {"tag": "corrected"}
    with pytest.raises(ValueError, match="state tag"):
        WindowCache.from_payload(
            wrong_tag,
            expected_method=LoopMethod.TRADITIONAL,
        )


def test_cache_rejects_invalid_frame_span():
    payload = window_cache_fixture().to_payload()
    payload["frame_end"] = payload["frame_start"]
    with pytest.raises(ValueError, match="frame"):
        WindowCache.from_payload(
            payload,
            expected_method=LoopMethod.TRADITIONAL,
        )


@pytest.mark.parametrize("scale", (0.0, -1.0, float("nan")))
def test_constraint_rejects_non_positive_or_nonfinite_sim3(scale):
    candidate = LoopCandidate(90, 10, 0.81)
    with pytest.raises(ValueError, match="scale"):
        LoopConstraint(
            window_a=2,
            window_b=0,
            measurement=_identity_sim3(scale),
            candidate=candidate,
        )


def test_solution_rejects_nonfinite_sim3_components():
    with pytest.raises(ValueError, match="finite"):
        LoopSolution(
            optimized_transforms=(
                (
                    1.0,
                    torch.eye(3),
                    torch.tensor([float("inf"), 0.0, 0.0]),
                ),
            ),
            constraints=(),
            used_no_loop_path=True,
        )


class RecordingStrategy:
    def __init__(self, name):
        self.name = name
        self.received_candidates = None

    def build_constraints(
        self,
        caches,
        candidates,
        *,
        constraint_estimator=None,
    ):
        self.received_candidates = candidates
        return []


def test_both_methods_receive_same_candidate_tuple():
    candidates = (
        LoopCandidate(frame_a=90, frame_b=10, similarity=0.81),
    )
    traditional = RecordingStrategy(LoopMethod.TRADITIONAL)
    corrected = RecordingStrategy(LoopMethod.CORRECTED)
    traditional.build_constraints([], candidates)
    corrected.build_constraints([], candidates)
    assert traditional.received_candidates is candidates
    assert corrected.received_candidates is candidates


def _strategy_and_caches(loop_method, constraint_estimator):
    optimizer_config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer
    strategy_type = {
        LoopMethod.TRADITIONAL: TraditionalLoopClosureStrategy,
        LoopMethod.CORRECTED: CorrectedLoopClosureStrategy,
    }[loop_method]
    state = (
        {
            "tag": loop_method.value,
            "relative_sim3": _identity_sim3(),
            "anchor_scale_applied": False,
        }
        if loop_method is LoopMethod.TRADITIONAL
        else {
            "tag": loop_method.value,
            "sim3_abs": _identity_sim3(),
            "sim3_edge": _identity_sim3(),
            "anchor_scale_applied": True,
        }
    )
    caches = tuple(
        WindowCache(
            schema_version=WINDOW_CACHE_SCHEMA_VERSION,
            loop_method=loop_method,
            window_index=index,
            frame_start=index * 2,
            frame_end=index * 2 + 2,
            local_points=torch.ones((2, 1, 1, 3)),
            camera_poses=torch.eye(4).repeat(2, 1, 1),
            confidence=torch.ones((2, 1, 1)),
            segmentation_labels=(
                np.zeros((1, 1), dtype=np.intp),
                np.zeros((1, 1), dtype=np.intp),
            ),
            anchor_scale_mask=None,
            loop_state=state,
        )
        for index in range(3)
    )
    return strategy_type(
        optimizer_config=optimizer_config,
        registration_confidence_keep_ratio=0.3,
        constraint_estimator=constraint_estimator,
    ), caches


@pytest.mark.parametrize("loop_method", tuple(LoopMethod))
def test_loop_constraint_value_error_isolated_and_pair_deduplicated(
    loop_method,
):
    calls = []

    def estimate(*arguments):
        calls.append(arguments[2])
        if len(calls) == 1:
            raise ValueError("no mutual confidence")
        return _identity_sim3(), _identity_sim3()

    strategy, caches = _strategy_and_caches(loop_method, estimate)
    first = LoopCandidate(frame_a=2, frame_b=0, similarity=0.8)
    second = LoopCandidate(frame_a=4, frame_b=0, similarity=0.7)
    third = LoopCandidate(frame_a=3, frame_b=1, similarity=0.6)

    constraints = strategy.build_constraints(caches, (first, second, third))

    assert [constraint.candidate for constraint in constraints] == [second]
    assert calls == [first, second]


@pytest.mark.parametrize("loop_method", tuple(LoopMethod))
def test_loop_constraint_system_error_escapes_candidate_isolation(loop_method):
    def estimate(*arguments):
        raise RuntimeError("CUDA out of memory")

    strategy, caches = _strategy_and_caches(loop_method, estimate)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        strategy.build_constraints(
            caches,
            (LoopCandidate(frame_a=2, frame_b=0, similarity=0.8),),
        )


def test_loop_detector_consumes_typed_config_and_explicit_manifest(tmp_path):
    config = load_pipeline_config(
        "configs/pipeline/test.yaml",
    ).config.loop.detection
    paths = (
        (tmp_path / "frame1.png").resolve(),
        (tmp_path / "frame2.png").resolve(),
    )
    manifest = ImageManifest(paths=paths)
    detector = LoopDetector(
        detection_config=config,
        image_manifest=manifest,
        output_path=tmp_path / "loops.txt",
    )
    assert detector.image_paths == [str(path) for path in paths]
    assert detector.ckpt_path == config.salad_checkpoint


def test_sim3_optimizer_uses_typed_optimizer_config():
    config = load_pipeline_config(
        "configs/pipeline/test.yaml",
        (
            "loop.optimizer.implementation=python",
            "loop.optimizer.max_iterations=17",
            "loop.optimizer.initial_damping=0.0002",
        ),
    ).config.loop.optimizer
    optimizer = Sim3LoopOptimizer(config, device="cpu")
    assert optimizer.solve_system_version == "python"
    assert optimizer.max_iterations == 17
    assert optimizer.initial_damping == pytest.approx(0.0002)
