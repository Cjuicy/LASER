from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from loop_closure.methods.base import LoopCandidate, ReconstructionResult
from mv_recon.loop_experiment import (
    LoopExperimentDependencies,
    reconstruct_traditional_loop_point_maps,
)
from pipeline.config import LoopMethod
from pipeline.manifest import ImageManifest


@dataclass
class CallState:
    stages: list[str] = field(default_factory=list)
    constraint_estimator: object | None = None


class FakeStore:
    fingerprint = SimpleNamespace(key="f" * 64)

    @contextmanager
    def entry_lock(self):
        yield


class FakeStrategy:
    def __init__(self, state: CallState, *, no_loop: bool = False):
        self.state = state
        self.no_loop = no_loop

    def build_constraints(
        self,
        caches,
        candidates,
        *,
        constraint_estimator=None,
    ):
        assert caches == ("cache-0", "cache-1")
        self.state.stages.append("constraints")
        self.state.constraint_estimator = constraint_estimator
        return [] if not candidates else ["constraint"]

    def optimize(self, caches, constraints):
        assert caches == ("cache-0", "cache-1")
        self.state.stages.append("optimize")
        return SimpleNamespace(
            used_no_loop_path=not bool(constraints),
            constraints=tuple(constraints),
        )

    def aggregate(self, caches, solution):
        assert caches == ("cache-0", "cache-1")
        self.state.stages.append("aggregate")
        points = torch.zeros((3, 1, 1, 3))
        points[..., 2] = 1.0
        return ReconstructionResult(
            payload={
                "points": points,
                "confidence": torch.ones((3, 1, 1)),
            },
            summary={"used_no_loop_path": solution.used_no_loop_path},
        )


def _engine(state: CallState, *, enabled: bool = True, method=None):
    loop_method = method or LoopMethod.TRADITIONAL
    config = SimpleNamespace(
        loop=SimpleNamespace(
            enabled=enabled,
            method=loop_method,
            detection=SimpleNamespace(method="salad"),
            constraint=SimpleNamespace(chunk_size=20),
            registration=SimpleNamespace(confidence_keep_ratio=0.5),
        ),
        model=SimpleNamespace(name=SimpleNamespace(value="pi3")),
        prediction_cache=SimpleNamespace(mode=SimpleNamespace(value="auto")),
    )
    return SimpleNamespace(
        pipeline_config=config,
        prediction_store=FakeStore(),
        window_specs=("spec-0", "spec-1"),
        loop_strategy=FakeStrategy(state),
        model_handle="model-handle",
    )


def _manifest(tmp_path: Path, count: int = 3) -> ImageManifest:
    paths = []
    for index in range(count):
        path = tmp_path / f"frame-{index}.png"
        path.write_bytes(b"image")
        paths.append(path.resolve())
    return ImageManifest(paths=tuple(paths))


def _dependencies(state: CallState, *, candidates):
    def run_windows(engine, manifest, images, specs, config):
        del engine, manifest, images, specs, config
        state.stages.append("windows")
        return ("cache-0", "cache-1")

    def detect_loop_candidates(config, manifest, output_path):
        del config, manifest
        assert output_path.name == "loop_candidates.json"
        assert output_path.parent.is_dir()
        state.stages.append("candidates")
        return candidates

    def build_constraint_estimator(**kwargs):
        assert kwargs == {
            "model": "model-handle",
            "images": kwargs["images"],
            "manifest": kwargs["manifest"],
            "chunk_size": 20,
            "confidence_keep_ratio": 0.5,
        }
        state.stages.append("estimator")
        return "joint-estimator"

    return LoopExperimentDependencies(
        run_windows=run_windows,
        detect_loop_candidates=detect_loop_candidates,
        build_constraint_estimator=build_constraint_estimator,
        collect_prediction_diagnostics=lambda **kwargs: {
            "ordinary_hits": 2,
            "ordinary_misses": 0,
            "joint_forward_count": 1 if candidates else 0,
        },
    )


def test_traditional_loop_reconstruction_consumes_complete_loop_flow(tmp_path):
    state = CallState()
    candidate = LoopCandidate(frame_a=2, frame_b=0, similarity=0.9)

    result = reconstruct_traditional_loop_point_maps(
        engine=_engine(state),
        images=torch.zeros((3, 3, 1, 1)),
        manifest=_manifest(tmp_path),
        artifact_dir=tmp_path / "loop-artifacts",
        dependencies=_dependencies(state, candidates=(candidate,)),
    )

    assert state.stages == [
        "windows",
        "candidates",
        "estimator",
        "constraints",
        "optimize",
        "aggregate",
    ]
    assert state.constraint_estimator == "joint-estimator"
    assert result.points.shape == (3, 1, 1, 3)
    assert result.confidence.shape == (3, 1, 1)
    assert result.ordinary_prediction_key == "f" * 64
    assert result.diagnostics["candidate_count"] == 1
    assert result.diagnostics["constraint_count"] == 1
    assert result.diagnostics["rejected_candidate_count"] == 0
    assert result.diagnostics["used_no_loop_path"] is False
    assert result.diagnostics["joint_forward_count"] == 1


def test_zero_candidates_use_traditional_no_loop_fallback(tmp_path):
    state = CallState()
    dependencies = _dependencies(state, candidates=())
    dependencies = LoopExperimentDependencies(
        run_windows=dependencies.run_windows,
        detect_loop_candidates=dependencies.detect_loop_candidates,
        build_constraint_estimator=lambda **kwargs: pytest.fail(
            "zero candidates must not construct a joint estimator"
        ),
        collect_prediction_diagnostics=(
            dependencies.collect_prediction_diagnostics
        ),
    )

    result = reconstruct_traditional_loop_point_maps(
        engine=_engine(state),
        images=torch.zeros((3, 3, 1, 1)),
        manifest=_manifest(tmp_path),
        artifact_dir=tmp_path / "loop-artifacts",
        dependencies=dependencies,
    )

    assert state.stages == [
        "windows",
        "candidates",
        "constraints",
        "optimize",
        "aggregate",
    ]
    assert result.diagnostics["candidate_count"] == 0
    assert result.diagnostics["constraint_count"] == 0
    assert result.diagnostics["rejected_candidate_count"] == 0
    assert result.diagnostics["used_no_loop_path"] is True


@pytest.mark.parametrize(
    ("enabled", "method", "message"),
    (
        (False, LoopMethod.TRADITIONAL, "enabled"),
        (True, LoopMethod.CORRECTED, "traditional"),
    ),
)
def test_loop_adapter_rejects_wrong_loop_configuration(
    tmp_path, enabled, method, message
):
    with pytest.raises(ValueError, match=message):
        reconstruct_traditional_loop_point_maps(
            engine=_engine(CallState(), enabled=enabled, method=method),
            images=torch.zeros((3, 3, 1, 1)),
            manifest=_manifest(tmp_path),
            artifact_dir=tmp_path / "loop-artifacts",
        )


@pytest.mark.parametrize("failure", ("missing", "shape", "nonfinite"))
def test_loop_adapter_rejects_invalid_aggregate_payload(tmp_path, failure):
    state = CallState()
    engine = _engine(state)

    def invalid_aggregate(caches, solution):
        del caches, solution
        points = torch.zeros((2 if failure == "shape" else 3, 1, 1, 3))
        if failure == "nonfinite":
            points[0, 0, 0, 0] = float("nan")
        payload = {"points": points, "confidence": torch.ones((3, 1, 1))}
        if failure == "missing":
            payload.pop("confidence")
        return ReconstructionResult(payload=payload, summary={})

    engine.loop_strategy.aggregate = invalid_aggregate

    with pytest.raises(ValueError, match="aggregate|finite|frame"):
        reconstruct_traditional_loop_point_maps(
            engine=engine,
            images=torch.zeros((3, 3, 1, 1)),
            manifest=_manifest(tmp_path),
            artifact_dir=tmp_path / "loop-artifacts",
            dependencies=_dependencies(state, candidates=()),
        )
