from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_engine.streaming_window_engine import StreamingWindowEngine
from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopSolution,
    ReconstructionResult,
    WindowCache,
)
from pipeline.config import LoopMethod, load_pipeline_config
from pipeline.runner import (
    PipelineDependencies,
    PipelineRunner,
    run_from_config,
)
from run_laser import build_parser


IDENTITY_SIM3 = (1.0, torch.eye(3), torch.zeros(3))


@dataclass
class RecordingState:
    calls: list[str] = field(default_factory=list)
    segmentation_calls: list[str] = field(default_factory=list)
    loop_calls: list[str] = field(default_factory=list)
    inference_manifests: list[object] = field(default_factory=list)
    salad_manifests: list[object] = field(default_factory=list)
    optimizer_calls: int = 0
    constraint_model_calls: int = 0


class RecordingLoopStrategy:
    def __init__(self, method, state):
        self.name = method
        self.state = state

    def create_window_engine(self, **dependencies):
        self.state.calls.append("create_window_engine")
        return object()

    def build_constraints(self, caches, candidates):
        self.state.calls.append("build_constraints")
        if candidates:
            self.state.constraint_model_calls += 1
        return []

    def optimize(self, caches, constraints):
        self.state.calls.append("optimize")
        if constraints:
            self.state.optimizer_calls += 1
        return LoopSolution(
            optimized_transforms=tuple(IDENTITY_SIM3 for _ in caches),
            constraints=(),
            used_no_loop_path=True,
        )

    def aggregate(self, caches, solution):
        self.state.calls.append("aggregate")
        frame_count = caches[-1].frame_end
        local_points = torch.zeros((frame_count, 2, 2, 3))
        local_points[..., 2] = 1.0
        camera_poses = torch.eye(4).repeat(frame_count, 1, 1)
        confidence = torch.ones((frame_count, 2, 2))
        points = local_points.clone()
        return ReconstructionResult(
            payload={
                "local_points": local_points,
                "camera_poses": camera_poses,
                "confidence": confidence,
                "points": points,
            },
            summary={
                "loop_method": self.name.value,
                "window_count": len(caches),
                "constraint_count": 0,
                "used_no_loop_path": True,
            },
        )


def _cache(method, frame_count=3):
    loop_state = {"tag": method.value}
    if method is LoopMethod.TRADITIONAL:
        loop_state.update(
            relative_sim3=IDENTITY_SIM3,
            anchor_scale_applied=False,
        )
    else:
        loop_state.update(
            sim3_abs=IDENTITY_SIM3,
            anchor_scale_applied=True,
        )
    return WindowCache(
        schema_version=WINDOW_CACHE_SCHEMA_VERSION,
        loop_method=method,
        window_index=0,
        frame_start=0,
        frame_end=frame_count,
        local_points=torch.zeros((frame_count, 2, 2, 3)),
        camera_poses=torch.eye(4).repeat(frame_count, 1, 1),
        confidence=torch.ones((frame_count, 2, 2)),
        segmentation_labels=tuple(
            np.zeros((2, 2), dtype=np.intp) for _ in range(frame_count)
        ),
        anchor_scale_mask=None,
        loop_state=loop_state,
        segmentation_diagnostics=tuple(
            {"method": "test", "region_count": 1}
            for _ in range(frame_count)
        ),
    )


def recording_dependencies(state):
    def preflight(config, manifest, cuda_available):
        state.calls.append("preflight")

    def load_pi3(config):
        state.calls.append("load_pi3")
        return object()

    def build_segmenter(config):
        state.segmentation_calls.append(config.method.value)
        return object()

    def load_images(manifest):
        return torch.zeros((len(manifest), 3, 2, 2))

    def build_loop(method, **dependencies):
        state.loop_calls.append(method.value)
        return RecordingLoopStrategy(method, state)

    def run_windows(engine, manifest, images, config):
        state.inference_manifests.append(manifest)
        assert images.shape[0] == len(manifest)
        return (_cache(config.loop.method, len(manifest)),)

    def detect_candidates(config, manifest, output_path):
        state.salad_manifests.append(manifest)
        return ()

    def save_result(payload, scene_name, result_dir, inverse_extrinsic):
        state.calls.append("save_result")
        assert inverse_extrinsic is False

    return PipelineDependencies(
        validate_preflight=preflight,
        load_pi3=load_pi3,
        load_images=load_images,
        build_segmentation_strategy=build_segmenter,
        build_loop_strategy=build_loop,
        run_windows=run_windows,
        detect_loop_candidates=detect_candidates,
        save_for_viser=save_result,
        cuda_available=lambda: False,
        git_commit=lambda: "test-commit",
    )


def _pipeline_args(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(3):
        (image_dir / f"frame{index}.png").touch()
    result_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    overrides = (
        f"input.image_dir={image_dir}",
        "input.sample_stride=1",
        f"output.result_dir={result_dir}",
        f"output.cache_dir={cache_dir}",
        "output.scene_name=test_scene",
        "model.inference_device=cpu",
        "model.process_device=cpu",
        "model.dtype=float32",
        "window.size=3",
        "window.overlap=1",
    )
    return Path("configs/pipeline/test.yaml"), overrides, result_dir


@pytest.mark.parametrize(
    ("segmentation_method", "loop_method"),
    (
        ("depth", "traditional"),
        ("depth", "corrected"),
        ("geometry", "traditional"),
        ("geometry", "corrected"),
        ("atomic", "traditional"),
        ("atomic", "corrected"),
    ),
)
def test_runner_selects_requested_strategies(
    tmp_path,
    segmentation_method,
    loop_method,
):
    state = RecordingState()
    config_path, base_overrides, _ = _pipeline_args(tmp_path)
    result = run_from_config(
        config_path,
        (
            *base_overrides,
            f"segmentation.method={segmentation_method}",
            f"loop.method={loop_method}",
        ),
        dependencies=recording_dependencies(state),
    )
    assert state.segmentation_calls == [segmentation_method]
    assert state.loop_calls == [loop_method]
    assert result.summary["segmentation_method"] == segmentation_method
    assert result.summary["loop_method"] == loop_method


def test_preflight_runs_before_model_loader(tmp_path):
    state = RecordingState()
    config_path, overrides, _ = _pipeline_args(tmp_path)
    run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )
    assert state.calls.index("preflight") < state.calls.index("load_pi3")


def test_preflight_failure_prevents_model_loading(tmp_path):
    state = RecordingState()
    dependencies = recording_dependencies(state)

    def fail_preflight(config, manifest, cuda_available):
        state.calls.append("preflight")
        raise RuntimeError("preflight stopped the run")

    dependencies = PipelineDependencies(
        **{
            **dependencies.__dict__,
            "validate_preflight": fail_preflight,
        }
    )
    config_path, overrides, _ = _pipeline_args(tmp_path)
    with pytest.raises(RuntimeError, match="preflight stopped"):
        run_from_config(
            config_path,
            overrides,
            dependencies=dependencies,
        )
    assert "load_pi3" not in state.calls


def test_same_manifest_instance_reaches_inference_and_salad(tmp_path):
    state = RecordingState()
    config_path, overrides, _ = _pipeline_args(tmp_path)
    run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )
    assert state.inference_manifests[0] is state.salad_manifests[0]


def test_no_loop_candidates_skip_constraint_model_and_optimizer(tmp_path):
    state = RecordingState()
    config_path, overrides, _ = _pipeline_args(tmp_path)
    result = run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )
    assert state.constraint_model_calls == 0
    assert state.optimizer_calls == 0
    assert result.summary["used_no_loop_path"] is True


def test_diagnostics_contain_resolved_config_hash(tmp_path):
    state = RecordingState()
    config_path, overrides, result_dir = _pipeline_args(tmp_path)
    loaded = load_pipeline_config(config_path, overrides)
    run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )
    output_root = result_dir / "test_scene"
    assert (
        (output_root / "resolved_config.yaml").read_text(encoding="utf-8")
        == loaded.resolved_yaml
    )
    summary = json.loads(
        (output_root / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["config_hash"] == loaded.sha256
    assert {
        "loop_candidates.json",
        "loop_constraints.json",
        "segmentation_diagnostics.json",
    } <= {path.name for path in output_root.iterdir()}


def test_cli_parser_rejects_legacy_method_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "configs/pipeline/default.yaml",
                "--segment_mode",
                "depth",
            ]
        )


class RaisingDelegate(torch.nn.Module):
    def forward(self, sample):
        raise ValueError("delegate exploded")


class UnusedSegmenter:
    def segment(self, point_maps, confidence, images):
        raise AssertionError("registration must not run")


def test_background_worker_exception_propagates_to_main_thread(tmp_path):
    engine = StreamingWindowEngine(
        RaisingDelegate(),
        inference_device="cpu",
        dtype=torch.float32,
        segmentation_strategy=UnusedSegmenter(),
        anchor_propagator=object(),
        registration_confidence_keep_ratio=0.5,
        anchor_enabled=False,
        temporal_iou_threshold=0.3,
        window_size=2,
        overlap=1,
        cache_root=str(tmp_path),
        intermediate_device="cpu",
        process_device="cpu",
        benchmark_latency=False,
    )
    engine.begin()
    engine(torch.zeros((2, 3, 2, 2)))
    with pytest.raises(RuntimeError, match="model inference worker"):
        engine.end()
    assert engine.running is False
