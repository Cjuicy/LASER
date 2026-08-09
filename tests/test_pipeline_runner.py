from __future__ import annotations

from contextlib import contextmanager
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_engine.streaming_window_engine import (
    STOP_SIGNAL,
    StreamingWindowEngine,
)
from inference_engine.prediction_cache.types import WindowSpec
from inference_engine.models.lazy import ModelExecutionStats
from inference_engine.prediction_cache.fingerprint import (
    PredictionFingerprint,
)
from inference_engine.prediction_cache.store import PredictionStoreStats
from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopCandidate,
    LoopSolution,
    ReconstructionResult,
    WindowCache,
)
from pipeline.config import LoopMethod, ModelName, load_pipeline_config
from pipeline.runner import (
    PipelineDependencies,
    PipelineRunner,
    StreamingPipelineModel,
    require_local_model_checkpoint,
    run_windows,
    run_from_config,
)
from pipeline.manifest import ImageManifest
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
    constraint_estimators: list[object] = field(default_factory=list)
    estimator_factory_kwargs: list[dict[str, object]] = field(
        default_factory=list
    )
    loaded_models: list[object] = field(default_factory=list)
    loaded_images: list[torch.Tensor] = field(default_factory=list)
    window_specs: list[tuple[WindowSpec, ...]] = field(
        default_factory=list
    )
    engine_dependencies: list[dict[str, object]] = field(
        default_factory=list
    )


class RecordingLoopStrategy:
    def __init__(self, method, state):
        self.name = method
        self.state = state

    def create_window_engine(self, **dependencies):
        self.state.calls.append("create_window_engine")
        self.state.engine_dependencies.append(dependencies)
        return SimpleNamespace(
            prediction_key="prediction-key",
            model_name=ModelName.PI3,
            checkpoint_digest="a" * 64,
        )

    def build_constraints(
        self,
        caches,
        candidates,
        *,
        constraint_estimator=None,
    ):
        self.state.calls.append("build_constraints")
        self.state.constraint_estimators.append(constraint_estimator)
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
        prediction_key="prediction-key",
        model_name=ModelName.PI3,
        checkpoint_digest="a" * 64,
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


def recording_dependencies(state, *, candidates=()):
    def preflight(config, manifest, cuda_available):
        state.calls.append("preflight")

    def build_model_handle(
        config,
        *,
        expected_checkpoint_sha256=None,
    ):
        state.calls.append("build_model_handle")
        assert expected_checkpoint_sha256 == "a" * 64
        handle = SimpleNamespace(stats=ModelExecutionStats())
        state.loaded_models.append(handle)
        return handle

    def build_fingerprint(**kwargs):
        state.calls.append("build_fingerprint")
        return PredictionFingerprint(
            key="prediction-key",
            checkpoint_sha256="a" * 64,
            image_manifest_sha256="b" * 64,
            runtime_source_sha256="c" * 64,
            canonical_payload={"model_name": "pi3"},
        )

    def build_store(**kwargs):
        state.calls.append("build_prediction_store")

        @contextmanager
        def entry_lock():
            state.calls.append("prediction_entry_lock_enter")
            try:
                yield
            finally:
                state.calls.append("prediction_entry_lock_exit")

        return SimpleNamespace(
            stats=PredictionStoreStats(),
            mode=kwargs["mode"],
            entry_path=Path(kwargs["root"]) / "v2" / "prediction-key",
            entry_lock=entry_lock,
        )

    def build_provider(**kwargs):
        state.calls.append("build_prediction_provider")
        return SimpleNamespace(**kwargs)

    def build_segmenter(config):
        state.segmentation_calls.append(config.method.value)
        return object()

    def load_images(manifest):
        images = torch.zeros((len(manifest), 3, 2, 2))
        state.loaded_images.append(images)
        return images

    def build_loop(method, **dependencies):
        state.loop_calls.append(method.value)
        return RecordingLoopStrategy(method, state)

    def run_windows(engine, manifest, images, specs, config):
        state.inference_manifests.append(manifest)
        state.window_specs.append(tuple(specs))
        assert images.shape[0] == len(manifest)
        return (_cache(config.loop.method, len(manifest)),)

    def detect_candidates(config, manifest, output_path):
        state.salad_manifests.append(manifest)
        return candidates

    def build_estimator(**kwargs):
        state.estimator_factory_kwargs.append(kwargs)
        return object()

    def save_result(payload, scene_name, result_dir, inverse_extrinsic):
        state.calls.append("save_result")
        assert inverse_extrinsic is False

    return PipelineDependencies(
        validate_preflight=preflight,
        build_model_handle=build_model_handle,
        build_prediction_fingerprint=build_fingerprint,
        build_prediction_store=build_store,
        build_prediction_provider=build_provider,
        load_images=load_images,
        build_segmentation_strategy=build_segmenter,
        build_loop_strategy=build_loop,
        run_windows=run_windows,
        detect_loop_candidates=detect_candidates,
        build_constraint_estimator=build_estimator,
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


def test_preflight_runs_before_prediction_fingerprint_and_model_handle(
    tmp_path,
):
    state = RecordingState()
    config_path, overrides, _ = _pipeline_args(tmp_path)
    run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )
    assert state.calls.index("preflight") < state.calls.index(
        "build_fingerprint"
    )
    assert state.calls.index("preflight") < state.calls.index(
        "build_model_handle"
    )


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
    assert "build_model_handle" not in state.calls
    assert "build_fingerprint" not in state.calls


def test_same_manifest_instance_reaches_inference_and_salad(tmp_path):
    state = RecordingState()
    config_path, overrides, _ = _pipeline_args(tmp_path)
    run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )
    assert state.inference_manifests[0] is state.salad_manifests[0]
    assert state.window_specs[0] == (WindowSpec(0, 0, 3),)


def test_no_loop_candidates_skip_constraint_model_and_optimizer(tmp_path):
    state = RecordingState()
    config_path, overrides, _ = _pipeline_args(tmp_path)
    result = run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )
    assert state.estimator_factory_kwargs == []
    assert state.constraint_model_calls == 0
    assert state.optimizer_calls == 0
    assert result.summary["used_no_loop_path"] is True
    assert state.loaded_models[0].stats.model_constructed is False


def test_runner_builds_estimator_with_active_pipeline_inputs(tmp_path):
    state = RecordingState()
    sentinel_estimator = object()

    def build_estimator(**kwargs):
        state.estimator_factory_kwargs.append(kwargs)
        return sentinel_estimator

    dependencies = recording_dependencies(
        state,
        candidates=(LoopCandidate(frame_a=2, frame_b=0, similarity=0.9),),
    )
    dependencies = PipelineDependencies(
        **{
            **dependencies.__dict__,
            "build_constraint_estimator": build_estimator,
        }
    )
    config_path, overrides, _ = _pipeline_args(tmp_path)

    run_from_config(config_path, overrides, dependencies=dependencies)

    kwargs = state.estimator_factory_kwargs[0]
    assert kwargs["model"] is state.loaded_models[0]
    assert kwargs["images"] is state.loaded_images[0]
    assert kwargs["manifest"] is state.inference_manifests[0]
    assert kwargs["chunk_size"] == 20
    assert kwargs["confidence_keep_ratio"] == pytest.approx(0.30)
    assert state.constraint_estimators == [sentinel_estimator]


def test_runner_builds_provider_before_selected_window_engine(tmp_path):
    state = RecordingState()
    config_path, overrides, _ = _pipeline_args(tmp_path)

    run_from_config(
        config_path,
        overrides,
        dependencies=recording_dependencies(state),
    )

    assert state.calls.index("build_prediction_provider") < state.calls.index(
        "create_window_engine"
    )
    dependencies = state.engine_dependencies[0]
    assert dependencies["delegate"].model is state.loaded_models[0]
    assert dependencies["prediction_key"] == "prediction-key"
    assert dependencies["model_name"] is ModelName.PI3
    assert dependencies["checkpoint_digest"] == "a" * 64


def test_runner_holds_prediction_entry_lock_while_running_windows(tmp_path):
    state = RecordingState()
    dependencies = recording_dependencies(state)
    original_run_windows = dependencies.run_windows

    def assert_locked(engine, manifest, images, specs, config):
        assert state.calls[-1] == "prediction_entry_lock_enter"
        state.calls.append("run_windows")
        return original_run_windows(
            engine,
            manifest,
            images,
            specs,
            config,
        )

    dependencies = PipelineDependencies(
        **{
            **dependencies.__dict__,
            "run_windows": assert_locked,
        }
    )
    config_path, overrides, _ = _pipeline_args(tmp_path)

    run_from_config(
        config_path,
        overrides,
        dependencies=dependencies,
    )

    assert state.calls.index("prediction_entry_lock_enter") < (
        state.calls.index("run_windows")
    )
    assert state.calls.index("run_windows") < state.calls.index(
        "prediction_entry_lock_exit"
    )


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
    assert summary["model_name"] == "pi3"
    assert summary["ordinary_prediction_key"] == "prediction-key"
    assert summary["prediction_cache_mode"] == "auto"
    assert summary["model_constructed"] is False
    assert summary["ordinary_forward_count"] == 0
    assert summary["joint_forward_count"] == 0
    assert summary["prediction_cache_events"] == []
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


def test_streaming_checkpoint_rejects_hugging_face_repository_id():
    with pytest.raises(FileNotFoundError, match="local checkpoint"):
        require_local_model_checkpoint("yyfz233/Pi3")


def test_run_windows_rejects_image_manifest_mismatch_before_begin(tmp_path):
    class MustNotBegin:
        def begin(self):
            raise AssertionError("engine must not begin")

    manifest = ImageManifest(
        paths=tuple(tmp_path / f"{index}.png" for index in range(3))
    )
    config_path, overrides, _ = _pipeline_args(tmp_path)
    config = load_pipeline_config(config_path, overrides).config

    with pytest.raises(
        ValueError,
        match="images must have shape",
    ):
        run_windows(
            MustNotBegin(),
            manifest,
            torch.zeros((2, 3, 2, 2)),
            (WindowSpec(0, 0, 3),),
            config,
        )


class RaisingDelegate(torch.nn.Module):
    def get(self, spec, sample):
        raise ValueError("delegate exploded")


class FastDelegate(torch.nn.Module):
    def get(self, spec, sample):
        frames, _, height, width = sample.shape
        return {
            "local_points": torch.zeros(
                (1, frames, height, width, 3)
            ),
            "camera_poses": torch.eye(4).repeat(1, frames, 1, 1),
            "conf": torch.ones((1, frames, height, width)),
            "images": sample.unsqueeze(0),
        }


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
    engine(
        torch.zeros((2, 3, 2, 2)),
        window_spec=WindowSpec(0, 0, 2),
    )
    with pytest.raises(RuntimeError, match="model inference worker"):
        engine.end()
    assert engine.running is False


def test_slow_registration_applies_bounded_backpressure(tmp_path):
    engine = StreamingWindowEngine(
        FastDelegate(),
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
    release_registration = threading.Event()

    def slow_registration():
        release_registration.wait()
        while True:
            item = engine.registration_queue.get()
            if item is STOP_SIGNAL:
                return

    engine._registration_worker = slow_registration
    errors = []

    def execute():
        try:
            engine.begin()
            for index in range(12):
                engine(
                    torch.zeros((2, 3, 2, 2)),
                    window_spec=WindowSpec(index, index, index + 2),
                )
            engine.end()
        except BaseException as error:
            errors.append(error)

    execution = threading.Thread(target=execute)
    execution.start()
    deadline = time.monotonic() + 2.0
    while (
        engine.registration_queue.qsize() < 5
        and execution.is_alive()
        and time.monotonic() < deadline
    ):
        threading.Event().wait(0.01)

    observed_maxsize = engine.registration_queue.maxsize
    observed_depth = engine.registration_queue.qsize()
    observed_running = execution.is_alive()
    release_registration.set()
    execution.join(timeout=2.0)

    assert observed_maxsize == 4
    assert observed_depth <= 4
    assert observed_running
    assert not execution.is_alive()
    assert errors == []


def test_registration_failure_cancels_blocked_producer(tmp_path):
    engine = StreamingWindowEngine(
        FastDelegate(),
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

    def fail_registration():
        engine.registration_queue.get()
        raise ValueError("registration exploded")

    engine._registration_worker = fail_registration
    engine.begin()
    forward_error = None
    end_error = None
    try:
        for index in range(20):
            engine(
                torch.zeros((2, 3, 2, 2)),
                window_spec=WindowSpec(index, index, index + 2),
            )
    except RuntimeError as error:
        forward_error = error
    try:
        engine.end()
    except RuntimeError as error:
        end_error = error

    assert "registration worker" in str(forward_error)
    assert "registration worker" in str(end_error)
    assert engine.running is False


def test_streaming_model_rebuilds_lazy_handle_after_checkpoint_change(
    tmp_path,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_paths = []
    for index in range(3):
        path = image_dir / f"frame-{index}.png"
        path.write_bytes(f"frame-{index}".encode("ascii"))
        image_paths.append(path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-one")
    loaded = load_pipeline_config("configs/pipeline/test.yaml")
    config = replace(
        loaded.config,
        input=replace(
            loaded.config.input,
            image_dir=str(image_dir),
            sample_stride=1,
        ),
        model=replace(
            loaded.config.model,
            checkpoint=str(checkpoint),
            inference_device="cpu",
            process_device="cpu",
            dtype="float32",
        ),
        prediction_cache=replace(
            loaded.config.prediction_cache,
            root=str(tmp_path / "predictions"),
        ),
        output=replace(
            loaded.config.output,
            cache_dir=str(tmp_path / "method-cache"),
        ),
        window=replace(loaded.config.window, size=3, overlap=1),
    )
    manifest = ImageManifest(paths=tuple(image_paths))
    images = torch.zeros((3, 3, 2, 2))
    model = StreamingPipelineModel(config)

    first = model.prepare(images, manifest)
    first_handle = first.model_handle
    second = model.prepare(images, manifest)
    assert second.model_handle is first_handle

    checkpoint.write_bytes(b"checkpoint-two")
    third = model.prepare(images, manifest)
    assert third.model_handle is not first_handle
    fourth = model.prepare(images, manifest)
    assert fourth.model_handle is third.model_handle
