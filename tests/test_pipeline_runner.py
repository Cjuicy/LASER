from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

from inference_engine.models.lazy import ModelExecutionStats
from inference_engine.prediction_cache.fingerprint import PredictionFingerprint
from inference_engine.prediction_cache.store import PredictionStoreStats
from pipeline.artifacts import (
    ReconstructionArtifact,
    ReconstructionDiagnostics,
    write_reconstruction_artifact,
)
from pipeline.config import ReconstructionMode, SegmentationMethod, load_pipeline_config
from pipeline.manifest import ImageManifest
from pipeline.runner import (
    PipelineDependencies,
    PipelineRunner,
    _build_services,
    run_from_config,
)


@dataclass
class Events:
    values: list[str] = field(default_factory=list)


def _artifact(mode):
    local = torch.zeros((3, 2, 2, 3))
    poses = torch.eye(4).repeat(3, 1, 1)
    confidence = torch.ones((3, 2, 2))
    return ReconstructionArtifact(
        schema_version=1,
        frame_ids=(0, 1, 2),
        local_points=local,
        global_points=local.clone(),
        camera_poses=poses,
        confidence=confidence,
        segmentation_method=SegmentationMethod.DEPTH,
        reconstruction_mode=mode,
        prediction_key="prediction-key",
        diagnostics=ReconstructionDiagnostics({}, (), 0, 0, {}),
    )


def _config_args(tmp_path, mode="no_loop"):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(3):
        (image_dir / f"frame-{index}.png").touch()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    config = (
        "configs/reconstruction/pi3_laser_no_loop.yaml"
        if mode == "no_loop"
        else "configs/reconstruction/pi3_laser.yaml"
    )
    overrides = [
            f"input.image_dir={image_dir}",
            f"model.checkpoint={checkpoint}",
            f"output.result_dir={tmp_path / 'results'}",
            "output.scene_name=scene",
            "model.inference_device=cpu",
            "model.process_device=cpu",
            "model.dtype=float32",
            "window.size=3",
            "window.overlap=1",
            f"reconstruction.mode={mode}",
    ]
    if mode != "no_loop":
        overrides.append("loop.optimizer.implementation=python")
    return config, tuple(overrides)


def _loaded(tmp_path, mode="no_loop"):
    config, overrides = _config_args(tmp_path, mode)
    return load_pipeline_config(config, overrides)


def _dependencies(events, mode, *, refiner_builder=None, context_sink=None):
    def preflight(*arguments):
        events.values.append("preflight")

    def images(manifest):
        events.values.append("images")
        return torch.zeros((len(manifest), 3, 2, 2))

    def fingerprint(**arguments):
        events.values.append("fingerprint")
        return PredictionFingerprint(
            key="prediction-key",
            checkpoint_sha256="a" * 64,
            image_manifest_sha256="b" * 64,
            runtime_source_sha256="c" * 64,
            canonical_payload={},
        )

    def store(**arguments):
        events.values.append("store")

        @contextmanager
        def entry_lock():
            yield

        return SimpleNamespace(
            stats=PredictionStoreStats(),
            entry_lock=entry_lock,
        )

    def model(*arguments, **keywords):
        events.values.append("model")
        return SimpleNamespace(stats=ModelExecutionStats())

    def provider(**arguments):
        events.values.append("provider")
        return SimpleNamespace(prediction_key="prediction-key")

    def segmenter(config):
        events.values.append("segmenter")
        return SimpleNamespace(name=config.method)

    def anchor(*arguments):
        events.values.append("anchor")
        return object()

    class Mode:
        def run(self, context):
            events.values.append(mode)
            if context_sink is not None:
                context_sink(context)
            return _artifact(ReconstructionMode(mode))

    def build_mode(selected, services):
        assert selected is ReconstructionMode(mode)
        return Mode()

    def write_artifact(artifact, output_dir, **metadata):
        events.values.append("write_artifact")
        assert artifact.reconstruction_mode is ReconstructionMode(mode)
        assert metadata["config_sha256"]
        return Path(output_dir)

    def forbidden(*arguments, **keywords):
        raise AssertionError("no_loop constructed loop services")

    dependency_kwargs = dict(
        validate_preflight=preflight,
        load_images=images,
        build_prediction_fingerprint=fingerprint,
        build_prediction_store=store,
        build_model_handle=model,
        build_prediction_provider=provider,
        build_segmentation_strategy=segmenter,
        build_anchor_propagator=anchor,
        build_reconstruction_mode=build_mode,
        build_loop_detector=forbidden,
        build_loop_evidence=forbidden,
        write_artifact=write_artifact,
        cuda_available=lambda: False,
        git_commit=lambda: "test-commit",
    )
    if refiner_builder is not None:
        dependency_kwargs["build_window_reference_refiner"] = refiner_builder
    return PipelineDependencies(**dependency_kwargs)


def test_runner_no_loop_never_builds_loop_services(tmp_path):
    events = Events()

    result = PipelineRunner(
        _loaded(tmp_path),
        dependencies=_dependencies(events, "no_loop"),
    ).run()

    assert result.reconstruction_mode is ReconstructionMode.NO_LOOP
    assert events.values == [
        "preflight",
        "images",
        "fingerprint",
        "store",
        "model",
        "provider",
        "segmenter",
        "anchor",
        "no_loop",
        "write_artifact",
    ]


def test_runner_builds_and_injects_window_reference_refiner(tmp_path):
    events = Events()
    loaded = _loaded(tmp_path)
    sentinel = object()
    builder_inputs = []
    captured = []

    def build_refiner(segmentation_config):
        builder_inputs.append(segmentation_config)
        return sentinel

    PipelineRunner(
        loaded,
        dependencies=_dependencies(
            events,
            "no_loop",
            refiner_builder=build_refiner,
            context_sink=captured.append,
        ),
    ).run()

    assert builder_inputs == [loaded.config.segmentation]
    assert captured[0].window_reference_refiner is sentinel


def test_run_from_config_returns_typed_artifact(tmp_path):
    events = Events()
    config, overrides = _config_args(tmp_path)

    result = run_from_config(
        config,
        overrides,
        dependencies=_dependencies(events, "no_loop"),
    )

    assert isinstance(result, ReconstructionArtifact)


def test_loop_diagnostics_do_not_precreate_final_artifact_directory(tmp_path):
    loaded = _loaded(tmp_path, "traditional")
    artifact_dir = tmp_path / "artifacts" / "identity"
    diagnostic_path = tmp_path / "artifacts" / "identity.loop_candidates.json"

    class WritingDetector:
        def __init__(self, output_path):
            self.output_path = Path(output_path)

        def detect(self, *arguments):
            del arguments
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text("[]\n", encoding="utf-8")
            return ()

    dependencies = PipelineDependencies(
        build_loop_detector=(
            lambda config, *, output_path: WritingDetector(output_path)
        ),
        build_loop_evidence=lambda **values: object(),
    )
    services = _build_services(
        ReconstructionMode.TRADITIONAL,
        loaded=loaded,
        dependencies=dependencies,
        model=object(),
        manifest=ImageManifest(paths=()),
        images=torch.empty((0, 3, 2, 2)),
        output_dir=artifact_dir,
    )

    services.detector.detect()
    written = write_reconstruction_artifact(
        _artifact(ReconstructionMode.TRADITIONAL),
        artifact_dir,
        resolved_yaml=loaded.resolved_yaml,
        config_sha256=loaded.sha256,
        checkpoint_sha256="a" * 64,
        git_commit="test-commit",
    )

    assert diagnostic_path.read_text(encoding="utf-8") == "[]\n"
    assert written == artifact_dir
    assert (artifact_dir / "manifest.json").is_file()
