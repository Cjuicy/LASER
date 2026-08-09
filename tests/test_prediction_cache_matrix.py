from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from inference_engine.inference_utils import (
    unproject_depth_to_local_points,
)
from inference_engine.models.lazy import LazyModelHandle
from loop_closure.methods.base import (
    WINDOW_CACHE_SCHEMA_VERSION,
    LoopSolution,
    ReconstructionResult,
    WindowCache,
)
from pipeline.config import LoopMethod, ModelName
from pipeline.runner import PipelineDependencies, run_from_config
from scripts.verify_pipeline_matrix import build_matrix


IDENTITY_SIM3 = (1.0, torch.eye(3), torch.zeros(3))


class CountingPredictionModel(torch.nn.Module):
    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, frames, _, height, width = images.shape
        depth = torch.ones((frames, height, width), dtype=torch.float32)
        intrinsic = torch.tensor(
            [
                [2.0, 0.0, width / 2.0],
                [0.0, 2.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        local_points = unproject_depth_to_local_points(
            depth,
            intrinsic,
        ).unsqueeze(0).expand(batch, -1, -1, -1, -1)
        return {
            "local_points": local_points,
            "camera_poses": torch.eye(4).repeat(
                batch,
                frames,
                1,
                1,
            ),
            "conf": torch.ones((batch, frames, height, width)),
            "images": images,
        }


class PredictionCacheMatrixEngine:
    def __init__(self, **dependencies):
        self.delegate = dependencies["delegate"]
        self.prediction_key = dependencies["prediction_key"]
        self.model_name = dependencies["model_name"]
        self.checkpoint_digest = dependencies["checkpoint_digest"]
        self.loop_method = dependencies["loop_method"]
        self.cache_root = Path(dependencies["cache_root"])
        self.temp_cache_dir = None

    def begin(self):
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.temp_cache_dir = Path(
            tempfile.mkdtemp(dir=self.cache_root)
        )

    def __call__(self, images, *, window_spec):
        prediction = self.delegate.get(window_spec, images)
        local_points = prediction["local_points"].squeeze(0)
        camera_poses = prediction["camera_poses"].squeeze(0)
        confidence = prediction["conf"].squeeze(0)
        cache = WindowCache(
            schema_version=WINDOW_CACHE_SCHEMA_VERSION,
            loop_method=self.loop_method,
            prediction_key=self.prediction_key,
            model_name=self.model_name,
            checkpoint_digest=self.checkpoint_digest,
            window_index=window_spec.index,
            frame_start=window_spec.frame_start,
            frame_end=window_spec.frame_end,
            local_points=local_points,
            camera_poses=camera_poses,
            confidence=confidence,
            segmentation_labels=tuple(
                np.zeros(
                    tuple(local_points.shape[1:3]),
                    dtype=np.intp,
                )
                for _ in range(window_spec.frame_count)
            ),
            anchor_scale_mask=None,
            loop_state={"tag": self.loop_method.value},
            segmentation_diagnostics=tuple(
                {"method": "matrix-fake", "region_count": 1}
                for _ in range(window_spec.frame_count)
            ),
        )
        torch.save(
            cache.to_payload(),
            self.temp_cache_dir
            / f"window_cache_{window_spec.index}.pt",
        )

    def end(self):
        return None


class PredictionCacheMatrixLoopStrategy:
    def __init__(self, method):
        self.method = method

    def create_window_engine(self, **dependencies):
        return PredictionCacheMatrixEngine(
            **dependencies,
            loop_method=self.method,
        )

    def build_constraints(
        self,
        caches,
        candidates,
        *,
        constraint_estimator=None,
    ):
        assert candidates == ()
        assert constraint_estimator is None
        return []

    def optimize(self, caches, constraints):
        return LoopSolution(
            optimized_transforms=tuple(
                IDENTITY_SIM3 for _ in caches
            ),
            constraints=(),
            used_no_loop_path=True,
        )

    def aggregate(self, caches, solution):
        frame_count = caches[-1].frame_end
        height, width = caches[0].confidence.shape[-2:]
        depth = torch.ones((frame_count, height, width))
        intrinsic = torch.tensor(
            [
                [2.0, 0.0, width / 2.0],
                [0.0, 2.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return ReconstructionResult(
            payload={
                "local_points": unproject_depth_to_local_points(
                    depth,
                    intrinsic,
                ),
                "camera_poses": torch.eye(4).repeat(
                    frame_count,
                    1,
                    1,
                ),
                "confidence": torch.ones(
                    (frame_count, height, width)
                ),
            },
            summary={
                "window_count": len(caches),
                "used_no_loop_path": True,
            },
        )


def _matrix_overrides(tmp_path, entry):
    return (
        f"input.image_dir={tmp_path / 'images'}",
        "input.sample_stride=1",
        f"model.checkpoint={tmp_path / 'model.safetensors'}",
        "model.inference_device=cpu",
        "model.process_device=cpu",
        "model.dtype=float32",
        f"prediction_cache.root={tmp_path / 'predictions'}",
        "prediction_cache.mode=auto",
        "window.size=10",
        "window.overlap=5",
        *entry.overrides(
            base_scene="matrix",
            cache_root=tmp_path / "method-cache",
            result_root=tmp_path / "results",
        ),
    )


def _matrix_dependencies(
    *,
    images,
    fail_model_factory,
    salad_calls,
):
    def build_model_handle(config):
        def factory():
            if fail_model_factory:
                raise AssertionError(
                    "warm matrix cache must not construct a model"
                )
            return CountingPredictionModel()

        return LazyModelHandle(
            factory,
            inference_device="cpu",
            dtype=torch.float32,
        )

    def detect_candidates(config, manifest, output_path):
        salad_calls.append(tuple(manifest.paths))
        return ()

    return PipelineDependencies(
        validate_preflight=lambda *args: None,
        build_model_handle=build_model_handle,
        load_images=lambda manifest: images.clone(),
        build_segmentation_strategy=lambda config: object(),
        build_anchor_propagator=lambda threshold: object(),
        build_loop_strategy=lambda method, **kwargs: (
            PredictionCacheMatrixLoopStrategy(method)
        ),
        detect_loop_candidates=detect_candidates,
        save_for_viser=lambda *args, **kwargs: None,
        cuda_available=lambda: False,
        git_commit=lambda: "matrix-test",
    )


def test_ten_configurations_reuse_two_ordinary_window_forwards(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(15):
        (image_dir / f"frame_{index:04d}.png").write_bytes(
            f"frame-{index}".encode("ascii")
        )
    (tmp_path / "model.safetensors").write_bytes(b"fake-checkpoint")
    images = torch.arange(
        15 * 3 * 2 * 2,
        dtype=torch.float32,
    ).reshape(15, 3, 2, 2)
    entries = build_matrix(ModelName.PI3)

    first_salad_calls = []
    first_summaries = [
        run_from_config(
            "configs/pipeline/test.yaml",
            _matrix_overrides(tmp_path, entry),
            dependencies=_matrix_dependencies(
                images=images,
                fail_model_factory=False,
                salad_calls=first_salad_calls,
            ),
        ).summary
        for entry in entries
    ]

    assert first_summaries[0]["ordinary_forward_count"] == 2
    assert [
        summary["ordinary_forward_count"]
        for summary in first_summaries[1:]
    ] == [0] * 9
    assert sum(
        summary["ordinary_forward_count"]
        for summary in first_summaries
    ) == 2
    assert len(first_salad_calls) == 10
    assert {
        summary["joint_forward_count"]
        for summary in first_summaries
    } == {0}

    second_salad_calls = []
    second_summaries = [
        run_from_config(
            "configs/pipeline/test.yaml",
            _matrix_overrides(tmp_path, entry),
            dependencies=_matrix_dependencies(
                images=images,
                fail_model_factory=True,
                salad_calls=second_salad_calls,
            ),
        ).summary
        for entry in entries
    ]

    assert [
        summary["ordinary_forward_count"]
        for summary in second_summaries
    ] == [0] * 10
    assert len(second_salad_calls) == 10
    assert {
        summary["joint_forward_count"]
        for summary in second_summaries
    } == {0}
