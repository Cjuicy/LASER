from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from inference_engine.prediction_cache.types import WindowSpec
from inference_engine.segmentation.base import SegmentationResult
from mv_recon.loop_experiment import LoopPointMapResult
from mv_recon.paper_streaming import (
    PaperDepthSegmentationStrategy,
    PaperStreamingDependencies,
    PaperStreamingResult,
    build_paper_segmentation_strategy,
    reconstruct_incremental_point_maps,
)
from pipeline.config import SegmentationMethod


class FakeProvider:
    def get(self, spec: WindowSpec, images: torch.Tensor):
        assert images.shape[0] == spec.frame_count
        count = spec.frame_count
        local_points = torch.zeros((count, 1, 1, 3))
        local_points[..., 2] = 1.0
        poses = torch.eye(4).repeat(count, 1, 1)
        confidence = torch.arange(count, dtype=torch.float32).view(
            count, 1, 1
        )
        return {
            "local_points": local_points.unsqueeze(0),
            "camera_poses": poses.unsqueeze(0),
            "conf": confidence.unsqueeze(0),
            "images": images.unsqueeze(0),
        }


class FakeSegmenter:
    def segment(self, point_maps, confidence, images):
        assert point_maps.shape[0] == confidence.shape[0] == images.shape[0]
        return [
            SegmentationResult(
                labels=np.zeros(point_maps.shape[1:3], dtype=np.intp),
                diagnostics={"window_depth": float(point_maps[index, 0, 0, 2])},
            )
            for index in range(point_maps.shape[0])
        ]


@dataclass
class RecordingAnchor:
    calls: list[tuple[float, float]]

    def propagate(
        self,
        source_points,
        target_points,
        source_graphs,
        target_graphs,
        overlap,
    ):
        del source_graphs, target_graphs, overlap
        self.calls.append(
            (
                float(source_points[-1, 0, 0, 2]),
                float(target_points[0, 0, 0, 2]),
            )
        )
        scale = 5.0 if len(self.calls) == 1 else 7.0
        return torch.full((*target_points.shape[:-1], 1), scale)


def test_incremental_reconstruction_applies_global_scale_and_anchor_immediately():
    specs = (
        WindowSpec(index=0, frame_start=0, frame_end=2),
        WindowSpec(index=1, frame_start=1, frame_end=3),
        WindowSpec(index=2, frame_start=2, frame_end=4),
    )
    registration_sources = []
    registration_scales = iter((2.0, 3.0))

    def register(source_points, target_points, source_poses, target_poses, mask):
        del source_poses, target_poses, mask
        registration_sources.append(
            (
                float(source_points[-1, 0, 0, 2]),
                float(target_points[0, 0, 0, 2]),
            )
        )
        return (
            next(registration_scales),
            torch.eye(3),
            torch.zeros(3),
        )

    anchor = RecordingAnchor(calls=[])
    dependencies = PaperStreamingDependencies(
        register_adjacent_windows=register,
        apply_sim3_to_pose=lambda poses, scale, rotation, translation: poses,
        build_temporal_graphs=lambda results, threshold: (
            tuple(result.labels for result in results),
            threshold,
        ),
    )

    result = reconstruct_incremental_point_maps(
        provider=FakeProvider(),
        specs=specs,
        images=torch.zeros((4, 3, 1, 1)),
        segmenter=FakeSegmenter(),
        anchor_propagator=anchor,
        overlap=1,
        confidence_keep_ratio=0.5,
        temporal_iou_threshold=0.3,
        anchor_enabled=True,
        dependencies=dependencies,
    )

    assert registration_sources == [(1.0, 1.0), (10.0, 1.0)]
    assert anchor.calls == [(1.0, 2.0), (10.0, 3.0)]
    assert result.local_points[:, 0, 0, 2].tolist() == [1.0, 1.0, 10.0, 21.0]
    assert result.points[:, 0, 0, 2].tolist() == [1.0, 1.0, 10.0, 21.0]
    assert result.confidence.shape == (4, 1, 1)
    assert len(result.segmentation_diagnostics) == 3


def test_incremental_reconstruction_enforces_first_window_intrinsics():
    specs = (
        WindowSpec(index=0, frame_start=0, frame_end=2),
        WindowSpec(index=1, frame_start=1, frame_end=3),
    )
    reference_intrinsic = torch.tensor(
        [[7.0, 0.0, 0.5], [0.0, 7.0, 0.5], [0.0, 0.0, 1.0]]
    )
    estimate_calls = []
    unproject_calls = []

    def estimate(points):
        estimate_calls.append(points.clone())
        return points[..., 2], reference_intrinsic.unsqueeze(0)

    def unproject(depth, intrinsic):
        unproject_calls.append((depth.clone(), intrinsic.clone()))
        points = torch.zeros((*depth.shape, 3), dtype=depth.dtype)
        points[..., 0] = 10.0 * len(unproject_calls)
        points[..., 2] = depth
        return points

    dependencies = PaperStreamingDependencies(
        register_adjacent_windows=lambda *args: (
            1.0,
            torch.eye(3),
            torch.zeros(3),
        ),
        apply_sim3_to_pose=lambda poses, scale, rotation, translation: poses,
        build_temporal_graphs=lambda results, threshold: results,
        estimate_pseudo_depth_and_intrinsics=estimate,
        unproject_depth_to_local_points=unproject,
    )

    result = reconstruct_incremental_point_maps(
        provider=FakeProvider(),
        specs=specs,
        images=torch.zeros((3, 3, 1, 1)),
        segmenter=FakeSegmenter(),
        anchor_propagator=RecordingAnchor(calls=[]),
        overlap=1,
        confidence_keep_ratio=0.5,
        temporal_iou_threshold=0.3,
        anchor_enabled=False,
        dependencies=dependencies,
    )

    assert len(estimate_calls) == 1
    assert len(unproject_calls) == 2
    assert all(
        torch.equal(intrinsic, reference_intrinsic)
        for _, intrinsic in unproject_calls
    )
    assert result.local_points[:, 0, 0, 0].tolist() == [10.0, 10.0, 20.0]


def test_paper_depth_segmentation_uses_original_nearest_quantile(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "mv_recon.paper_streaming.felzenszwalb",
        lambda depth, **kwargs: np.zeros(depth.shape, dtype=np.intp),
    )

    def merge_regions(labels, depth, threshold):
        del depth
        captured["threshold"] = threshold
        return labels

    monkeypatch.setattr(
        "mv_recon.paper_streaming.merge_regions",
        merge_regions,
    )
    config = type(
        "Config",
        (),
        {
            "confidence_keep_ratio": 0.5,
            "depth_merge_threshold": 0.1,
            "felzenszwalb": type(
                "Felzenszwalb",
                (),
                {"scale": 300, "sigma": 1.1, "min_size": 500},
            )(),
        },
    )()
    points = np.zeros((1, 2, 3, 3), dtype=np.float32)
    points[0, ..., 2] = np.arange(6).reshape(2, 3) * 10.0
    confidence = np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    result = PaperDepthSegmentationStrategy(config).segment(
        points,
        confidence,
        None,
    )

    # q=.5 over six values is 2 with "nearest" but 3 with "higher".
    # The official selection therefore spans depths 20..50.
    assert captured["threshold"] == 3.0
    assert result[0].diagnostics["region_count"] == 1


def test_only_depth_uses_paper_compatibility_segmenter(monkeypatch):
    fallback = object()
    monkeypatch.setattr(
        "mv_recon.paper_streaming.build_segmentation_strategy",
        lambda config: fallback,
    )
    depth_config = type(
        "DepthConfig",
        (),
        {
            "method": SegmentationMethod.DEPTH,
            "confidence_keep_ratio": 0.5,
            "depth_merge_threshold": 0.1,
            "felzenszwalb": type(
                "Felzenszwalb",
                (),
                {"scale": 300, "sigma": 1.1, "min_size": 500},
            )(),
        },
    )()
    geometry_config = type(
        "GeometryConfig",
        (),
        {"method": SegmentationMethod.GEOMETRY},
    )()

    assert isinstance(
        build_paper_segmentation_strategy(depth_config),
        PaperDepthSegmentationStrategy,
    )
    assert build_paper_segmentation_strategy(geometry_config) is fallback


def test_eval_inference_uses_paper_incremental_replay(monkeypatch, tmp_path):
    from mv_recon import eval as eval_module

    image_paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"image")
        image_paths.append(str(path))

    class Store:
        fingerprint = SimpleNamespace(key="f" * 64)

        @contextmanager
        def entry_lock(self):
            yield

    class ForbiddenLoopStrategy:
        def build_constraints(self, *args, **kwargs):
            raise AssertionError("evaluation must not use deferred loop aggregation")

    engine = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            segmentation=SimpleNamespace(
                method=SegmentationMethod.DEPTH,
                temporal_iou_threshold=0.3,
            ),
            anchor_propagation=SimpleNamespace(
                enabled=True,
                correspondence_iou_threshold=0.4,
            ),
            window=SimpleNamespace(overlap=1),
            loop=SimpleNamespace(
                registration=SimpleNamespace(confidence_keep_ratio=0.5)
            ),
            model=SimpleNamespace(process_device="cpu"),
        ),
        prediction_store=Store(),
        model_handle=object(),
        window_specs=(WindowSpec(0, 0, 4),),
        delegate=object(),
        loop_strategy=ForbiddenLoopStrategy(),
    )
    model = SimpleNamespace(prepare=lambda images, manifest: engine)
    replay_calls = []

    def replay(**kwargs):
        replay_calls.append(kwargs)
        points = torch.zeros((4, 1, 1, 3))
        points[..., 2] = 2.0
        return PaperStreamingResult(
            local_points=points.clone(),
            camera_poses=torch.eye(4).repeat(4, 1, 1),
            confidence=torch.ones((4, 1, 1)),
            points=points,
            segmentation_diagnostics=(),
        )

    monkeypatch.setattr(
        eval_module,
        "load_and_preprocess_images",
        lambda paths: torch.zeros((len(paths), 3, 1, 1)),
    )
    monkeypatch.setattr(
        eval_module,
        "reconstruct_incremental_point_maps",
        replay,
        raising=False,
    )
    monkeypatch.setattr(
        eval_module,
        "build_paper_segmentation_strategy",
        lambda config: "paper-segmenter",
        raising=False,
    )
    monkeypatch.setattr(
        eval_module,
        "AnchorPropagator",
        lambda threshold: ("paper-anchor", threshold),
        raising=False,
    )
    monkeypatch.setattr(
        eval_module,
        "collect_prediction_diagnostics",
        lambda **kwargs: {"ordinary_hits": 1, "ordinary_misses": 0},
    )
    monkeypatch.setattr(
        eval_module,
        "reconstruct_traditional_loop_point_maps",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("paper evaluation must not use the loop adapter")
        ),
        raising=False,
    )

    output = eval_module.run_streaming_inference(
        image_paths,
        model,
        OmegaConf.create(
            {"protocol": {"mode": "paper"}, "output_dir": str(tmp_path)}
        ),
        (2, 3),
    )

    assert len(replay_calls) == 1
    assert replay_calls[0]["provider"] is engine.delegate
    assert replay_calls[0]["segmenter"] == "paper-segmenter"
    assert output.points.shape == (4, 2, 3, 3)
    assert np.all(output.points[..., 2] == 2.0)
    assert output.ordinary_prediction_key == "f" * 64
    assert output.cache_diagnostics == {
        "ordinary_hits": 1,
        "ordinary_misses": 0,
    }


def test_eval_inference_dispatches_experiment_to_traditional_loop(
    monkeypatch,
    tmp_path,
):
    from mv_recon import eval as eval_module

    image_paths = []
    for index in range(4):
        path = tmp_path / f"loop-{index}.png"
        path.write_bytes(b"image")
        image_paths.append(str(path))

    class Store:
        fingerprint = SimpleNamespace(key="e" * 64)

    engine = SimpleNamespace(
        prediction_store=Store(),
        model_handle=object(),
    )
    model = SimpleNamespace(prepare=lambda images, manifest: engine)
    loop_calls = []

    def loop_reconstruct(**kwargs):
        loop_calls.append(kwargs)
        points = torch.zeros((4, 1, 1, 3))
        points[..., 2] = 3.0
        return LoopPointMapResult(
            points=points,
            confidence=torch.ones((4, 1, 1)),
            ordinary_prediction_key="e" * 64,
            diagnostics={
                "candidate_count": 1,
                "constraint_count": 1,
                "rejected_candidate_count": 0,
                "used_no_loop_path": False,
            },
        )

    monkeypatch.setattr(
        eval_module,
        "load_and_preprocess_images",
        lambda paths: torch.zeros((len(paths), 3, 1, 1)),
    )
    monkeypatch.setattr(
        eval_module,
        "reconstruct_traditional_loop_point_maps",
        loop_reconstruct,
        raising=False,
    )
    monkeypatch.setattr(
        eval_module,
        "reconstruct_incremental_point_maps",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("experiment must not use paper incremental replay")
        ),
    )

    output = eval_module.run_streaming_inference(
        image_paths,
        model,
        OmegaConf.create(
            {
                "protocol": {"mode": "experiment"},
                "output_dir": str(tmp_path),
            }
        ),
        (2, 3),
    )

    assert len(loop_calls) == 1
    assert loop_calls[0]["engine"] is engine
    assert loop_calls[0]["artifact_dir"] == (
        tmp_path / "loop_artifacts" / ("e" * 64)
    )
    assert output.points.shape == (4, 2, 3, 3)
    assert np.all(output.points[..., 2] == 3.0)
    assert output.ordinary_prediction_key == "e" * 64
    assert output.cache_diagnostics["constraint_count"] == 1
