from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from skimage.segmentation import felzenszwalb

from inference_engine.inference_utils import (
    estimate_pseudo_depth_and_intrinsics,
    register_adjacent_windows,
    unproject_depth_to_local_points,
)
from inference_engine.prediction_cache.types import WindowSpec
from inference_engine.segmentation import (
    build_segmentation_strategy,
    build_temporal_graphs,
)
from inference_engine.segmentation.base import (
    SegmentationResult,
    compact_labels,
    validate_strategy_inputs,
)
from inference_engine.utils._segmentation_cy import merge_regions
from inference_engine.utils.geometry import (
    apply_sim3_to_pose,
    homogenize_points,
)
from pipeline.config import SegmentationMethod


class PaperDepthSegmentationStrategy:
    """Exact depth-layer extraction used by the released LASER evaluator."""

    name = SegmentationMethod.DEPTH

    def __init__(self, config: object) -> None:
        self.confidence_keep_ratio = float(config.confidence_keep_ratio)
        self.depth_merge_threshold = float(config.depth_merge_threshold)
        self.seg_scale = float(config.felzenszwalb.scale)
        self.seg_sigma = float(config.felzenszwalb.sigma)
        self.seg_min_size = int(config.felzenszwalb.min_size)

    def segment(
        self,
        point_maps: np.ndarray,
        confidence: np.ndarray | None,
        images: np.ndarray | None,
    ) -> list[SegmentationResult]:
        del images
        points, confidence_array = validate_strategy_inputs(
            point_maps,
            confidence,
        )
        results = []
        for frame_index, point_map in enumerate(points):
            depth = point_map[..., 2]
            initial_labels = felzenszwalb(
                depth,
                scale=self.seg_scale,
                sigma=self.seg_sigma,
                min_size=self.seg_min_size,
            )
            if confidence_array is None:
                confident_depth = depth
            else:
                frame_confidence = confidence_array[frame_index]
                threshold = np.quantile(
                    frame_confidence.reshape(-1),
                    1.0 - self.confidence_keep_ratio,
                    method="nearest",
                )
                confident_depth = depth[frame_confidence >= threshold]
            merge_threshold = self.depth_merge_threshold * (
                np.max(confident_depth) - np.min(confident_depth)
            )
            labels = compact_labels(
                merge_regions(initial_labels, depth, merge_threshold)
            )
            results.append(
                SegmentationResult(
                    labels=labels,
                    diagnostics={
                        "method": self.name.value,
                        "region_count": int(np.unique(labels).size),
                        "paper_compatibility": True,
                    },
                )
            )
        return results


def build_paper_segmentation_strategy(config: object) -> object:
    if config.method is SegmentationMethod.DEPTH:
        return PaperDepthSegmentationStrategy(config)
    return build_segmentation_strategy(config)


@dataclass(frozen=True)
class PaperStreamingResult:
    local_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    points: torch.Tensor
    segmentation_diagnostics: tuple[tuple[Mapping[str, object], ...], ...]


@dataclass(frozen=True)
class PaperStreamingDependencies:
    register_adjacent_windows: Callable = register_adjacent_windows
    apply_sim3_to_pose: Callable = apply_sim3_to_pose
    build_temporal_graphs: Callable = build_temporal_graphs
    estimate_pseudo_depth_and_intrinsics: Callable = (
        estimate_pseudo_depth_and_intrinsics
    )
    unproject_depth_to_local_points: Callable = (
        unproject_depth_to_local_points
    )


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _paper_confidence_mask(
    confidence: torch.Tensor,
    keep_ratio: float,
) -> torch.Tensor:
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("confidence keep ratio must be in (0, 1]")
    if not torch.isfinite(confidence).all():
        raise ValueError("confidence must contain only finite values")
    quantile_values = (
        confidence
        if confidence.dtype in (torch.float32, torch.float64)
        else confidence.float()
    )
    threshold = torch.quantile(
        quantile_values,
        1.0 - float(keep_ratio),
        interpolation="nearest",
    )
    return confidence >= threshold


def _working_prediction(
    prediction: Mapping[str, object],
    spec: WindowSpec,
    process_device: str,
) -> dict[str, torch.Tensor]:
    result = {}
    expected_frames = spec.frame_count
    for key in ("local_points", "camera_poses", "conf", "images"):
        value = prediction.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"paper streaming prediction is missing {key!r}")
        if value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(
                f"paper streaming prediction {key!r} must have a batch dimension"
            )
        squeezed = value.squeeze(0).to(process_device)
        if squeezed.shape[0] != expected_frames:
            raise ValueError(
                f"paper streaming prediction {key!r} frame count does not "
                "match WindowSpec"
            )
        if not torch.isfinite(squeezed).all():
            raise ValueError(
                f"paper streaming prediction {key!r} contains non-finite values"
            )
        result[key] = squeezed
    if result["local_points"].ndim != 4 or result["local_points"].shape[-1] != 3:
        raise ValueError("local_points must have shape (N,H,W,3)")
    if result["camera_poses"].shape != (expected_frames, 4, 4):
        raise ValueError("camera_poses must have shape (N,4,4)")
    if result["conf"].shape != result["local_points"].shape[:-1]:
        raise ValueError("confidence shape must match local point maps")
    return result


def reconstruct_incremental_point_maps(
    *,
    provider: object,
    specs: Sequence[WindowSpec],
    images: torch.Tensor,
    segmenter: object,
    anchor_propagator: object,
    overlap: int,
    confidence_keep_ratio: float,
    temporal_iou_threshold: float,
    anchor_enabled: bool,
    process_device: str = "cpu",
    dependencies: PaperStreamingDependencies | None = None,
) -> PaperStreamingResult:
    """Replay LASER's paper-time incremental global-map assembly.

    The production loop strategies intentionally retain raw per-window caches
    and defer transforms until aggregation.  LASER Table 4 instead registers
    every window with the first window's estimated intrinsics, registers each
    incoming window against the already corrected global predecessor, and
    applies both Sim(3) scale and layer scale before the next registration.
    This evaluator-local replay preserves that published ordering without
    changing the reusable inference or loop-closure implementations.
    """

    selected = dependencies or PaperStreamingDependencies()
    normalized_specs = tuple(specs)
    if not normalized_specs:
        raise ValueError("paper streaming requires at least one window")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must have shape (N,3,H,W)")
    if overlap < 1:
        raise ValueError("overlap must be positive")
    if not 0.0 <= float(temporal_iou_threshold) <= 1.0:
        raise ValueError("temporal IoU threshold must be in [0, 1]")

    previous: dict[str, torch.Tensor] | None = None
    previous_graph = None
    reference_intrinsic: torch.Tensor | None = None
    previous_frame_end: int | None = None
    point_chunks = []
    pose_chunks = []
    confidence_chunks = []
    all_segmentation_diagnostics = []

    for expected_index, spec in enumerate(normalized_specs):
        if not isinstance(spec, WindowSpec) or spec.index != expected_index:
            raise ValueError("paper streaming WindowSpecs must be in order")
        window_images = images[spec.frame_start : spec.frame_end]
        prediction = provider.get(spec, window_images)
        working = _working_prediction(
            prediction,
            spec,
            process_device,
        )
        target_mask = _paper_confidence_mask(
            working["conf"][:overlap],
            confidence_keep_ratio,
        )

        if previous is not None:
            if reference_intrinsic is None:
                raise RuntimeError("paper streaming reference intrinsic is missing")
            working["local_points"] = selected.unproject_depth_to_local_points(
                working["local_points"][..., 2],
                reference_intrinsic,
            )
            previous_mask = _paper_confidence_mask(
                previous["conf"][-overlap:],
                confidence_keep_ratio,
            )
            if previous_mask.shape != target_mask.shape:
                raise ValueError(
                    "adjacent paper streaming confidence masks do not match"
                )
            mutual_mask = previous_mask & target_mask
            if not torch.any(mutual_mask):
                raise ValueError(
                    "paper streaming registration has no mutual confidence pixels"
                )
            scale, rotation, translation = selected.register_adjacent_windows(
                previous["local_points"][-overlap:],
                working["local_points"][:overlap],
                previous["camera_poses"][-overlap:],
                working["camera_poses"][:overlap],
                mutual_mask,
            )
            working["local_points"] = scale * working["local_points"]
            working["camera_poses"] = selected.apply_sim3_to_pose(
                working["camera_poses"],
                scale,
                rotation,
                translation,
            )
        else:
            _, estimated_intrinsics = (
                selected.estimate_pseudo_depth_and_intrinsics(
                    working["local_points"]
                )
            )
            if (
                not isinstance(estimated_intrinsics, torch.Tensor)
                or estimated_intrinsics.ndim != 3
                or estimated_intrinsics.shape[1:] != (3, 3)
                or estimated_intrinsics.shape[0] < 1
                or not torch.isfinite(estimated_intrinsics).all()
            ):
                raise ValueError(
                    "paper streaming intrinsic estimator returned invalid intrinsics"
                )
            reference_intrinsic = estimated_intrinsics[0]
            working["local_points"] = selected.unproject_depth_to_local_points(
                working["local_points"][..., 2],
                reference_intrinsic,
            )

        segmentation_results = segmenter.segment(
            _as_numpy(working["local_points"]),
            _as_numpy(working["conf"]),
            _as_numpy(working["images"]),
        )
        target_graph = selected.build_temporal_graphs(
            segmentation_results,
            temporal_iou_threshold,
        )
        all_segmentation_diagnostics.append(
            tuple(dict(result.diagnostics) for result in segmentation_results)
        )

        if previous is not None and anchor_enabled:
            scale_mask = anchor_propagator.propagate(
                _as_numpy(previous["local_points"]),
                _as_numpy(working["local_points"]),
                previous_graph,
                target_graph,
                overlap,
            ).to(
                device=working["local_points"].device,
                dtype=working["local_points"].dtype,
            )
            if scale_mask.shape != (*working["local_points"].shape[:-1], 1):
                raise ValueError(
                    "paper streaming anchor scale mask has invalid shape"
                )
            if not torch.isfinite(scale_mask).all():
                raise ValueError(
                    "paper streaming anchor scale mask contains non-finite values"
                )
            working["local_points"] = working["local_points"] * scale_mask

        trim = 0
        if previous_frame_end is not None:
            trim = max(0, previous_frame_end - spec.frame_start)
        point_chunks.append(working["local_points"][trim:])
        pose_chunks.append(working["camera_poses"][trim:])
        confidence_chunks.append(working["conf"][trim:])
        previous_frame_end = max(
            spec.frame_end,
            previous_frame_end or spec.frame_end,
        )
        previous = working
        previous_graph = target_graph

    local_points = torch.cat(point_chunks, dim=0)
    camera_poses = torch.cat(pose_chunks, dim=0)
    confidence = torch.cat(confidence_chunks, dim=0)
    if local_points.shape[0] != images.shape[0]:
        raise ValueError(
            "paper streaming aggregation did not cover every input frame once"
        )
    points = torch.einsum(
        "nij,nhwj->nhwi",
        camera_poses,
        homogenize_points(local_points),
    )[..., :3]
    return PaperStreamingResult(
        local_points=local_points,
        camera_poses=camera_poses,
        confidence=confidence,
        points=points,
        segmentation_diagnostics=tuple(all_segmentation_diagnostics),
    )
