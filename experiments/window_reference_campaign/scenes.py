"""Read-only campaign scene resolution and positional frame selection.

The campaign deliberately resolves source frames before anything is staged.  A
resolved scene therefore carries the original frame IDs and the already
selected source image paths; consumers must not infer another sampling stride
from the dense staged filenames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.manifest import IMAGE_SUFFIXES, natural_sort_key

from .config import (
    CampaignConfig,
    DatasetKind,
    EvaluationKind,
    PresetSceneConfig,
    SceneConfig,
)


@dataclass(frozen=True)
class ResolvedFrameSelection:
    """The one positional selection applied to a source frame vector."""

    start: int
    stop: int
    stride: int
    source_frame_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("resolved selection start must be non-negative")
        if type(self.stop) is not int or self.stop <= self.start:
            raise ValueError("resolved selection stop must be greater than start")
        if type(self.stride) is not int or self.stride < 1:
            raise ValueError("resolved selection stride must be positive")
        if not isinstance(self.source_frame_ids, tuple) or not self.source_frame_ids:
            raise ValueError("resolved source frame vector must be non-empty")
        if any(
            type(frame_id) is not int or frame_id < 0
            for frame_id in self.source_frame_ids
        ):
            raise ValueError("resolved source frame IDs must be non-negative integers")
        if len(set(self.source_frame_ids)) != len(self.source_frame_ids):
            raise ValueError("resolved source frame IDs must be unique")


@dataclass(frozen=True)
class KittiLayout:
    image_dir: Path
    poses_path: Path
    layout_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.image_dir, Path) or not isinstance(self.poses_path, Path):
            raise ValueError("KITTI layout paths must be paths")
        if self.layout_name not in {"normalized", "official"}:
            raise ValueError("KITTI layout name is invalid")


@dataclass(frozen=True)
class ResolvedScene:
    scene_id: str
    dataset: DatasetKind
    scene: str
    slice_id: str
    approved_data_root: Path
    source_images: tuple[Path, ...]
    selection: ResolvedFrameSelection
    evaluation_kind: EvaluationKind
    poses_path: Path | None
    prepared_gt_path: Path | None
    frame_index_map: Path | None
    expected_gt_shape: tuple[int, int] | None

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("resolved scene_id must be a non-empty string")
        if not isinstance(self.dataset, DatasetKind):
            raise ValueError("resolved dataset is invalid")
        if not isinstance(self.scene, str) or not self.scene:
            raise ValueError("resolved scene must be a non-empty string")
        if not isinstance(self.slice_id, str) or not self.slice_id:
            raise ValueError("resolved slice_id must be a non-empty string")
        if not isinstance(self.approved_data_root, Path):
            raise ValueError("approved_data_root must be a path")
        if not isinstance(self.selection, ResolvedFrameSelection):
            raise ValueError("resolved frame selection is invalid")
        if not isinstance(self.source_images, tuple) or not self.source_images:
            raise ValueError("resolved scene requires source images")
        if any(not isinstance(path, Path) for path in self.source_images):
            raise ValueError("resolved source images must be paths")
        if len(self.source_images) != len(self.selection.source_frame_ids):
            raise ValueError("source images and frame IDs must have equal length")
        if not isinstance(self.evaluation_kind, EvaluationKind):
            raise ValueError("resolved evaluation kind is invalid")
        for name in ("poses_path", "prepared_gt_path", "frame_index_map"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise ValueError(f"resolved {name} must be a path or None")
        if self.expected_gt_shape is not None:
            if (
                not isinstance(self.expected_gt_shape, tuple)
                or len(self.expected_gt_shape) != 2
                or any(type(item) is not int or item < 1 for item in self.expected_gt_shape)
            ):
                raise ValueError("resolved expected_gt_shape must be two positive integers")


def _validate_frame_vector(values: Sequence[int], label: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{label} must not be empty")
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{label} must contain non-negative integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique IDs")
    return result


def apply_frame_selection(
    base_frame_ids: Sequence[int],
    selection: PresetSceneConfig,
    *,
    start_override: int | None,
    max_frames: int | None,
    stride_override: int | None,
) -> ResolvedFrameSelection:
    """Apply preset/CLI bounds exactly once, positionally, to ``base_frame_ids``."""

    if not isinstance(selection, PresetSceneConfig):
        raise ValueError("frame selection requires PresetSceneConfig")
    base = _validate_frame_vector(base_frame_ids, "base frame vector")
    if start_override is not None and (type(start_override) is not int or start_override < 0):
        raise ValueError("start_override must be a non-negative integer")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    if stride_override is not None and (type(stride_override) is not int or stride_override < 1):
        raise ValueError("stride_override must be a positive integer")

    start = selection.start if start_override is None else start_override
    stride = selection.stride if stride_override is None else stride_override
    configured_stop = len(base) if selection.stop is None else min(selection.stop, len(base))
    if start >= len(base):
        raise ValueError("frame selection start is outside the source vector")
    if configured_stop <= start:
        raise ValueError("frame selection resolves to an empty range")
    stop = configured_stop
    if max_frames is not None:
        stop = min(stop, start + max_frames * stride)
    if stop <= start:
        raise ValueError("frame selection resolves to an empty range")
    selected = base[start:stop:stride]
    if not selected:
        raise ValueError("frame selection resolves to an empty source vector")
    return ResolvedFrameSelection(start, stop, stride, tuple(selected))


def resolve_kitti_layout(dataset_root: str | Path, sequence: str) -> KittiLayout:
    """Resolve exactly one of normalized and official KITTI extraction layouts."""

    root = Path(dataset_root).resolve(strict=True)
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("KITTI sequence must be a non-empty string")
    candidates = (
        KittiLayout(root / sequence / "image_2", root / sequence / "poses.txt", "normalized"),
        KittiLayout(
            root / "dataset/sequences" / sequence / "image_2",
            root / "dataset/poses" / f"{sequence}.txt",
            "official",
        ),
    )
    matches = tuple(
        item
        for item in candidates
        if item.image_dir.is_dir() and item.poses_path.is_file()
    )
    if len(matches) != 1:
        expected = "; ".join(
            f"{item.image_dir} + {item.poses_path}" for item in candidates
        )
        raise FileNotFoundError(
            f"KITTI {sequence} requires exactly one supported layout: {expected}"
        )
    return KittiLayout(
        matches[0].image_dir.resolve(strict=True),
        matches[0].poses_path.resolve(strict=True),
        matches[0].layout_name,
    )


def _safe_data_path(path: Path, data_root: Path, label: str, *, strict: bool) -> Path:
    resolved = path.resolve(strict=strict)
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError(f"{label} is outside approved data root: {resolved}")
    return resolved


def _source_root(config: CampaignConfig, scene_config: SceneConfig) -> tuple[Path, Path]:
    data_root = Path(config.storage.data_root).resolve(strict=True)
    raw = Path(scene_config.source_root) if scene_config.source_root else data_root
    if not raw.is_absolute():
        raw = data_root / raw
    raw = _safe_data_path(raw, data_root, "scene source root", strict=True)
    return data_root, raw


def _load_sequence_ids(path: Path, scene: str) -> tuple[int, ...]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"sequence map is invalid: {path}") from exc
    if not isinstance(payload, dict) or scene not in payload:
        raise ValueError(f"scene is not present in sequence map: {scene}")
    raw_ids = payload[scene]
    if not isinstance(raw_ids, list):
        raise ValueError(f"invalid sequence map entry: {scene!r}")
    ids = _validate_frame_vector(raw_ids, f"sequence map entry {scene}")
    if ids != tuple(sorted(ids)):
        raise ValueError(f"sequence map entry {scene!r} must be naturally sorted")
    return ids


def _discover_kitti_images(image_dir: Path) -> tuple[Path, ...]:
    images = tuple(
        sorted(
            (
                path.resolve(strict=True)
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=natural_sort_key,
        )
    )
    if not images:
        raise ValueError(f"KITTI image directory contains no supported images: {image_dir}")
    return images


def _read_kitti_pose_rows(path: Path) -> np.ndarray:
    """Read KITTI poses while preserving physical line boundaries."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"KITTI poses are invalid: {path}") from exc
    rows: list[list[float]] = []
    for line_number, line in enumerate(lines, start=1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        tokens = content.split()
        if len(tokens) != 12:
            raise ValueError(
                "KITTI poses must contain 12 values per physical row "
                f"(line {line_number})"
            )
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as exc:
            raise ValueError(f"KITTI poses are invalid: {path}") from exc
    if not rows:
        raise ValueError("KITTI poses must contain at least one physical row")
    array = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("KITTI poses must contain finite values")
    return array


def _load_kitti_poses(path: Path) -> np.ndarray:
    return _read_kitti_pose_rows(path)


def _resolve_mapped_scene(
    config: CampaignConfig,
    scene_config: SceneConfig,
    selected: PresetSceneConfig,
    data_root: Path,
    source_root: Path,
) -> tuple[tuple[Path, ...], ResolvedFrameSelection, Path | None]:
    if scene_config.frame_index_map is None:
        raise ValueError(f"{scene_config.dataset.value} scene requires frame_index_map")
    map_path = Path(scene_config.frame_index_map).resolve(strict=True)
    ids = _load_sequence_ids(map_path, scene_config.scene)
    approved_lengths = {
        (DatasetKind.SEVEN_SCENES, "chess/seq-03"): 100,
        (DatasetKind.NRGBD, "thin_geometry"): 40,
        (DatasetKind.NRGBD, "complete_kitchen"): 122,
    }
    expected_length = approved_lengths.get((scene_config.dataset, scene_config.scene))
    if expected_length is not None and len(ids) != expected_length:
        raise ValueError(
            f"point-cloud sequence {scene_config.scene} must contain exactly "
            f"{expected_length} mapped frames"
        )
    selection = apply_frame_selection(
        ids,
        selected,
        start_override=None,
        max_frames=None,
        stride_override=None,
    )
    source_ids = selection.source_frame_ids
    image_paths: list[Path] = []
    if scene_config.dataset is DatasetKind.SEVEN_SCENES:
        scene_dir = _safe_data_path(source_root / scene_config.scene, data_root, "7-Scenes scene", strict=True)
        for frame_id in source_ids:
            base = scene_dir / f"frame-{frame_id:06d}"
            rgb = base.with_suffix(".color.png")
            depth = base.with_suffix(".depth.png")
            pose = base.with_suffix(".pose.txt")
            for path, label in ((rgb, "RGB"), (depth, "depth"), (pose, "pose")):
                if not path.is_file():
                    raise FileNotFoundError(f"7-Scenes {label} partner is missing: {path}")
                _safe_data_path(path, data_root, f"7-Scenes {label} partner", strict=True)
            image_paths.append(rgb.resolve(strict=True))
    elif scene_config.dataset is DatasetKind.NRGBD:
        scene_dir = _safe_data_path(source_root / scene_config.scene, data_root, "NeuralRGBD scene", strict=True)
        poses_path = scene_dir / "poses.txt"
        _safe_data_path(poses_path, data_root, "NeuralRGBD poses", strict=True)
        poses = _load_nrgbd_poses(poses_path)
        if max(source_ids) >= len(poses):
            raise ValueError("NeuralRGBD pose rows do not cover the selected frame IDs")
        for frame_id in source_ids:
            rgb = scene_dir / "images" / f"img{frame_id}.png"
            depth = scene_dir / "depth" / f"depth{frame_id}.png"
            if not rgb.is_file() or not depth.is_file():
                missing = rgb if not rgb.is_file() else depth
                raise FileNotFoundError(f"NeuralRGBD frame partner is missing: {missing}")
            _safe_data_path(rgb, data_root, "NeuralRGBD RGB frame", strict=True)
            _safe_data_path(depth, data_root, "NeuralRGBD depth frame", strict=True)
            image_paths.append(rgb.resolve(strict=True))
    else:  # pragma: no cover - guarded by resolve_scene
        raise ValueError(f"unsupported mapped dataset: {scene_config.dataset.value}")
    return tuple(image_paths), selection, map_path


def _load_nrgbd_poses(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"NeuralRGBD poses are invalid: {path}") from exc
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or array.size % 16:
        raise ValueError("NeuralRGBD poses must contain 4x4 matrices")
    array = array.reshape(-1, 4, 4)
    if not np.isfinite(array).all():
        raise ValueError("NeuralRGBD poses must contain finite values")
    return array


def _slice_id(selection: PresetSceneConfig) -> str:
    stop = "end" if selection.stop is None else f"{selection.stop:06d}"
    return f"f{selection.start:06d}-{stop}-s{selection.stride}"


def resolve_scene(config: CampaignConfig, selected: PresetSceneConfig) -> ResolvedScene:
    """Resolve one typed campaign scene without writing any source data."""

    if not isinstance(config, CampaignConfig) or not isinstance(selected, PresetSceneConfig):
        raise ValueError("scene resolution requires CampaignConfig and PresetSceneConfig")
    try:
        scene_config = config.scenes[selected.scene_id]
    except KeyError as exc:
        raise ValueError(f"unknown selected scene: {selected.scene_id}") from exc
    if scene_config.dataset is DatasetKind.SYNTHETIC:
        # Synthetic campaigns intentionally do not require a data directory;
        # the runner owns their fixture and staging manifest.  Keep the
        # resolved source paths deterministic for identity/debug payloads.
        data_root = Path(config.storage.data_root).resolve(strict=False)
        frame_total = max(4, selected.stop or (selected.start + 4))
        selection = apply_frame_selection(
            tuple(range(frame_total)),
            selected,
            start_override=None,
            max_frames=None,
            stride_override=None,
        )
        source_images = tuple(
            data_root / "__synthetic__" / f"frame-{frame_id:06d}.png"
            for frame_id in selection.source_frame_ids
        )
        return ResolvedScene(
            scene_id=scene_config.scene_id,
            dataset=scene_config.dataset,
            scene=scene_config.scene,
            slice_id=_slice_id(selected),
            approved_data_root=data_root,
            source_images=source_images,
            selection=selection,
            evaluation_kind=EvaluationKind.NONE,
            poses_path=None,
            prepared_gt_path=None,
            frame_index_map=None,
            expected_gt_shape=None,
        )

    data_root, source_root = _source_root(config, scene_config)

    poses_path: Path | None = None
    frame_index_map: Path | None = None
    if scene_config.dataset is DatasetKind.KITTI:
        layout = resolve_kitti_layout(source_root, scene_config.scene)
        _safe_data_path(layout.image_dir, data_root, "KITTI image directory", strict=True)
        _safe_data_path(layout.poses_path, data_root, "KITTI poses", strict=True)
        images = _discover_kitti_images(layout.image_dir)
        for image in images:
            _safe_data_path(image, data_root, "KITTI source image", strict=True)
        poses = _load_kitti_poses(layout.poses_path)
        if len(images) != len(poses):
            raise ValueError(
                f"KITTI image/pose count mismatch: {len(images)} images, {len(poses)} poses"
            )
        selection = apply_frame_selection(
            tuple(range(len(images))),
            selected,
            start_override=None,
            max_frames=None,
            stride_override=None,
        )
        source_images = tuple(images[index] for index in selection.source_frame_ids)
        poses_path = layout.poses_path
        evaluation = EvaluationKind.INTERNAL_TRAJECTORY
    elif scene_config.dataset in {DatasetKind.SEVEN_SCENES, DatasetKind.NRGBD}:
        source_images, selection, frame_index_map = _resolve_mapped_scene(
            config, scene_config, selected, data_root, source_root
        )
        evaluation = EvaluationKind.POINTCLOUD
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(f"unsupported dataset: {scene_config.dataset.value}")

    prepared: Path | None = None
    if scene_config.prepared_gt is not None:
        candidate = _safe_data_path(
            Path(scene_config.prepared_gt), data_root, "prepared GT", strict=False
        )
        if candidate.exists():
            if not candidate.is_file():
                raise ValueError(f"prepared GT candidate is not a file: {candidate}")
            prepared = candidate
    expected = scene_config.expected_gt_shape
    if evaluation is EvaluationKind.POINTCLOUD and expected != (392, 518):
        raise ValueError("point-cloud GT shape must be exactly (392, 518)")

    return ResolvedScene(
        scene_id=scene_config.scene_id,
        dataset=scene_config.dataset,
        scene=scene_config.scene,
        slice_id=_slice_id(selected),
        approved_data_root=data_root,
        source_images=source_images,
        selection=selection,
        evaluation_kind=evaluation,
        poses_path=poses_path,
        prepared_gt_path=prepared,
        frame_index_map=frame_index_map,
        expected_gt_shape=expected,
    )


__all__ = [
    "KittiLayout",
    "ResolvedFrameSelection",
    "ResolvedScene",
    "_read_kitti_pose_rows",
    "apply_frame_selection",
    "resolve_kitti_layout",
    "resolve_scene",
]
