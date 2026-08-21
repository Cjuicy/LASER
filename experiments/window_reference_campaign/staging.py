"""Campaign-owned, atomic staging for aligned images, poses, and point maps."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .config import DatasetKind, EvaluationKind
from .scenes import ResolvedFrameSelection, ResolvedScene


STAGING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StagedScene:
    scene_id: str
    dataset: DatasetKind
    scene: str
    slice_id: str
    image_dir: Path
    source_frame_ids: tuple[int, ...]
    selection: ResolvedFrameSelection
    poses_path: Path | None
    pointcloud_gt_path: Path | None
    manifest_path: Path
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("staged scene_id must be a non-empty string")
        if not isinstance(self.dataset, DatasetKind):
            raise ValueError("staged dataset is invalid")
        if not isinstance(self.scene, str) or not self.scene:
            raise ValueError("staged scene must be a non-empty string")
        if not isinstance(self.slice_id, str) or not self.slice_id:
            raise ValueError("staged slice_id must be a non-empty string")
        for name in ("image_dir", "manifest_path"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"staged {name} must be a path")
        if not isinstance(self.source_frame_ids, tuple) or not self.source_frame_ids:
            raise ValueError("staged source frame IDs must be non-empty")
        if any(type(item) is not int or item < 0 for item in self.source_frame_ids):
            raise ValueError("staged source frame IDs must be non-negative integers")
        if tuple(self.source_frame_ids) != tuple(self.selection.source_frame_ids):
            raise ValueError("staged source frame IDs must match the selection")
        for name in ("poses_path", "pointcloud_gt_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise ValueError(f"staged {name} must be a path or None")
        if (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.manifest_sha256)
        ):
            raise ValueError("staged manifest SHA256 is invalid")


def _canonical_json_bytes(value: object) -> bytes:
    def default(item: object) -> str:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"not JSON serializable: {type(item).__name__}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"staging file hash mismatch: {path}") from exc
    return digest.hexdigest()


def require_descendant(
    path: str | Path,
    root: str | Path,
    *,
    allow_equal: bool = False,
) -> Path:
    """Return a resolved path only when it is inside the owned ``root``."""

    target = Path(path).resolve(strict=False)
    owner = Path(root).resolve(strict=False)
    if target == owner and allow_equal:
        return target
    if target == owner or owner not in target.parents:
        raise ValueError(f"path is outside campaign-owned root: {target}")
    return target


def guarded_remove(path: str | Path, owned_root: str | Path) -> None:
    """Remove one proven-owned descendant, never the ownership root itself."""

    target = require_descendant(path, owned_root)
    original = Path(path)
    if original.is_symlink() or original.is_file():
        original.unlink(missing_ok=True)
    elif original.is_dir():
        shutil.rmtree(original)
    elif target.exists():  # defensive handling for unusual path objects
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)


def _atomic_path(target: Path, writer: Callable[[Path], None]) -> Path:
    """Write a sibling temporary file then atomically replace ``target``."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _atomic_json(target: Path, payload: object) -> Path:
    serialized = _canonical_json_bytes(payload)

    def writer(path: Path) -> None:
        with path.open("wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())

    return _atomic_path(target, writer)


def _atomic_npy(target: Path, array: np.ndarray) -> Path:
    def writer(path: Path) -> None:
        # Passing an open file prevents NumPy from adding its own .npy suffix
        # to our temporary sibling.
        with path.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())

    return _atomic_path(target, writer)


def _atomic_npz(target: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    def writer(path: Path) -> None:
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())

    return _atomic_path(target, writer)


def _relative_link(source: Path, target: Path, approved_root: Path) -> None:
    resolved = source.resolve(strict=True)
    try:
        require_descendant(resolved, approved_root)
    except ValueError as exc:
        raise ValueError(
            f"source path is outside approved data root: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"source image is not a file: {resolved}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ValueError(f"staging target already exists: {target}")
    target.symlink_to(os.path.relpath(resolved, start=target.parent))


def _scene_source_records(scene: ResolvedScene) -> list[dict[str, object]]:
    approved = scene.approved_data_root.resolve(strict=False)
    records: list[dict[str, object]] = []
    if len(scene.source_images) != len(scene.selection.source_frame_ids):
        raise ValueError("source images and frame IDs must have equal length")
    for ordinal, (frame_id, path) in enumerate(
        zip(scene.selection.source_frame_ids, scene.source_images, strict=True)
    ):
        resolved = path.resolve(strict=True)
        try:
            require_descendant(resolved, approved)
        except ValueError as exc:
            raise ValueError(
                f"source path is outside approved data root: {resolved}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"source image is not a file: {resolved}")
        records.append(
            {
                "ordinal": ordinal,
                "source_frame_id": frame_id,
                "source_path": str(resolved),
                "source_sha256": _sha256_file(resolved),
            }
        )
    return records


def _source_file_record(path: Path | None, approved: Path, label: str) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.resolve(strict=True)
    try:
        require_descendant(resolved, approved)
    except ValueError as exc:
        raise ValueError(f"{label} is outside approved data root: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return {"source_path": str(resolved), "source_sha256": _sha256_file(resolved)}


def preview_staging_manifest(scene: ResolvedScene) -> tuple[dict[str, object], str]:
    """Build the source-only manifest payload and its stable digest."""

    if not isinstance(scene, ResolvedScene):
        raise ValueError("staging preview requires ResolvedScene")
    image_records = _scene_source_records(scene)
    approved = scene.approved_data_root.resolve(strict=False)
    payload: dict[str, object] = {
        "schema_version": STAGING_SCHEMA_VERSION,
        "scene_id": scene.scene_id,
        "dataset": scene.dataset.value,
        "scene": scene.scene,
        "slice_id": scene.slice_id,
        "approved_data_root": str(approved),
        "selection": {
            "start": scene.selection.start,
            "stop": scene.selection.stop,
            "stride": scene.selection.stride,
        },
        "source_frame_ids": list(scene.selection.source_frame_ids),
        "pipeline_sample_stride": 1,
        "images": image_records,
    }
    pose_record = _source_file_record(scene.poses_path, approved, "poses source")
    if pose_record is not None:
        payload["poses_source"] = pose_record
    gt_record = _source_file_record(
        scene.prepared_gt_path, approved, "prepared GT source"
    )
    if gt_record is not None:
        payload["prepared_gt_source"] = gt_record
    return payload, _canonical_sha256(payload)


def _read_kitti_pose_rows(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"KITTI poses are invalid: {path}") from exc
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or array.size % 12:
        raise ValueError("KITTI poses must contain rows of 12 values")
    array = array.reshape(-1, 12)
    if not np.isfinite(array).all():
        raise ValueError("KITTI poses must contain finite values")
    return array


def _validate_external_gt(
    scene: ResolvedScene,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = scene.expected_gt_shape
    if expected != (392, 518):
        raise ValueError("point-cloud GT shape must be exactly (392, 518)")
    try:
        require_descendant(path.resolve(strict=True), scene.approved_data_root)
    except ValueError as exc:
        raise ValueError(
            f"prepared GT is outside approved data root: {path}"
        ) from exc
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"point-map GT archive is invalid: {path}") from exc
    with archive:
        if "point_maps" not in archive or "valid_mask" not in archive:
            raise ValueError("point-map GT archive requires point_maps and valid_mask")
        point_maps = np.asarray(archive["point_maps"])
        valid_mask = np.asarray(archive["valid_mask"])
        candidates: list[tuple[str, np.ndarray]] = []
        if "frame_ids" in archive:
            candidates.append(("frame_ids", np.asarray(archive["frame_ids"])))
        for name in ("source_frame_ids.npy", "frame_ids.npy"):
            sibling = path.parent / name
            if sibling.is_file():
                try:
                    candidates.append((name, np.load(sibling, allow_pickle=False)))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"point-map GT frame IDs are invalid: {sibling}") from exc
        if not candidates:
            raise ValueError("point-map GT archive requires frame IDs")
        frame_ids = np.asarray(candidates[0][1])
        if frame_ids.ndim != 1 or frame_ids.dtype.kind not in "iu":
            raise ValueError("point-map GT frame IDs must be a one-dimensional integer array")
        frame_ids = frame_ids.astype(np.int64, copy=False)
        if any(not np.array_equal(frame_ids, np.asarray(other)) for _, other in candidates[1:]):
            raise ValueError("point-map GT frame ID sources disagree")
        if len(set(frame_ids.tolist())) != len(frame_ids):
            raise ValueError("point-map GT frame IDs must be unique")
        if point_maps.ndim != 4 or point_maps.shape[-1] != 3:
            raise ValueError("point-map GT must have shape (N,H,W,3) for all frames")
        if point_maps.shape[1:3] != expected:
            raise ValueError(
                f"point-map GT spatial shape {point_maps.shape[1:3]} does not match {expected}"
            )
        if valid_mask.ndim != 3 or valid_mask.shape != point_maps.shape[:3]:
            raise ValueError("point-map GT valid_mask shape does not match point-map GT frames")
        if valid_mask.dtype.kind != "b":
            raise ValueError("point-map GT valid_mask must be boolean")
        if point_maps.shape[0] != len(frame_ids):
            raise ValueError("point-map GT frame IDs do not match point-map GT frames")
        if np.any(valid_mask) and not np.isfinite(point_maps[valid_mask]).all():
            raise ValueError("point-map GT contains non-finite selected values")
        positions = {int(frame_id): index for index, frame_id in enumerate(frame_ids.tolist())}
        requested = scene.selection.source_frame_ids
        missing = [frame_id for frame_id in requested if frame_id not in positions]
        if missing:
            raise ValueError(f"point-map GT is missing selected frame IDs: {missing}")
        order = np.asarray([positions[frame_id] for frame_id in requested], dtype=np.int64)
        return (
            np.asarray(point_maps[order], dtype=np.float32),
            np.asarray(valid_mask[order], dtype=bool),
            np.asarray(requested, dtype=np.int64),
        )


def _raw_scene_root(scene: ResolvedScene) -> Path:
    if not scene.source_images:
        raise ValueError("raw point-cloud preparation requires source images")
    image = scene.source_images[0].resolve(strict=True)
    if scene.dataset is DatasetKind.NRGBD:
        return image.parent.parent.parent
    parts = Path(scene.scene).parts
    scene_dir = image.parent
    return scene_dir.parents[len(parts) - 1]


def _raw_nrgbd_gt(scene: ResolvedScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Keep the existing dataset's geometry conversion as the single source of
    # truth.  Construction is intentionally metadata-limited to one scene and
    # the call requests exactly the already-selected IDs.
    from datasets.nrgbd import NRGBD
    from utils.geometry import closed_form_inverse_se3

    root = _raw_scene_root(scene)
    pose_path = root / scene.scene / "poses.txt"
    values = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 4, 4)
    if not np.isfinite(values).all():
        raise ValueError("NeuralRGBD poses must contain finite values")
    values = values.astype(np.float32, copy=True)
    values[:, :, 1:3] *= -1.0
    extrinsics = closed_form_inverse_se3(values)[:, :3, :]
    data_set = NRGBD.__new__(NRGBD)
    data_set.NRGBD_DIR = str(root)
    data_set.load_img_size = 518
    data_set.metadata = {scene.scene: extrinsics}
    data_set.sequence_list = [scene.scene]
    ids = np.asarray(scene.selection.source_frame_ids, dtype=np.int64)
    data = data_set.get_data(sequence_name=scene.scene, ids=ids)
    return _validate_raw_dataset_output(scene, data)


def _project_7scenes_depth(source: Path, target: Path) -> None:
    """Create the projected-depth view expected by SevenScenes.get_data()."""

    import imageio.v2 as imageio

    transform = np.asarray(
        [
            [0.9999651801256764, 0.0026765126468950344, -0.00790410123130009, -0.025558943178152542],
            [-0.00274093112813167, 0.9999630280302759, -0.008150452077801328, 0.00010109633168066106],
            [0.007881994213044533, 0.008171832877189064, 0.9999355455801404, 0.002031832172948704],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    depth = imageio.imread(source).astype(np.float64) / 1000.0
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    z = depth.reshape(-1)
    valid = (z > 0.0) & (z < 100.0)
    z = z[valid]
    x_pixel = (xx + 0.5).reshape(-1)[valid]
    y_pixel = (yy + 0.5).reshape(-1)[valid]
    points = np.vstack(
        (
            (x_pixel - width / 2.0) / 585.0 * z,
            (y_pixel - height / 2.0) / 585.0 * z,
            z,
            np.ones_like(z),
        )
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        points = transform @ points
    z_rgb = points[2]
    finite = np.isfinite(z_rgb) & (z_rgb > 0)
    points = points[:, finite]
    z_rgb = z_rgb[finite]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = np.rint(points[0] / z_rgb * 525.0 + 320.0).astype(np.int64)
        v = np.rint(points[1] / z_rgb * 525.0 + 240.0).astype(np.int64)
    inside = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    registered = np.full((480, 640), 2000.0, dtype=np.float64)
    np.minimum.at(registered, (v[inside], u[inside]), z_rgb[inside])
    registered[registered > 1000.0] = 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(target, (registered * 1000.0).astype(np.uint16))


def _raw_sevenscenes_gt(
    scene: ResolvedScene,
    workspace: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from datasets.sevenscenes import SevenScenes

    root = _raw_scene_root(scene)
    scene_dir = root / scene.scene
    ids = np.asarray(scene.selection.source_frame_ids, dtype=np.int64)
    workspace = workspace.resolve(strict=False)
    workspace.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".sevenscenes-depth-", dir=str(workspace)))
    try:
        scratch_scene = scratch / scene.scene
        for frame_id in ids.tolist():
            base = f"frame-{frame_id:06d}"
            for suffix in (".color.png", ".pose.txt"):
                target = scratch_scene / f"{base}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.path.relpath(scene_dir / f"{base}{suffix}", target.parent))
            projected = scratch_scene / f"{base}.depth.proj.png"
            raw_depth = scene_dir / f"{base}.depth.png"
            existing_projected = scene_dir / f"{base}.depth.proj.png"
            if existing_projected.is_file():
                projected.symlink_to(os.path.relpath(existing_projected, projected.parent))
            else:
                _project_7scenes_depth(raw_depth, projected)
        data_set = SevenScenes.__new__(SevenScenes)
        data_set.SEVENSCENES_DIR = str(scratch)
        data_set.load_img_size = 518
        data_set.metadata = {scene.scene: int(max(ids)) + 1}
        data_set.sequence_list = [scene.scene]
        data = data_set.get_data(sequence_name=scene.scene, ids=ids)
        return _validate_raw_dataset_output(scene, data)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _validate_raw_dataset_output(
    scene: ResolvedScene,
    data: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"ind", "image_paths", "pointclouds", "valid_mask"}
    if not required.issubset(data):
        raise ValueError("raw point-cloud preparation returned incomplete dataset data")
    requested = np.asarray(scene.selection.source_frame_ids, dtype=np.int64)
    raw_ind = data["ind"]
    if hasattr(raw_ind, "detach"):
        raw_ind = raw_ind.detach().cpu().numpy()
    ind = np.asarray(raw_ind)
    if (
        ind.ndim != 1
        or ind.dtype.kind not in "iu"
        or not np.array_equal(ind.astype(np.int64), requested)
    ):
        raise ValueError("raw point-cloud preparation returned unexpected frame IDs")
    image_paths = tuple(Path(item).resolve(strict=True) for item in data["image_paths"])
    expected_paths = tuple(path.resolve(strict=True) for path in scene.source_images)
    if image_paths != expected_paths:
        raise ValueError("raw point-cloud preparation returned unexpected image paths")
    point_maps = np.asarray(data["pointclouds"])
    valid_mask = np.asarray(data["valid_mask"])
    if point_maps.ndim != 4 or point_maps.shape[-1] != 3:
        raise ValueError("point-map GT must have shape (N,H,W,3) for all frames")
    if scene.expected_gt_shape != (392, 518) or point_maps.shape[1:3] != (392, 518):
        raise ValueError("point-map GT spatial shape does not match (392, 518)")
    if valid_mask.shape != point_maps.shape[:3]:
        raise ValueError("point-map GT valid_mask shape does not match point-map GT frames")
    valid_mask = valid_mask.astype(bool, copy=False)
    if np.any(valid_mask) and not np.isfinite(point_maps[valid_mask]).all():
        raise ValueError("point-map GT contains non-finite selected values")
    return (
        point_maps.astype(np.float32, copy=False),
        valid_mask,
        requested,
    )


def prepare_pointcloud_gt(
    scene: ResolvedScene,
    destination: Path,
) -> tuple[Path, tuple[int, int, int, int]]:
    """Prepare campaign-owned point-map GT selected by the exact source IDs."""

    if not isinstance(scene, ResolvedScene):
        raise ValueError("point-cloud preparation requires ResolvedScene")
    target = Path(destination)
    if scene.evaluation_kind is not EvaluationKind.POINTCLOUD:
        raise ValueError("point-cloud GT preparation requires point-cloud evaluation")
    if scene.prepared_gt_path is not None:
        point_maps, valid_mask, frame_ids = _validate_external_gt(
            scene, scene.prepared_gt_path.resolve(strict=True)
        )
    elif scene.dataset is DatasetKind.SEVEN_SCENES:
        point_maps, valid_mask, frame_ids = _raw_sevenscenes_gt(
            scene, target.parent / ".raw-pointcloud-work"
        )
    elif scene.dataset is DatasetKind.NRGBD:
        point_maps, valid_mask, frame_ids = _raw_nrgbd_gt(scene)
    else:
        raise ValueError(f"raw point-cloud preparation is unsupported for {scene.dataset.value}")
    _atomic_npz(
        target,
        {
            "point_maps": point_maps,
            "valid_mask": valid_mask,
            "frame_ids": frame_ids,
        },
    )
    shape = tuple(int(value) for value in point_maps.shape)
    if len(shape) != 4:
        raise ValueError("prepared point-map GT has invalid shape")
    return target, (shape[0], shape[1], shape[2], shape[3])


def _base_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in manifest.items() if key != "files"}


def _manifest_error(message: str, path: Path | None = None) -> ValueError:
    if path is None:
        return ValueError(message)
    return ValueError(f"{message}: {path}")


def _validate_manifest_files(directory: Path, manifest: Mapping[str, object]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise _manifest_error("staging file hash mismatch", directory / "staging.json")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise _manifest_error("staging file hash mismatch", directory / "staging.json")
        candidate = directory / relative
        try:
            require_descendant(candidate, directory)
        except ValueError as exc:
            raise _manifest_error("staging file hash mismatch", candidate) from exc
        if not candidate.is_file() or _sha256_file(candidate) != expected:
            raise _manifest_error("staging file hash mismatch", candidate)


def _selection_from_manifest(manifest: Mapping[str, object]) -> ResolvedFrameSelection:
    selection = manifest.get("selection")
    source_ids = manifest.get("source_frame_ids")
    if not isinstance(selection, dict) or not isinstance(source_ids, list):
        raise _manifest_error("staging manifest mismatch")
    try:
        if (
            not all(type(selection.get(name)) is int for name in ("start", "stop", "stride"))
            or any(type(item) is not int for item in source_ids)
        ):
            raise ValueError("selection fields must be integers")
        return ResolvedFrameSelection(
            selection["start"],
            selection["stop"],
            selection["stride"],
            tuple(source_ids),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _manifest_error("staging manifest mismatch") from exc


def load_valid_staging(
    path: str | Path,
    expected_payload: Mapping[str, object],
) -> StagedScene:
    """Validate and load a complete, reusable staging directory."""

    candidate = Path(path)
    directory = candidate.parent if candidate.is_file() else candidate
    manifest_path = directory / "staging.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _manifest_error("staging file hash mismatch", manifest_path) from exc
    if not isinstance(manifest, dict):
        raise _manifest_error("staging manifest mismatch", manifest_path)
    expected_base = _base_payload(expected_payload)
    actual_base = _base_payload(manifest)
    if actual_base != expected_base:
        raise _manifest_error("staging manifest mismatch", manifest_path)
    _validate_manifest_files(directory, manifest)

    source_ids = manifest.get("source_frame_ids")
    images = manifest.get("images")
    approved_raw = manifest.get("approved_data_root")
    if not isinstance(source_ids, list) or not isinstance(images, list) or not isinstance(approved_raw, str):
        raise _manifest_error("staging manifest mismatch", manifest_path)
    approved = Path(approved_raw).resolve(strict=False)
    if len(images) != len(source_ids):
        raise _manifest_error("staging file hash mismatch", manifest_path)
    image_dir = directory / "images"
    if not image_dir.is_dir() or image_dir.is_symlink():
        raise _manifest_error("staging file hash mismatch", image_dir)
    for ordinal, record in enumerate(images):
        if not isinstance(record, dict):
            raise _manifest_error("staging manifest mismatch", manifest_path)
        expected_name = f"{ordinal:06d}.png"
        link = image_dir / expected_name
        if not link.is_symlink():
            raise _manifest_error("staging file hash mismatch", link)
        try:
            target = link.resolve(strict=True)
            require_descendant(target, approved)
        except (OSError, ValueError) as exc:
            raise _manifest_error("staging file hash mismatch", link) from exc
        if str(target) != record.get("source_path"):
            raise _manifest_error("staging file hash mismatch", link)
        if _sha256_file(target) != record.get("source_sha256"):
            raise _manifest_error("staging file hash mismatch", link)
        if record.get("source_frame_id") != source_ids[ordinal]:
            raise _manifest_error("staging manifest mismatch", manifest_path)
    extras = tuple(item for item in image_dir.iterdir() if item.name not in {f"{i:06d}.png" for i in range(len(images))})
    if extras:
        raise _manifest_error("staging file hash mismatch", extras[0])

    try:
        selection = _selection_from_manifest(manifest)
        dataset = DatasetKind(manifest["dataset"])
        scene_id = manifest["scene_id"]
        scene_name = manifest["scene"]
        slice_id = manifest["slice_id"]
    except (KeyError, TypeError, ValueError) as exc:
        raise _manifest_error("staging manifest mismatch", manifest_path) from exc
    pose_path = directory / "poses.txt"
    pointcloud_path = directory / "ground_truth.npz"
    poses = pose_path if pose_path.is_file() else None
    gt = pointcloud_path if pointcloud_path.is_file() else None
    digest = _canonical_sha256(manifest)
    return StagedScene(
        scene_id=scene_id,
        dataset=dataset,
        scene=scene_name,
        slice_id=slice_id,
        image_dir=image_dir,
        source_frame_ids=tuple(int(item) for item in source_ids),
        selection=selection,
        poses_path=poses,
        pointcloud_gt_path=gt,
        manifest_path=manifest_path,
        manifest_sha256=digest,
    )


def _write_staged_poses(scene: ResolvedScene, directory: Path) -> Path | None:
    if scene.poses_path is None:
        return None
    source = scene.poses_path.resolve(strict=True)
    try:
        require_descendant(source, scene.approved_data_root)
    except ValueError as exc:
        raise ValueError(
            f"poses source is outside approved data root: {source}"
        ) from exc
    poses = _read_kitti_pose_rows(source)
    max_id = max(scene.selection.source_frame_ids)
    if max_id >= len(poses):
        raise ValueError("KITTI poses do not cover selected source frame IDs")
    selected = poses[np.asarray(scene.selection.source_frame_ids, dtype=np.int64)]
    target = directory / "poses.txt"

    def writer(path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            np.savetxt(stream, selected, fmt="%.17g")
            stream.flush()
            os.fsync(stream.fileno())

    return _atomic_path(target, writer)


def stage_scene(scene: ResolvedScene, campaign_root: str | Path) -> StagedScene:
    """Build or safely reuse one campaign-owned staged scene."""

    if not isinstance(scene, ResolvedScene):
        raise ValueError("staging requires ResolvedScene")
    root = Path(campaign_root).resolve(strict=False)
    payload, _ = preview_staging_manifest(scene)
    dataset_root = require_descendant(root / "prepared" / scene.dataset.value, root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    final = require_descendant(dataset_root / scene.slice_id, root)
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_dir():
            raise ValueError(f"staging path is not an owned directory: {final}")
        return load_valid_staging(final, payload)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f"{scene.slice_id}.tmp-{os.getpid()}-", dir=str(dataset_root)
        )
    )
    try:
        image_dir = temporary / "images"
        for ordinal, source in enumerate(scene.source_images):
            _relative_link(
                source,
                image_dir / f"{ordinal:06d}.png",
                scene.approved_data_root,
            )
        _atomic_npy(
            temporary / "source_frame_ids.npy",
            np.asarray(scene.selection.source_frame_ids, dtype=np.int64),
        )
        _write_staged_poses(scene, temporary)
        if scene.evaluation_kind is EvaluationKind.POINTCLOUD:
            prepare_pointcloud_gt(scene, temporary / "ground_truth.npz")

        generated: dict[str, str] = {}
        for name in ("source_frame_ids.npy", "poses.txt", "ground_truth.npz"):
            path = temporary / name
            if path.is_file():
                generated[name] = _sha256_file(path)
        manifest = dict(payload)
        manifest["files"] = generated
        _atomic_json(temporary / "staging.json", manifest)

        if final.exists() or final.is_symlink():
            # Another process won the atomic publication race.  Never replace
            # or clean an unknown directory; validate the winner instead.
            guarded_remove(temporary, root)
            return load_valid_staging(final, payload)
        os.replace(temporary, final)
        temporary = Path()
        return load_valid_staging(final, payload)
    finally:
        if str(temporary) not in {"", "."} and temporary.exists():
            guarded_remove(temporary, root)


__all__ = [
    "STAGING_SCHEMA_VERSION",
    "StagedScene",
    "guarded_remove",
    "load_valid_staging",
    "prepare_pointcloud_gt",
    "preview_staging_manifest",
    "require_descendant",
    "stage_scene",
]
