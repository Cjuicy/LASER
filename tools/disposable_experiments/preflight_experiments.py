#!/usr/bin/env python3
"""Disposable orchestration helpers for the cloud evaluation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SOURCE_BASELINE = "cfe26f8b57341a9320eae2ef71532a353654d1a6"
SOURCE_FILE_SHA256 = {
    "configs/evaluation/ate.yaml": "82f49312c5c3422e33abef11cf40e8e263606c4ed77a469995b64423a724dec4",
    "configs/evaluation/pointcloud.yaml": "249eab30335bd7dbc6622a31e67dead9a64f16e071a0c07cde1e6949ce938f86",
    "configs/reconstruction/pi3_laser.yaml": "4c85b1eae9216daa6aac6d4a25a51e3848963c8d498543828b36dcb266df45c9",
    "configs/reconstruction/pi3_laser_no_loop.yaml": "495898e1dbded79fb090ba3c52871ac128afbf717a0f295dc2a566d28f48e7ec",
    "datasets/nrgbd.py": "043e84d979e943a3c78426e050c7aacdf6337efa66e29099e732abb79f16cfe2",
    "datasets/sevenscenes.py": "df571b5eb45944098c5e6dace2263f2aa3e2d1959ad9dd49a1432e92baf867bf",
    "datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json": "e9954bfcf4b4a3273224e8375d468638e1fe4d7b6d926ff32147367bb4574008",
    "datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json": "f18f2143f8a373727aa4d7043b779b77639354fda80523cdc2a139164ddc33ba",
    "evaluate_ate.py": "384a904f52923664edcc1f53650ba3c38fce7890c9a52c5ceb7bf1895363c58d",
    "evaluation/pointcloud/evaluator.py": "4678486dd584bb78a5a6747d8bad2f5fe0cf00ed818a83cfc21dbf925da92ef0",
    "evaluation/pointcloud/geometry_metrics.py": "c6bf89a781d03e5f1fc08522be7a602ba561a691906159a5251763cc57591971",
    "pipeline/artifacts.py": "ea404633497f42699f0de8ee59bea1b232f2e3b335ae322686200457d690d3b7",
    "pipeline/runner.py": "f55a203e02ef78da94852e44eb1fd68eaa58c18a48519da52aefc54c19b728e6",
    "run_reconstruction.py": "13ef4d1d00958bf3d4410b0b1380b2672c6f6e1e31140a353ae653458f35ae4e",
}
KITTI_SEQUENCES = tuple(f"{index:02d}" for index in range(11))

POINTCLOUD_METHODS = {
    "depth": ("depth", "none", "no_loop"),
    "geometry": ("geometry", "none", "no_loop"),
    "atomic-original": ("atomic", "none", "no_loop"),
    "atomic-split-assisted": ("atomic", "conservative", "no_loop"),
    "atomic-split-no-assisted": ("atomic", "normal_only", "no_loop"),
}

ATE_METHODS = {
    "depth-traditional": ("depth", "none", "traditional"),
    "depth-corrected": ("depth", "none", "corrected"),
    "geometry-corrected": ("geometry", "none", "corrected"),
    "atomic-original-corrected": ("atomic", "none", "corrected"),
    "atomic-split-assisted-corrected": ("atomic", "conservative", "corrected"),
    "atomic-split-no-assisted-corrected": ("atomic", "normal_only", "corrected"),
}

POINTCLOUD_METRICS = (
    "accuracy_mean_m",
    "accuracy_median_m",
    "completion_mean_m",
    "completion_median_m",
    "normal_consistency_mean",
    "normal_consistency_median",
    "chamfer_l1_m",
)
ATE_METRICS = (
    "ate_rmse_m",
    "rpe_translation_rmse_m",
    "rpe_rotation_rmse_deg",
    "matched_frame_count",
)
FSCORE_THRESHOLDS = (0.01, 0.02, 0.05)


def build_loop_detector_outside_artifact(
    config,
    *,
    output_path: str | Path,
    detector_factory=None,
):
    """Keep SALAD diagnostics beside, rather than inside, a pending artifact."""
    if detector_factory is None:
        from loop_closure.detection import SaladLoopDetector

        detector_factory = SaladLoopDetector
    requested = Path(output_path)
    artifact_directory = requested.parent
    diagnostic_path = artifact_directory.parent / (
        f"{artifact_directory.name}.loop_candidates.json"
    )
    return detector_factory(config, output_path=diagnostic_path)


def run_reconstruction_loop_safe(
    config_path: str | Path,
    overrides: Sequence[str],
) -> Path:
    """Run loop reconstruction without pre-creating the final artifact path."""
    from pipeline.config import load_pipeline_config
    from pipeline.runner import PipelineDependencies, PipelineRunner

    loaded = load_pipeline_config(config_path, tuple(overrides))
    dependencies = PipelineDependencies(
        build_loop_detector=build_loop_detector_outside_artifact,
    )
    runner = PipelineRunner(loaded, dependencies=dependencies)
    artifact = runner.run()
    diagnostics = artifact.diagnostics
    print(
        " ".join(
            (
                f"mode={artifact.reconstruction_mode.value}",
                f"segmentation={artifact.segmentation_method.value}",
                f"prediction_key={artifact.prediction_key}",
                f"window={loaded.config.window.size}",
                f"overlap={loaded.config.window.overlap}",
                f"config_hash={loaded.sha256}",
                f"frames={len(artifact.frame_ids)}",
                f"windows={int(diagnostics.mode_scalars.get('window_count', 0))}",
                f"artifact_dir={runner.artifact_dir}",
            )
        )
    )
    if runner.artifact_dir is None:  # pragma: no cover - runner owns this invariant
        raise RuntimeError("reconstruction did not write an artifact")
    return runner.artifact_dir


def _atomic_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def _read_json(path: str | Path) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON is missing or invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _finite_number(value: object, label: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    if integer and (not isinstance(value, int) or value < 1):
        raise ValueError(f"{label} must be a positive integer")


def build_method_overrides(method: str, evaluation: str) -> tuple[str, ...]:
    matrices = {"pointcloud": POINTCLOUD_METHODS, "ate": ATE_METHODS}
    try:
        segmentation, split_mode, reconstruction = matrices[evaluation][method]
    except KeyError as exc:
        raise ValueError(f"unknown {evaluation} method: {method!r}") from exc
    return (
        f"segmentation.method={segmentation}",
        f"segmentation.atomic.split_mode={split_mode}",
        f"reconstruction.mode={reconstruction}",
    )


def method_identity(
    *,
    evaluation: str,
    dataset: str,
    scene: str,
    method: str,
) -> dict[str, object]:
    matrices = {"pointcloud": POINTCLOUD_METHODS, "ate": ATE_METHODS}
    try:
        segmentation, split_mode, reconstruction = matrices[evaluation][method]
    except KeyError as exc:
        raise ValueError(f"unknown {evaluation} method: {method!r}") from exc
    pointcloud = evaluation == "pointcloud"
    return {
        "evaluation": evaluation,
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "segmentation_method": segmentation,
        "split_mode": split_mode,
        "reconstruction_mode": reconstruction,
        "sample_stride": 1,
        "window_size": 20 if pointcloud else 75,
        "overlap": 5 if pointcloud else 30,
        "registration_confidence_keep_ratio": 0.5,
    }


def load_valid_result(
    path: str | Path,
    expected_identity: Mapping[str, object],
) -> Mapping[str, object]:
    payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("result schema version mismatch")
    if payload.get("identity") != dict(expected_identity):
        raise ValueError("result identity mismatch")
    prediction_key = payload.get("prediction_key")
    if (
        not isinstance(prediction_key, str)
        or len(prediction_key) != 64
        or any(character not in "0123456789abcdef" for character in prediction_key)
    ):
        raise ValueError("result prediction_key must be SHA256 hex")
    _finite_number(payload.get("frame_count"), "frame_count", integer=True)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("result metrics must be an object")
    evaluation = expected_identity.get("evaluation")
    if evaluation == "pointcloud":
        for name in POINTCLOUD_METRICS:
            _finite_number(metrics.get(name), name)
        thresholds = metrics.get("thresholds")
        if not isinstance(thresholds, list) or len(thresholds) != 3:
            raise ValueError("point-cloud thresholds must contain 1/2/5 cm")
        for record, expected in zip(thresholds, FSCORE_THRESHOLDS, strict=True):
            if not isinstance(record, dict):
                raise ValueError("point-cloud threshold record must be an object")
            _finite_number(record.get("threshold_m"), "threshold_m")
            if not math.isclose(float(record["threshold_m"]), expected, abs_tol=1e-12):
                raise ValueError("point-cloud thresholds must be exactly 1/2/5 cm")
            for name in ("precision", "recall", "fscore"):
                _finite_number(record.get(name), name)
                if not 0 <= float(record[name]) <= 1:
                    raise ValueError(f"{name} must be in [0, 1]")
    elif evaluation == "ate":
        for name in ATE_METRICS:
            _finite_number(
                metrics.get(name),
                name,
                integer=name == "matched_frame_count",
            )
    else:
        raise ValueError(f"unsupported result evaluation: {evaluation!r}")
    return payload


def guarded_remove(path: str | Path, allowed_root: str | Path) -> None:
    target = Path(path).resolve(strict=False)
    root = Path(allowed_root).resolve(strict=False)
    if target == root or root not in target.parents:
        raise ValueError(f"cleanup target is outside allowed root: {target}")
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
    elif target.is_dir():
        shutil.rmtree(target)


def remove_prediction_keys(cache_root: str | Path, keys: Sequence[str]) -> None:
    root = Path(cache_root).resolve(strict=False)
    for key in tuple(dict.fromkeys(keys)):
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
        ):
            raise ValueError("prediction cache key must be SHA256 hex")
        guarded_remove(root / "v2" / key, root)


def _safe_scene_name(scene: str) -> str:
    safe = scene.replace("/", "__")
    if not safe or safe in {".", ".."} or "/" in safe or "\\" in safe:
        raise ValueError(f"unsafe scene name: {scene!r}")
    return safe


def _make_link(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve(strict=False) == source:
            return
        target.unlink()
    target.symlink_to(source)


_DEPTH_TO_RGB = None


def _project_7scenes_depth(source: Path, target: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np

    global _DEPTH_TO_RGB
    if _DEPTH_TO_RGB is None:
        _DEPTH_TO_RGB = np.asarray(
            [
                [0.9999651801256764, 0.0026765126468950343, -0.00790410123130009, -0.025558943178152542],
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
    points = _DEPTH_TO_RGB @ points
    z_rgb = points[2]
    finite = np.isfinite(z_rgb) & (z_rgb > 0)
    points = points[:, finite]
    z_rgb = z_rgb[finite]
    u = np.rint(points[0] / z_rgb * 525.0 + 320.0).astype(np.int64)
    v = np.rint(points[1] / z_rgb * 525.0 + 240.0).astype(np.int64)
    inside = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    registered = np.full((480, 640), 2000.0, dtype=np.float64)
    np.minimum.at(registered, (v[inside], u[inside]), z_rgb[inside])
    registered[registered > 1000.0] = 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(target, (registered * 1000.0).astype(np.uint16))


def _load_sequence_map(path: str | Path) -> dict[str, list[int]]:
    payload = _read_json(path)
    result = {}
    for scene, raw_ids in payload.items():
        if (
            not isinstance(scene, str)
            or not scene
            or not isinstance(raw_ids, list)
            or not raw_ids
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw_ids)
            or raw_ids != sorted(set(raw_ids))
        ):
            raise ValueError(f"invalid sequence map entry: {scene!r}")
        result[scene] = raw_ids
    if not result:
        raise ValueError("sequence map must not be empty")
    return result


def validate_pinned_sequence_map(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> dict[str, list[int]]:
    candidate = Path(path).resolve(strict=True)
    actual_sha256 = _sha256_file(candidate)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} sequence-map checksum mismatch: {actual_sha256}"
        )
    return _load_sequence_map(candidate)


def prepare_pointcloud(
    *,
    dataset: str,
    root: str | Path,
    sequence_map: str | Path,
    scene: str,
    output_root: str | Path,
) -> Path:
    import numpy as np

    mapping = _load_sequence_map(sequence_map)
    if scene not in mapping:
        raise ValueError(f"scene is not present in sequence map: {scene}")
    ids = mapping[scene]
    dataset_root = Path(root).resolve(strict=True)
    output = Path(output_root) / dataset / _safe_scene_name(scene)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)

    if dataset == "7scenes":
        from datasets.sevenscenes import SevenScenes

        mirror_root = output / "source"
        mirror_scene = mirror_root / scene
        for ordinal, frame_id in enumerate(ids):
            raw_scene = dataset_root / scene
            rgb = raw_scene / f"frame-{frame_id:06d}.color.png"
            depth = raw_scene / f"frame-{frame_id:06d}.depth.png"
            pose = raw_scene / f"frame-{frame_id:06d}.pose.txt"
            _make_link(rgb, mirror_scene / rgb.name)
            _make_link(pose, mirror_scene / pose.name)
            projected = mirror_scene / f"frame-{frame_id:06d}.depth.proj.png"
            if not projected.is_file():
                _project_7scenes_depth(depth, projected)
            _make_link(rgb, images / f"{ordinal:06d}.png")
        data_set = SevenScenes.__new__(SevenScenes)
        data_set.SEVENSCENES_DIR = str(mirror_root)
        data_set.load_img_size = 518
        data_set.metadata = {scene: int(max(ids)) + 1}
        data_set.sequence_list = [scene]
    elif dataset == "nrgbd":
        from datasets.nrgbd import NRGBD
        from utils.geometry import closed_form_inverse_se3

        poses = np.loadtxt(dataset_root / scene / "poses.txt").reshape(-1, 4, 4)
        if max(ids) >= len(poses) or not np.isfinite(poses).all():
            raise ValueError(f"NeuralRGBD poses are incomplete: {scene}")
        poses = poses.astype(np.float32, copy=True)
        poses[:, :, 1:3] *= -1.0
        extrinsics = closed_form_inverse_se3(poses)[:, :3, :]
        data_set = NRGBD.__new__(NRGBD)
        data_set.NRGBD_DIR = str(dataset_root)
        data_set.load_img_size = 518
        data_set.metadata = {scene: extrinsics}
        data_set.sequence_list = [scene]
    else:
        raise ValueError(f"unsupported point-cloud dataset: {dataset!r}")

    data = data_set.get_data(
        sequence_name=scene,
        ids=np.asarray(ids, dtype=np.int64),
    )
    if dataset == "nrgbd":
        for ordinal, source in enumerate(data["image_paths"]):
            _make_link(Path(source), images / f"{ordinal:06d}.png")
    point_maps = np.asarray(data["pointclouds"], dtype=np.float32)
    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    np.savez_compressed(
        output / "ground_truth.npz",
        point_maps=point_maps,
        valid_mask=valid_mask,
    )
    np.save(output / "frame_ids.npy", np.asarray(ids, dtype=np.int64))
    _atomic_json(
        output / "prepared.json",
        {
            "dataset": dataset,
            "scene": scene,
            "frame_count": len(ids),
            "source_root": str(dataset_root),
            "sequence_map": str(Path(sequence_map).resolve(strict=True)),
        },
    )
    return output


def _artifact_metadata(artifact_dir: str | Path) -> tuple[Mapping[str, object], object]:
    from omegaconf import OmegaConf

    artifact = Path(artifact_dir)
    manifest = _read_json(artifact / "manifest.json")
    try:
        resolved = OmegaConf.load(artifact / "resolved_reconstruction.yaml")
    except Exception as exc:
        raise ValueError("artifact resolved configuration is invalid") from exc
    return manifest, resolved


def _validate_artifact_identity(
    artifact_dir: str | Path,
    expected: Mapping[str, object],
) -> tuple[Mapping[str, object], object]:
    manifest, resolved = _artifact_metadata(artifact_dir)
    checks = {
        "segmentation_method": manifest.get("segmentation_method"),
        "reconstruction_mode": manifest.get("reconstruction_mode"),
        "split_mode": str(resolved.segmentation.atomic.split_mode).lower(),
        "sample_stride": int(resolved.input.sample_stride),
        "window_size": int(resolved.window.size),
        "overlap": int(resolved.window.overlap),
        "registration_confidence_keep_ratio": float(
            resolved.registration.confidence_keep_ratio
        ),
    }
    expected_checks = {name: expected[name] for name in checks}
    if checks != expected_checks:
        raise ValueError(f"artifact identity mismatch: {checks!r}")
    return manifest, resolved


def compact_pointcloud(
    *,
    artifact: str | Path,
    ground_truth: str | Path,
    evaluation_config: str | Path,
    output: str | Path,
    identity: Mapping[str, object],
) -> Path:
    import numpy as np

    from evaluation.pointcloud import (
        PointCloudGroundTruth,
        evaluate_point_maps,
        load_pointcloud_evaluation_config,
    )
    from pipeline.artifacts import load_pointmap_estimate

    manifest, _ = _validate_artifact_identity(artifact, identity)
    estimate = load_pointmap_estimate(artifact)
    with np.load(ground_truth, allow_pickle=False) as data:
        truth = PointCloudGroundTruth(data["point_maps"], data["valid_mask"])
    evaluation = evaluate_point_maps(
        estimate,
        truth,
        load_pointcloud_evaluation_config(evaluation_config),
    )
    payload = {
        "schema_version": 1,
        "source_baseline": SOURCE_BASELINE,
        "identity": dict(identity),
        "prediction_key": manifest["prediction_key"],
        "frame_count": len(estimate.frame_ids),
        "metrics": {
            **asdict(evaluation.primary),
            "chamfer_l1_m": evaluation.diagnostics.chamfer_l1_m,
            "thresholds": [asdict(item) for item in evaluation.diagnostics.thresholds],
        },
        "artifact_manifest_sha256": _sha256_file(Path(artifact) / "manifest.json"),
    }
    result = _atomic_json(output, payload)
    load_valid_result(result, identity)
    return result


def compact_ate(
    *,
    artifact: str | Path,
    evaluator_result: str | Path,
    output: str | Path,
    identity: Mapping[str, object],
) -> Path:
    manifest, _ = _validate_artifact_identity(artifact, identity)
    metrics = _read_json(evaluator_result)
    manifest_sha256 = _sha256_file(Path(artifact) / "manifest.json")
    if metrics.get("artifact_manifest_sha256") != manifest_sha256:
        raise ValueError("ATE evaluator artifact manifest checksum mismatch")
    frame_count = metrics.get("matched_frame_count")
    payload = {
        "schema_version": 1,
        "source_baseline": SOURCE_BASELINE,
        "identity": dict(identity),
        "prediction_key": manifest["prediction_key"],
        "frame_count": frame_count,
        "metrics": {name: metrics.get(name) for name in ATE_METRICS},
        "artifact_manifest_sha256": manifest_sha256,
    }
    result = _atomic_json(output, payload)
    load_valid_result(result, identity)
    return result


def validate_ate_artifact(
    artifact: str | Path,
    identity: Mapping[str, object],
) -> Path:
    from pipeline.artifacts import load_trajectory_estimate

    _validate_artifact_identity(artifact, identity)
    load_trajectory_estimate(artifact)
    return Path(artifact)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_image(path: Path, expected_size: tuple[int, int]) -> None:
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(f"image shape mismatch: {path}: {image.size}")
        image.verify()


def _audit_pointcloud_inputs(
    dataset: str,
    root: Path,
    mapping: Mapping[str, Sequence[int]],
) -> int:
    import numpy as np

    frames = 0
    for scene, ids in mapping.items():
        if dataset == "nrgbd":
            raw = np.loadtxt(root / scene / "poses.txt").reshape(-1, 4, 4)
            if not np.isfinite(raw).all() or max(ids) >= len(raw):
                raise ValueError(f"NeuralRGBD poses are incomplete: {scene}")
        for frame_id in ids:
            if dataset == "7scenes":
                base = root / scene
                _check_image(base / f"frame-{frame_id:06d}.color.png", (640, 480))
                _check_image(base / f"frame-{frame_id:06d}.depth.png", (640, 480))
                pose = np.loadtxt(base / f"frame-{frame_id:06d}.pose.txt")
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError(f"7-Scenes pose is invalid: {scene}/{frame_id}")
            else:
                base = root / scene
                _check_image(base / "images" / f"img{frame_id}.png", (640, 480))
                _check_image(base / "depth" / f"depth{frame_id}.png", (640, 480))
            frames += 1
    return frames


def run_preflight(arguments) -> Mapping[str, object]:
    import numpy as np
    import open3d
    import scipy
    import torch

    repository = Path(arguments.repository).resolve(strict=True)
    source_files = {}
    for relative_path, expected_sha256 in SOURCE_FILE_SHA256.items():
        path = repository / relative_path
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"source checksum mismatch: {relative_path}: {actual_sha256}"
            )
        source_files[relative_path] = actual_sha256
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in the selected LASER environment")
    seven_map = validate_pinned_sequence_map(
        arguments.seven_map,
        SOURCE_FILE_SHA256[
            "datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json"
        ],
        "7-Scenes kf10",
    )
    nrgbd_map = validate_pinned_sequence_map(
        arguments.nrgbd_map,
        SOURCE_FILE_SHA256[
            "datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json"
        ],
        "NeuralRGBD kf10",
    )
    if len(seven_map) != 18 or sum(map(len, seven_map.values())) != 1700:
        raise ValueError("7-Scenes kf10 map must contain 18 scenes and 1700 frames")
    if len(nrgbd_map) != 9 or sum(map(len, nrgbd_map.values())) != 1101:
        raise ValueError("NeuralRGBD kf10 map must contain 9 scenes and 1101 frames")
    weight_paths = {
        "pi3": Path(arguments.pi3_checkpoint),
        "salad": Path(arguments.salad_checkpoint),
        "dino": Path(arguments.dino_checkpoint),
    }
    weights = {}
    for name, path in weight_paths.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size < 1024 * 1024:
            raise ValueError(f"weight file is missing or too small: {resolved}")
        weights[name] = {"path": str(resolved), "bytes": resolved.stat().st_size}
    seven_frames = _audit_pointcloud_inputs(
        "7scenes", Path(arguments.seven_root).resolve(strict=True), seven_map
    )
    nrgbd_frames = _audit_pointcloud_inputs(
        "nrgbd", Path(arguments.nrgbd_root).resolve(strict=True), nrgbd_map
    )
    kitti_counts = {}
    for sequence in KITTI_SEQUENCES:
        sequence_root = Path(arguments.kitti_root).resolve(strict=True) / sequence
        images = sorted((sequence_root / "image_2").glob("*.png"))
        poses = np.loadtxt(sequence_root / "poses.txt")
        poses = np.atleast_2d(poses)
        if len(images) < 2 or poses.shape != (len(images), 12) or not np.isfinite(poses).all():
            raise ValueError(f"KITTI sequence is incomplete: {sequence}")
        kitti_counts[sequence] = len(images)
    free_bytes = shutil.disk_usage(repository).free
    if free_bytes < int(arguments.minimum_free_gb * 1024**3):
        raise ValueError(
            f"insufficient free disk: {free_bytes / 1024**3:.1f} GiB; "
            f"need {arguments.minimum_free_gb:.1f} GiB"
        )
    report = {
        "schema_version": 1,
        "source_baseline": SOURCE_BASELINE,
        "repository": str(repository),
        "python": sys.version,
        "runtime": {
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "open3d": open3d.__version__,
            "cuda_available": True,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "source_files": source_files,
        "seven_scenes": {"scenes": len(seven_map), "frames": seven_frames},
        "neuralrgbd": {"scenes": len(nrgbd_map), "frames": nrgbd_frames},
        "kitti": {"sequences": len(kitti_counts), "image_counts": kitti_counts},
        "weights": weights,
        "free_disk_bytes": free_bytes,
        "status": "ok",
    }
    return report


def _identity_from_arguments(arguments) -> dict[str, object]:
    return method_identity(
        evaluation=arguments.evaluation,
        dataset=arguments.dataset,
        scene=arguments.scene,
        method=arguments.method,
    )


def _add_identity_arguments(parser, *, evaluation_choices=("pointcloud", "ate")):
    parser.add_argument("--evaluation", choices=evaluation_choices, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", required=True)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument(
        "--seven-map",
        default=str(root / "datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json"),
    )
    plan.add_argument(
        "--nrgbd-map",
        default=str(root / "datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json"),
    )

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--repository", default=str(root))
    preflight.add_argument("--seven-root", default=str(root / "data/7-Scenes"))
    preflight.add_argument("--nrgbd-root", default=str(root / "data/NeuralRGBD"))
    preflight.add_argument("--kitti-root", default=str(root / "data/KITTI"))
    preflight.add_argument(
        "--seven-map",
        default=str(root / "datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json"),
    )
    preflight.add_argument(
        "--nrgbd-map",
        default=str(root / "datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json"),
    )
    preflight.add_argument("--pi3-checkpoint", default=str(root / "weights/PI3/model.safetensors"))
    preflight.add_argument("--salad-checkpoint", default=str(root / "weights/dino_salad.ckpt"))
    preflight.add_argument("--dino-checkpoint", default=str(root / "weights/dinov2_vitb14_pretrain.pth"))
    preflight.add_argument("--minimum-free-gb", type=float, default=20.0)
    preflight.add_argument(
        "--output", default=str(root / "outputs/disposable_experiments/preflight.json")
    )

    overrides = subparsers.add_parser("method-overrides")
    overrides.add_argument("--evaluation", choices=("pointcloud", "ate"), required=True)
    overrides.add_argument("--method", required=True)

    reconstruct = subparsers.add_parser("run-reconstruction-loop-safe")
    reconstruct.add_argument("--config", required=True)
    reconstruct.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )

    prepare = subparsers.add_parser("prepare-pointcloud")
    prepare.add_argument("--dataset", choices=("7scenes", "nrgbd"), required=True)
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--sequence-map", required=True)
    prepare.add_argument("--scene", required=True)
    prepare.add_argument("--output-root", required=True)

    compact_pc = subparsers.add_parser("compact-pointcloud")
    compact_pc.add_argument("--artifact", required=True)
    compact_pc.add_argument("--ground-truth", required=True)
    compact_pc.add_argument("--evaluation-config", required=True)
    compact_pc.add_argument("--output", required=True)
    _add_identity_arguments(compact_pc, evaluation_choices=("pointcloud",))

    compact_trajectory = subparsers.add_parser("compact-ate")
    compact_trajectory.add_argument("--artifact", required=True)
    compact_trajectory.add_argument("--evaluator-result", required=True)
    compact_trajectory.add_argument("--output", required=True)
    _add_identity_arguments(compact_trajectory, evaluation_choices=("ate",))

    artifact_ok = subparsers.add_parser("artifact-ok")
    artifact_ok.add_argument("--artifact", required=True)
    _add_identity_arguments(artifact_ok, evaluation_choices=("ate",))

    result_ok = subparsers.add_parser("result-ok")
    result_ok.add_argument("--path", required=True)
    _add_identity_arguments(result_ok)

    cleanup_artifact = subparsers.add_parser("cleanup-artifact")
    cleanup_artifact.add_argument("--path", required=True)
    cleanup_artifact.add_argument("--allowed-root", required=True)

    cleanup_cache = subparsers.add_parser("cleanup-scene-cache")
    cleanup_cache.add_argument("--cache-root", required=True)
    cleanup_cache.add_argument("--result", action="append", default=[], required=True)

    cleanup_prepared = subparsers.add_parser("cleanup-prepared")
    cleanup_prepared.add_argument("--path", required=True)
    cleanup_prepared.add_argument("--allowed-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "plan":
        pointcloud_scenes = len(_load_sequence_map(arguments.seven_map)) + len(
            _load_sequence_map(arguments.nrgbd_map)
        )
        payload = {
            "ate_method_runs": len(KITTI_SEQUENCES) * len(ATE_METHODS),
            "kitti_sequences": len(KITTI_SEQUENCES),
            "pointcloud_method_runs": pointcloud_scenes * len(POINTCLOUD_METHODS),
            "pointcloud_scenes": pointcloud_scenes,
            "total_method_runs": (
                pointcloud_scenes * len(POINTCLOUD_METHODS)
                + len(KITTI_SEQUENCES) * len(ATE_METHODS)
            ),
        }
        print(json.dumps(payload, sort_keys=True))
    elif arguments.command == "preflight":
        report = run_preflight(arguments)
        _atomic_json(arguments.output, report)
        print(json.dumps(report, sort_keys=True))
    elif arguments.command == "method-overrides":
        for override in build_method_overrides(arguments.method, arguments.evaluation):
            print(override)
    elif arguments.command == "run-reconstruction-loop-safe":
        run_reconstruction_loop_safe(arguments.config, arguments.overrides)
    elif arguments.command == "prepare-pointcloud":
        print(
            prepare_pointcloud(
                dataset=arguments.dataset,
                root=arguments.root,
                sequence_map=arguments.sequence_map,
                scene=arguments.scene,
                output_root=arguments.output_root,
            )
        )
    elif arguments.command == "compact-pointcloud":
        print(
            compact_pointcloud(
                artifact=arguments.artifact,
                ground_truth=arguments.ground_truth,
                evaluation_config=arguments.evaluation_config,
                output=arguments.output,
                identity=_identity_from_arguments(arguments),
            )
        )
    elif arguments.command == "compact-ate":
        print(
            compact_ate(
                artifact=arguments.artifact,
                evaluator_result=arguments.evaluator_result,
                output=arguments.output,
                identity=_identity_from_arguments(arguments),
            )
        )
    elif arguments.command == "artifact-ok":
        print(
            validate_ate_artifact(
                arguments.artifact,
                _identity_from_arguments(arguments),
            )
        )
    elif arguments.command == "result-ok":
        load_valid_result(arguments.path, _identity_from_arguments(arguments))
        print(arguments.path)
    elif arguments.command == "cleanup-artifact":
        guarded_remove(arguments.path, arguments.allowed_root)
    elif arguments.command == "cleanup-scene-cache":
        keys = []
        for result in arguments.result:
            payload = _read_json(result)
            keys.append(payload.get("prediction_key"))
        remove_prediction_keys(arguments.cache_root, keys)
    elif arguments.command == "cleanup-prepared":
        guarded_remove(arguments.path, arguments.allowed_root)
    else:  # pragma: no cover - argparse owns command validation
        raise AssertionError(arguments.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
