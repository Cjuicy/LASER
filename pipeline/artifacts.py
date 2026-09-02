from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch

from pipeline.config import ReconstructionMode, SegmentationMethod


ARTIFACT_SCHEMA_VERSION = 1
_TENSOR_FILES = ("trajectory.pt", "pointmap.pt", "confidence.pt")


def _validate_json_value(value: object, context: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} must contain finite JSON scalars")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} mapping keys must be strings")
            _validate_json_value(item, f"{context}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{context}[{index}]")
        return
    raise ValueError(f"{context} must contain JSON scalars, lists, or mappings")


def _immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic value must be a mapping")
    _validate_json_value(value, "diagnostics")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    stage_timings_ms: Mapping[str, float]
    segmentation_summaries: tuple[Mapping[str, object], ...]
    candidate_count: int
    constraint_count: int
    mode_scalars: Mapping[str, int | float]

    def __post_init__(self) -> None:
        timings = {
            str(name): float(value)
            for name, value in self.stage_timings_ms.items()
        }
        if any(not math.isfinite(value) or value < 0 for value in timings.values()):
            raise ValueError("stage timings must be finite and non-negative")
        for name in ("candidate_count", "constraint_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        scalars = dict(self.mode_scalars)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in scalars.values()
        ):
            raise ValueError("mode diagnostics must contain finite scalars")
        object.__setattr__(self, "stage_timings_ms", MappingProxyType(timings))
        object.__setattr__(
            self,
            "segmentation_summaries",
            tuple(_immutable_mapping(item) for item in self.segmentation_summaries),
        )
        object.__setattr__(self, "mode_scalars", MappingProxyType(scalars))

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_timings_ms": dict(self.stage_timings_ms),
            "segmentation_summaries": [
                dict(item) for item in self.segmentation_summaries
            ],
            "candidate_count": self.candidate_count,
            "constraint_count": self.constraint_count,
            "mode_scalars": dict(self.mode_scalars),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]):
        try:
            return cls(
                stage_timings_ms=payload["stage_timings_ms"],
                segmentation_summaries=tuple(payload["segmentation_summaries"]),
                candidate_count=payload["candidate_count"],
                constraint_count=payload["constraint_count"],
                mode_scalars=payload["mode_scalars"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("diagnostics payload is invalid") from exc


def _validate_frame_ids(frame_ids: tuple[int, ...], frame_count: int) -> None:
    if not isinstance(frame_ids, tuple):
        raise ValueError("frame_ids must be a tuple")
    if len(frame_ids) != frame_count:
        raise ValueError("frame_ids frame count does not match tensors")
    if any(
        isinstance(frame_id, bool)
        or not isinstance(frame_id, int)
        or frame_id < 0
        for frame_id in frame_ids
    ):
        raise ValueError("frame_ids must be non-negative integers")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique")


def _validate_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int | None, ...],
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        expected_text = ",".join("N" if item is None else str(item) for item in shape)
        raise ValueError(f"{name} must have shape ({expected_text})")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


@dataclass(frozen=True)
class TrajectoryEstimate:
    frame_ids: tuple[int, ...]
    camera_poses: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensor("camera_poses", self.camera_poses, (None, 4, 4))
        _validate_frame_ids(self.frame_ids, int(self.camera_poses.shape[0]))


@dataclass(frozen=True)
class PointMapEstimate:
    frame_ids: tuple[int, ...]
    global_points: torch.Tensor
    confidence: torch.Tensor
    reconstruction_mode: ReconstructionMode

    def __post_init__(self) -> None:
        _validate_tensor("global_points", self.global_points, (None, None, None, 3))
        _validate_tensor("confidence", self.confidence, (None, None, None))
        _validate_frame_ids(self.frame_ids, int(self.global_points.shape[0]))
        if self.confidence.shape != self.global_points.shape[:-1]:
            raise ValueError("confidence shape must match global_points")
        if not isinstance(self.reconstruction_mode, ReconstructionMode):
            try:
                mode = ReconstructionMode(self.reconstruction_mode)
            except (TypeError, ValueError) as exc:
                raise ValueError("reconstruction_mode is invalid") from exc
            object.__setattr__(self, "reconstruction_mode", mode)


@dataclass(frozen=True)
class ReconstructionArtifact:
    schema_version: int
    frame_ids: tuple[int, ...]
    local_points: torch.Tensor
    global_points: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    segmentation_method: SegmentationMethod
    reconstruction_mode: ReconstructionMode
    prediction_key: str
    diagnostics: ReconstructionDiagnostics

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact schema_version must be {ARTIFACT_SCHEMA_VERSION}"
            )
        _validate_tensor("local_points", self.local_points, (None, None, None, 3))
        _validate_tensor("global_points", self.global_points, (None, None, None, 3))
        _validate_tensor("camera_poses", self.camera_poses, (None, 4, 4))
        _validate_tensor("confidence", self.confidence, (None, None, None))
        frame_count = int(self.local_points.shape[0])
        _validate_frame_ids(self.frame_ids, frame_count)
        if (
            self.global_points.shape != self.local_points.shape
            or self.camera_poses.shape[0] != frame_count
            or self.confidence.shape != self.local_points.shape[:-1]
        ):
            raise ValueError("artifact frame count or dense tensor shapes differ")
        if not isinstance(self.segmentation_method, SegmentationMethod):
            try:
                method = SegmentationMethod(self.segmentation_method)
            except (TypeError, ValueError) as exc:
                raise ValueError("segmentation_method is invalid") from exc
            object.__setattr__(self, "segmentation_method", method)
        if not isinstance(self.reconstruction_mode, ReconstructionMode):
            try:
                mode = ReconstructionMode(self.reconstruction_mode)
            except (TypeError, ValueError) as exc:
                raise ValueError("reconstruction_mode is invalid") from exc
            object.__setattr__(self, "reconstruction_mode", mode)
        if not isinstance(self.prediction_key, str) or not self.prediction_key:
            raise ValueError("prediction_key must be a non-empty string")
        if not isinstance(self.diagnostics, ReconstructionDiagnostics):
            raise ValueError("diagnostics must be ReconstructionDiagnostics")

    @property
    def trajectory(self) -> TrajectoryEstimate:
        return TrajectoryEstimate(self.frame_ids, self.camera_poses)

    @property
    def pointmap(self) -> PointMapEstimate:
        return PointMapEstimate(
            self.frame_ids,
            self.global_points,
            self.confidence,
            self.reconstruction_mode,
        )


@dataclass(frozen=True)
class StagedReconstructionArtifacts:
    stage1: ReconstructionArtifact
    stage2: ReconstructionArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.stage1, ReconstructionArtifact) or not isinstance(
            self.stage2,
            ReconstructionArtifact,
        ):
            raise ValueError("staged results require reconstruction artifacts")
        if self.stage1.reconstruction_mode is not ReconstructionMode.TRADITIONAL:
            raise ValueError("stage1 artifact must use traditional mode")
        if (
            self.stage2.reconstruction_mode
            is not ReconstructionMode.TRADITIONAL_SECOND_GLOBAL
        ):
            raise ValueError(
                "stage2 artifact must use traditional_second_global mode"
            )
        if self.stage1.frame_ids != self.stage2.frame_ids:
            raise ValueError("staged artifact frame_ids must match")
        if self.stage1.prediction_key != self.stage2.prediction_key:
            raise ValueError("staged artifact prediction keys must match")

    @property
    def primary(self) -> ReconstructionArtifact:
        return self.stage2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(dict(payload), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_text(path: Path, text: str) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _tensor_description(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
    }


def write_reconstruction_artifact(
    artifact: ReconstructionArtifact,
    output_dir: str | Path,
    *,
    resolved_yaml: str,
    config_sha256: str,
    checkpoint_sha256: str,
    git_commit: str,
) -> Path:
    if not isinstance(artifact, ReconstructionArtifact):
        raise ValueError("artifact must be ReconstructionArtifact")
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"artifact directory already exists: {output}") from exc

    payloads = {
        "trajectory.pt": {
            "frame_ids": artifact.frame_ids,
            "camera_poses": artifact.camera_poses,
        },
        "pointmap.pt": {
            "frame_ids": artifact.frame_ids,
            "local_points": artifact.local_points,
            "global_points": artifact.global_points,
            "reconstruction_mode": artifact.reconstruction_mode.value,
        },
        "confidence.pt": {"confidence": artifact.confidence},
    }
    for filename, payload in payloads.items():
        _atomic_torch_save(output / filename, payload)
    _atomic_text(output / "resolved_reconstruction.yaml", resolved_yaml)
    _atomic_text(
        output / "diagnostics.json",
        json.dumps(
            artifact.diagnostics.to_payload(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )

    tensor_manifest = {
        "trajectory.pt": {
            "sha256": _sha256(output / "trajectory.pt"),
            "camera_poses": _tensor_description(artifact.camera_poses),
        },
        "pointmap.pt": {
            "sha256": _sha256(output / "pointmap.pt"),
            "local_points": _tensor_description(artifact.local_points),
            "global_points": _tensor_description(artifact.global_points),
        },
        "confidence.pt": {
            "sha256": _sha256(output / "confidence.pt"),
            "confidence": _tensor_description(artifact.confidence),
        },
    }
    manifest = {
        "schema_version": artifact.schema_version,
        "frame_ids": list(artifact.frame_ids),
        "segmentation_method": artifact.segmentation_method.value,
        "reconstruction_mode": artifact.reconstruction_mode.value,
        "prediction_key": artifact.prediction_key,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "git_commit": git_commit,
        "resolved_yaml_sha256": _sha256(output / "resolved_reconstruction.yaml"),
        "diagnostics_sha256": _sha256(output / "diagnostics.json"),
        "tensors": tensor_manifest,
    }
    _atomic_text(
        output / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return output


def write_staged_reconstruction_artifacts(
    staged: StagedReconstructionArtifacts,
    output_dir: str | Path,
    *,
    resolved_yaml: str,
    config_sha256: str,
    checkpoint_sha256: str,
    git_commit: str,
) -> Mapping[str, Path]:
    if not isinstance(staged, StagedReconstructionArtifacts):
        raise ValueError("staged result must be StagedReconstructionArtifacts")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"artifact directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "resolved_yaml": resolved_yaml,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "git_commit": git_commit,
    }
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=f".{output.name}.staged.",
    ) as temporary:
        staged_root = Path(temporary) / "artifact"
        write_reconstruction_artifact(
            staged.stage2,
            staged_root,
            **metadata,
        )
        write_reconstruction_artifact(
            staged.stage1,
            staged_root / "stage1",
            **metadata,
        )
        staged_root.replace(output)
    return MappingProxyType(
        {
            "stage1": output / "stage1",
            "stage2": output,
        }
    )


def _load_manifest(output: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is missing or invalid") from exc
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact manifest schema version mismatch")
    if not isinstance(manifest.get("tensors"), dict):
        raise ValueError("artifact manifest tensor table is invalid")
    return manifest


def _load_tensor_file(
    output: Path,
    manifest: Mapping[str, object],
    filename: str,
) -> Mapping[str, object]:
    try:
        record = manifest["tensors"][filename]
        expected_digest = record["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"artifact manifest is missing {filename}") from exc
    path = output / filename
    if not path.is_file() or _sha256(path) != expected_digest:
        raise ValueError(f"{filename} digest mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{filename} could not be loaded") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{filename} payload is invalid")
    for name, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        metadata = record.get(name)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{filename} is missing {name} metadata")
        actual = _tensor_description(value)
        if (
            metadata.get("shape") != actual["shape"]
            or metadata.get("dtype") != actual["dtype"]
        ):
            raise ValueError(f"{name} metadata mismatch in {filename}")
    return payload


def _manifest_frame_ids(manifest: Mapping[str, object]) -> tuple[int, ...]:
    try:
        return tuple(manifest["frame_ids"])
    except (KeyError, TypeError) as exc:
        raise ValueError("artifact manifest frame_ids are invalid") from exc


def load_trajectory_estimate(output_dir: str | Path) -> TrajectoryEstimate:
    output = Path(output_dir)
    manifest = _load_manifest(output)
    payload = _load_tensor_file(output, manifest, "trajectory.pt")
    frame_ids = _manifest_frame_ids(manifest)
    if tuple(payload.get("frame_ids", ())) != frame_ids:
        raise ValueError("trajectory.pt frame_ids mismatch")
    return TrajectoryEstimate(frame_ids, payload.get("camera_poses"))


def load_pointmap_estimate(output_dir: str | Path) -> PointMapEstimate:
    output = Path(output_dir)
    manifest = _load_manifest(output)
    pointmap = _load_tensor_file(output, manifest, "pointmap.pt")
    confidence = _load_tensor_file(output, manifest, "confidence.pt")
    frame_ids = _manifest_frame_ids(manifest)
    if tuple(pointmap.get("frame_ids", ())) != frame_ids:
        raise ValueError("pointmap.pt frame_ids mismatch")
    if pointmap.get("reconstruction_mode") != manifest.get("reconstruction_mode"):
        raise ValueError("pointmap.pt reconstruction mode mismatch")
    return PointMapEstimate(
        frame_ids=frame_ids,
        global_points=pointmap.get("global_points"),
        confidence=confidence.get("confidence"),
        reconstruction_mode=manifest.get("reconstruction_mode"),
    )


def load_reconstruction_artifact(output_dir: str | Path) -> ReconstructionArtifact:
    output = Path(output_dir)
    manifest = _load_manifest(output)
    trajectory = _load_tensor_file(output, manifest, "trajectory.pt")
    pointmap = _load_tensor_file(output, manifest, "pointmap.pt")
    confidence = _load_tensor_file(output, manifest, "confidence.pt")
    frame_ids = _manifest_frame_ids(manifest)
    if (
        tuple(trajectory.get("frame_ids", ())) != frame_ids
        or tuple(pointmap.get("frame_ids", ())) != frame_ids
    ):
        raise ValueError("artifact tensor frame_ids mismatch")
    try:
        diagnostics_payload = json.loads(
            (output / "diagnostics.json").read_text("utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("artifact diagnostics are missing or invalid") from exc
    if _sha256(output / "diagnostics.json") != manifest.get("diagnostics_sha256"):
        raise ValueError("diagnostics.json digest mismatch")
    return ReconstructionArtifact(
        schema_version=manifest["schema_version"],
        frame_ids=frame_ids,
        local_points=pointmap.get("local_points"),
        global_points=pointmap.get("global_points"),
        camera_poses=trajectory.get("camera_poses"),
        confidence=confidence.get("confidence"),
        segmentation_method=manifest.get("segmentation_method"),
        reconstruction_mode=manifest.get("reconstruction_mode"),
        prediction_key=manifest.get("prediction_key"),
        diagnostics=ReconstructionDiagnostics.from_payload(diagnostics_payload),
    )
