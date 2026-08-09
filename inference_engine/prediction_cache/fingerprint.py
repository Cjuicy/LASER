from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inference_engine.models.adapters import (
    MODEL_ADAPTER_CONTRACT_VERSION,
)
from pipeline.config import ModelConfig, ModelName
from pipeline.manifest import ImageManifest

from .types import PREDICTION_CACHE_SCHEMA_VERSION, WindowSpec


PREPROCESSING_CONTRACT_VERSION = (
    "load_and_preprocess_images:v1:width-518:patch-14"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PredictionFingerprint:
    key: str
    checkpoint_sha256: str
    image_manifest_sha256: str
    runtime_source_sha256: str
    canonical_payload: Mapping[str, object]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _update_delimited(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def digest_image_manifest(manifest: ImageManifest) -> str:
    if not isinstance(manifest, ImageManifest) or len(manifest) < 1:
        raise ValueError("prediction fingerprint requires an image manifest")
    digest = hashlib.sha256()
    _update_delimited(digest, str(len(manifest)).encode("ascii"))
    for path in manifest.paths:
        _update_delimited(
            digest,
            bytes.fromhex(sha256_file(path)),
        )
    return digest.hexdigest()


def runtime_source_paths(model_name: ModelName) -> tuple[Path, ...]:
    if model_name is not ModelName.PI3:
        raise ValueError(f"unsupported model name: {model_name!r}")
    relative_paths = (
        "inference_engine/models/adapters.py",
        "inference_engine/models/loader.py",
        "utils/load_fn.py",
        "pi3/models/pi3.py",
        "pi3/models/layers/attention.py",
        "pi3/models/layers/block.py",
        "pi3/models/layers/camera_head.py",
        "pi3/models/layers/pos_embed.py",
        "pi3/models/layers/transformer_head.py",
        "pi3/utils/geometry.py",
    )
    paths = [_REPOSITORY_ROOT / path for path in relative_paths]
    paths.extend(
        sorted(
            (_REPOSITORY_ROOT / "pi3/models/dinov2").rglob("*.py")
        )
    )
    return tuple(sorted(set(paths), key=lambda path: path.as_posix()))


def _runtime_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def digest_runtime_sources(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    normalized = tuple(
        sorted((Path(path) for path in paths), key=_runtime_label)
    )
    if not normalized:
        raise ValueError("runtime source list must not be empty")
    for path in normalized:
        _update_delimited(digest, _runtime_label(path).encode("utf-8"))
        _update_delimited(digest, bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_prediction_fingerprint(
    *,
    model: ModelConfig,
    manifest: ImageManifest,
    image_shape: tuple[int, int, int, int],
    sample_stride: int,
    window_size: int,
    overlap: int,
    specs: Sequence[WindowSpec],
) -> PredictionFingerprint:
    if not isinstance(model, ModelConfig) or model.name is not ModelName.PI3:
        raise ValueError("prediction fingerprint requires Pi3 model name")
    if (
        len(image_shape) != 4
        or image_shape[0] != len(manifest)
        or image_shape[1] != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in image_shape
        )
    ):
        raise ValueError(
            "preprocessed image shape must be positive (N,3,H,W) "
            "and match the manifest"
        )
    if (
        isinstance(sample_stride, bool)
        or not isinstance(sample_stride, int)
        or sample_stride < 1
    ):
        raise ValueError("sample_stride must be a positive integer")
    if window_size <= overlap or overlap < 1:
        raise ValueError(
            "window_size must be greater than overlap >= 1"
        )
    normalized_specs = tuple(specs)
    if (
        not normalized_specs
        or any(not isinstance(spec, WindowSpec) for spec in normalized_specs)
        or normalized_specs[0].frame_start != 0
        or normalized_specs[-1].frame_end != len(manifest)
    ):
        raise ValueError(
            "prediction fingerprint requires complete WindowSpecs"
        )

    checkpoint_digest = sha256_file(model.checkpoint)
    manifest_digest = digest_image_manifest(manifest)
    runtime_digest = digest_runtime_sources(
        runtime_source_paths(model.name)
    )
    payload: dict[str, object] = {
        "prediction_cache_schema_version": (
            PREDICTION_CACHE_SCHEMA_VERSION
        ),
        "adapter_contract_version": MODEL_ADAPTER_CONTRACT_VERSION,
        "model_name": model.name.value,
        "checkpoint_sha256": checkpoint_digest,
        "runtime_source_sha256": runtime_digest,
        "dtype": model.dtype,
        "image_manifest_sha256": manifest_digest,
        "preprocessing_contract": PREPROCESSING_CONTRACT_VERSION,
        "image_shape": list(image_shape),
        "sample_stride": sample_stride,
        "window_size": window_size,
        "overlap": overlap,
        "window_specs": [
            spec.to_payload() for spec in normalized_specs
        ],
    }
    key = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return PredictionFingerprint(
        key=key,
        checkpoint_sha256=checkpoint_digest,
        image_manifest_sha256=manifest_digest,
        runtime_source_sha256=runtime_digest,
        canonical_payload=payload,
    )
