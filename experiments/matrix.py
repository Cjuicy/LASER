from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline.config import (
    LoadedPipelineConfig,
    ReconstructionMode,
    SegmentationMethod,
)
from pipeline.artifacts import load_reconstruction_artifact

from .config import EvaluationKind


@dataclass(frozen=True)
class MatrixEntry:
    segmentation_method: SegmentationMethod
    reconstruction_mode: ReconstructionMode

    @property
    def name(self) -> str:
        return f"{self.segmentation_method.value}-{self.reconstruction_mode.value}"


def validate_matrix_entry(kind: EvaluationKind, entry: MatrixEntry) -> None:
    if not isinstance(kind, EvaluationKind) or not isinstance(entry, MatrixEntry):
        raise ValueError("matrix kind or entry is invalid")
    if (
        kind is EvaluationKind.POINTCLOUD
        and entry.reconstruction_mode is not ReconstructionMode.NO_LOOP
    ):
        raise ValueError("pointcloud matrix only supports no_loop")


def build_matrix(kind: EvaluationKind) -> tuple[MatrixEntry, ...]:
    if not isinstance(kind, EvaluationKind):
        raise ValueError("evaluation kind is invalid")
    modes = (
        tuple(ReconstructionMode)
        if kind is EvaluationKind.ATE
        else (ReconstructionMode.NO_LOOP,)
    )
    entries = tuple(
        MatrixEntry(segmentation, mode)
        for segmentation in SegmentationMethod
        for mode in modes
    )
    for entry in entries:
        validate_matrix_entry(kind, entry)
    return entries


def reconstruction_identity(
    loaded: LoadedPipelineConfig,
    *,
    input_manifest_sha256: str,
    checkpoint_sha256: str,
) -> str:
    if not isinstance(loaded, LoadedPipelineConfig):
        raise ValueError("identity requires LoadedPipelineConfig")
    payload = {
        "schema": "laser-reconstruction-artifact-v1",
        "resolved_reconstruction_sha256": loaded.sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ArtifactRepository:
    REQUIRED_FILES = frozenset(
        {"manifest.json", "trajectory.pt", "pointmap.pt", "confidence.pt"}
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, identity: str) -> Path:
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise ValueError("reconstruction identity must be SHA256 hex")
        return self.root / identity

    def _is_complete(self, target: Path) -> bool:
        present = target.is_dir() and self.REQUIRED_FILES <= {
            path.name for path in target.iterdir() if path.is_file()
        }
        if not present:
            return False
        try:
            load_reconstruction_artifact(target)
        except ValueError:
            return False
        return True

    def get_or_create(
        self,
        identity: str,
        reconstruct: Callable[[Path], Path],
    ) -> Path:
        target = self.path_for(identity)
        if target.exists():
            if not self._is_complete(target):
                raise ValueError(
                    f"incomplete artifact or digest mismatch: {target}"
                )
            return target
        result = Path(reconstruct(target))
        if result != target or not self._is_complete(target):
            raise ValueError(f"reconstruction did not create complete artifact: {target}")
        return target
