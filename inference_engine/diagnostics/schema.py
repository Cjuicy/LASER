"""Versioned, JSON-safe diagnostic data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _check_schema(data: dict[str, Any]) -> None:
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}."
        )


@dataclass(frozen=True)
class DiagnosticContext:
    run_id: str
    config_id: str
    sequence_id: str
    pass_id: int
    window_id: int
    frame_start: int

    def __post_init__(self) -> None:
        for name in ("run_id", "config_id", "sequence_id"):
            _require_text(name, getattr(self, name))
        if self.pass_id not in (1, 2):
            raise ValueError("pass_id must be 1 or 2")
        if self.window_id < 0:
            raise ValueError("window_id must be non-negative")
        if self.frame_start < 0:
            raise ValueError("frame_start must be non-negative")

    def frame_id(self, local_index: int) -> int:
        if local_index < 0:
            raise ValueError("local_index must be non-negative")
        return self.frame_start + local_index

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiagnosticContext":
        _check_schema(data)
        values = {key: value for key, value in data.items() if key != "schema_version"}
        return cls(**values)


@dataclass(frozen=True)
class SelectedInterval:
    sequence_id: str
    start_frame: int
    end_frame: int
    reasons: tuple[str, ...]
    score: float

    def __post_init__(self) -> None:
        _require_text("sequence_id", self.sequence_id)
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if not self.reasons or any(not str(reason).strip() for reason in self.reasons):
            raise ValueError("reasons must contain at least one non-empty value")

    def contains(self, frame_id: int) -> bool:
        return self.start_frame <= frame_id <= self.end_frame

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["reasons"] = list(self.reasons)
        return {"schema_version": SCHEMA_VERSION, **values}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelectedInterval":
        _check_schema(data)
        return cls(
            sequence_id=data["sequence_id"],
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            reasons=tuple(data["reasons"]),
            score=float(data["score"]),
        )


@dataclass
class RunManifest:
    run_id: str
    git_commit: str
    checkpoint_sha256: str
    config_hash: str
    dataset_fingerprint: str
    seed: int
    environment: dict[str, Any]
    budget: dict[str, Any]
    status: str = "created"
    state: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "git_commit",
            "checkpoint_sha256",
            "config_hash",
            "dataset_fingerprint",
        ):
            _require_text(name, getattr(self, name))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}."
            )

    def mark(
        self,
        phase: str,
        config_id: str,
        sequence_id: str,
        status: str,
    ) -> None:
        for name, value in (
            ("phase", phase),
            ("config_id", config_id),
            ("sequence_id", sequence_id),
            ("status", status),
        ):
            _require_text(name, value)
        self.state.setdefault(phase, {}).setdefault(config_id, {})[sequence_id] = status

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        _check_schema(data)
        return cls(**data)
