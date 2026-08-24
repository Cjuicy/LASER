from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from .ate import evaluate_ate_inputs
from .config import CapabilityExperimentConfig, EvaluationInputConfig, EvaluationKind
from .matrix import CapabilityMatrixEntry
from .pointcloud import evaluate_pointcloud_inputs


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class EvaluatorStatus(str, Enum):
    PASSED = "passed"
    SKIPPED = "skipped"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class EvaluatorIdentity:
    reconstruction_identity: str
    artifact_manifest_sha256: str
    capability_declaration_sha256: str
    evaluator_config_sha256: str
    ground_truth_sha256: str
    source_revision: str

    def __post_init__(self) -> None:
        for name in (
            "reconstruction_identity",
            "artifact_manifest_sha256",
            "capability_declaration_sha256",
            "evaluator_config_sha256",
            "ground_truth_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be SHA256 hex")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be a non-empty string")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EvaluatorError:
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        for name in ("exception_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"evaluator error {name} must be non-empty")


@dataclass(frozen=True)
class EvaluatorRecord:
    kind: EvaluationKind
    status: EvaluatorStatus
    identity_digest: str | None
    output_path: str | None
    error: EvaluatorError | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvaluationKind):
            try:
                object.__setattr__(self, "kind", EvaluationKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("evaluator record kind is invalid") from exc
        if not isinstance(self.status, EvaluatorStatus):
            try:
                object.__setattr__(self, "status", EvaluatorStatus(self.status))
            except (TypeError, ValueError) as exc:
                raise ValueError("evaluator record status is invalid") from exc
        if self.identity_digest is not None and not _is_sha256(
            self.identity_digest
        ):
            raise ValueError("evaluator identity digest must be SHA256 hex")
        if self.output_path is not None and (
            not isinstance(self.output_path, str) or not self.output_path
        ):
            raise ValueError("evaluator output path must be non-empty")
        if self.error is not None and not isinstance(self.error, EvaluatorError):
            raise ValueError("evaluator record error is invalid")

        if self.status is EvaluatorStatus.PASSED:
            if (
                self.identity_digest is None
                or self.output_path is None
                or self.error is not None
            ):
                raise ValueError("passed evaluator record is incomplete")
        elif self.status is EvaluatorStatus.SKIPPED:
            if any(
                value is not None
                for value in (self.identity_digest, self.output_path, self.error)
            ):
                raise ValueError("skipped evaluator record must be empty")
        elif self.status is EvaluatorStatus.FAILED:
            if self.output_path is not None or self.error is None:
                raise ValueError("failed evaluator record requires only an error")
        elif self.status is EvaluatorStatus.BLOCKED:
            if (
                self.identity_digest is not None
                or self.output_path is not None
                or self.error is None
            ):
                raise ValueError("blocked evaluator record requires only an error")


def _record_payload(record: EvaluatorRecord) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": record.kind.value,
        "status": record.status.value,
        "identity_digest": record.identity_digest,
        "output_path": record.output_path,
        "error": None if record.error is None else asdict(record.error),
    }


def write_evaluator_record(record: EvaluatorRecord, path: str | Path) -> Path:
    if not isinstance(record, EvaluatorRecord):
        raise ValueError("record must be EvaluatorRecord")
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
            json.dump(
                _record_payload(record),
                temporary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{context} fields are invalid")


def read_evaluator_record(path: str | Path) -> EvaluatorRecord:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("evaluator record is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("evaluator record must be a mapping")
    _require_exact_fields(
        payload,
        {
            "schema_version",
            "kind",
            "status",
            "identity_digest",
            "output_path",
            "error",
        },
        context="evaluator record",
    )
    if payload["schema_version"] != 1:
        raise ValueError("evaluator record schema version must be 1")
    error_payload = payload["error"]
    error = None
    if error_payload is not None:
        if not isinstance(error_payload, dict):
            raise ValueError("evaluator error must be a mapping")
        _require_exact_fields(
            error_payload,
            {"exception_type", "message"},
            context="evaluator error",
        )
        error = EvaluatorError(**error_payload)
    try:
        return EvaluatorRecord(
            kind=EvaluationKind(payload["kind"]),
            status=EvaluatorStatus(payload["status"]),
            identity_digest=payload["identity_digest"],
            output_path=payload["output_path"],
            error=error,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluator record payload is invalid") from exc


_URL_USERINFO = re.compile(r"(?P<scheme>https?://)[^/@\s]+@", re.IGNORECASE)


def _sanitize_error(error: Exception) -> EvaluatorError:
    message = str(error) or "no error message"
    message = _URL_USERINFO.sub(r"\g<scheme><redacted>@", message)
    message = "".join(
        character
        if ord(character) >= 32 and ord(character) != 127
        else " "
        for character in message
    )
    message = " ".join(message.split())[:2000] or "no error message"
    return EvaluatorError(type(error).__name__, message)


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"declared evaluator input does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capability_declaration_sha256(
    kind: EvaluationKind,
    experiment: CapabilityExperimentConfig,
    evaluator_input: EvaluationInputConfig,
) -> str:
    declaration = {
        "kind": kind.value,
        "config_path": evaluator_input.config_path,
        "ground_truth_path": evaluator_input.ground_truth_path,
    }
    if kind is EvaluationKind.ATE:
        declaration["ground_truth_format"] = (
            evaluator_input.ground_truth_format
        )
    else:
        declaration["dataset_name"] = experiment.dataset_name
        declaration["sequence"] = experiment.sequence
    payload = json.dumps(
        declaration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _output_exists(path: str | Path) -> bool:
    output = Path(path)
    if output.is_file():
        return True
    if not output.is_dir():
        return False
    return any(
        child.name != "evaluator_record.json"
        for child in output.iterdir()
    )


class EvaluationBundleRunner:
    def __init__(
        self,
        *,
        ate_evaluator: Callable = evaluate_ate_inputs,
        pointcloud_evaluator: Callable = evaluate_pointcloud_inputs,
    ) -> None:
        self._evaluators = {
            EvaluationKind.ATE: ate_evaluator,
            EvaluationKind.POINTCLOUD: pointcloud_evaluator,
        }

    @staticmethod
    def _record_path(output_root: str | Path, kind: EvaluationKind) -> Path:
        return Path(output_root) / kind.value / "evaluator_record.json"

    @staticmethod
    def _scheduled(
        kind: EvaluationKind,
        entry: CapabilityMatrixEntry,
        experiment: CapabilityExperimentConfig,
    ) -> bool:
        return (
            kind in experiment.evaluator_inputs
            and kind in entry.evaluator_kinds
        )

    @staticmethod
    def _identity(
        artifact_dir: str | Path,
        reconstruction_identity: str,
        kind: EvaluationKind,
        experiment: CapabilityExperimentConfig,
        evaluator_input: EvaluationInputConfig,
        source_revision: str,
    ) -> EvaluatorIdentity:
        return EvaluatorIdentity(
            reconstruction_identity=reconstruction_identity,
            artifact_manifest_sha256=_sha256_file(
                Path(artifact_dir) / "manifest.json"
            ),
            capability_declaration_sha256=(
                _capability_declaration_sha256(
                    kind,
                    experiment,
                    evaluator_input,
                )
            ),
            evaluator_config_sha256=_sha256_file(
                evaluator_input.config_path
            ),
            ground_truth_sha256=_sha256_file(
                evaluator_input.ground_truth_path
            ),
            source_revision=source_revision,
        )

    @staticmethod
    def _reusable(
        record_path: Path,
        identity_digest: str,
    ) -> EvaluatorRecord | None:
        if not record_path.is_file():
            return None
        try:
            record = read_evaluator_record(record_path)
        except ValueError:
            return None
        if (
            record.status is EvaluatorStatus.PASSED
            and record.identity_digest == identity_digest
            and record.output_path is not None
            and _output_exists(record.output_path)
        ):
            return record
        return None

    def _evaluate(
        self,
        kind: EvaluationKind,
        artifact_dir: str | Path,
        experiment: CapabilityExperimentConfig,
        evaluator_input: EvaluationInputConfig,
        output_dir: Path,
    ) -> Path:
        evaluator = self._evaluators[kind]
        if kind is EvaluationKind.ATE:
            return Path(
                evaluator(
                    artifact_dir,
                    ground_truth=evaluator_input.ground_truth_path,
                    ground_truth_format=evaluator_input.ground_truth_format,
                    evaluation_config=evaluator_input.config_path,
                    output_dir=output_dir,
                )
            )
        return Path(
            evaluator(
                artifact_dir,
                dataset_name=experiment.dataset_name,
                sequence=experiment.sequence,
                ground_truth=evaluator_input.ground_truth_path,
                evaluation_config=evaluator_input.config_path,
                output_dir=output_dir,
            )
        )

    def run(
        self,
        *,
        artifact_dir: str | Path,
        reconstruction_identity: str,
        entry: CapabilityMatrixEntry,
        experiment: CapabilityExperimentConfig,
        output_root: str | Path,
        source_revision: str,
    ) -> tuple[EvaluatorRecord, ...]:
        if not isinstance(entry, CapabilityMatrixEntry):
            raise ValueError("evaluation bundle entry is invalid")
        if not isinstance(experiment, CapabilityExperimentConfig):
            raise ValueError("evaluation bundle experiment is invalid")
        records = []
        for kind in EvaluationKind:
            record_path = self._record_path(output_root, kind)
            if not self._scheduled(kind, entry, experiment):
                record = EvaluatorRecord(
                    kind=kind,
                    status=EvaluatorStatus.SKIPPED,
                    identity_digest=None,
                    output_path=None,
                    error=None,
                )
                write_evaluator_record(record, record_path)
                records.append(record)
                continue

            identity_digest = None
            try:
                evaluator_input = experiment.evaluator_inputs[kind]
                identity = self._identity(
                    artifact_dir,
                    reconstruction_identity,
                    kind,
                    experiment,
                    evaluator_input,
                    source_revision,
                )
                identity_digest = identity.digest
                reusable = self._reusable(record_path, identity_digest)
                if reusable is not None:
                    records.append(reusable)
                    continue
                output = self._evaluate(
                    kind,
                    artifact_dir,
                    experiment,
                    evaluator_input,
                    record_path.parent,
                )
                if not _output_exists(output):
                    raise FileNotFoundError(
                        f"evaluator returned missing output: {output}"
                    )
                record = EvaluatorRecord(
                    kind=kind,
                    status=EvaluatorStatus.PASSED,
                    identity_digest=identity_digest,
                    output_path=str(output.resolve()),
                    error=None,
                )
            except Exception as error:
                record = EvaluatorRecord(
                    kind=kind,
                    status=EvaluatorStatus.FAILED,
                    identity_digest=identity_digest,
                    output_path=None,
                    error=_sanitize_error(error),
                )
            write_evaluator_record(record, record_path)
            records.append(record)
        return tuple(records)

    def blocked(
        self,
        *,
        entry: CapabilityMatrixEntry,
        experiment: CapabilityExperimentConfig,
        output_root: str | Path,
        error: Exception,
    ) -> tuple[EvaluatorRecord, ...]:
        if not isinstance(entry, CapabilityMatrixEntry):
            raise ValueError("evaluation bundle entry is invalid")
        if not isinstance(experiment, CapabilityExperimentConfig):
            raise ValueError("evaluation bundle experiment is invalid")
        if not isinstance(error, Exception):
            raise ValueError("blocked evaluation requires reconstruction error")
        blocked_error = _sanitize_error(error)
        records = []
        for kind in EvaluationKind:
            if self._scheduled(kind, entry, experiment):
                record = EvaluatorRecord(
                    kind=kind,
                    status=EvaluatorStatus.BLOCKED,
                    identity_digest=None,
                    output_path=None,
                    error=blocked_error,
                )
            else:
                record = EvaluatorRecord(
                    kind=kind,
                    status=EvaluatorStatus.SKIPPED,
                    identity_digest=None,
                    output_path=None,
                    error=None,
                )
            write_evaluator_record(
                record,
                self._record_path(output_root, kind),
            )
            records.append(record)
        return tuple(records)
