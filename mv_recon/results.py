from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mv_recon.geometry_metrics import (
    DirectionalNormalMetrics,
    GeometryDiagnostics,
    PrimaryMetrics,
    ThresholdMetrics,
)
from mv_recon.protocol import DatasetReference


METRIC_SCHEMA_VERSION = "laser-pointmap-metrics-v2"
RUN_STATES = frozenset(
    {"preflight", "running", "complete", "subset", "incomplete", "failed"}
)


@dataclass(frozen=True)
class RunIdentity:
    protocol_identity_sha256: str
    pipeline_sha256: str
    checkpoint_sha256: str
    sequence_map_sha256: Mapping[str, str]
    auxiliary_checkpoint_sha256: Mapping[str, str] = field(
        default_factory=dict
    )
    metric_version: str = METRIC_SCHEMA_VERSION


@dataclass(frozen=True)
class SequenceResult:
    dataset: str
    sequence: str
    frame_count: int
    input_manifest_sha256: str
    ground_truth_sha256: str
    ordinary_prediction_key: str
    primary: PrimaryMetrics
    diagnostics: GeometryDiagnostics
    cache_diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class FailureRecord:
    dataset: str
    sequence: str
    category: str
    message: str


@dataclass(frozen=True)
class DatasetSummary:
    dataset: str
    status: str
    expected_sequences: int
    selected_sequences: int
    completed_sequences: int
    primary: PrimaryMetrics | None
    paper_reference: DatasetReference
    delta_to_paper: PrimaryMetrics | None


@dataclass(frozen=True)
class RunResults:
    schema_version: str
    state: str
    identity: RunIdentity
    expected_sequences: Mapping[str, tuple[str, ...]]
    full_sequence_counts: Mapping[str, int]
    subset: bool
    paper_reference: Mapping[str, DatasetReference]
    sequences: tuple[SequenceResult, ...]
    failures: tuple[FailureRecord, ...]
    datasets: tuple[DatasetSummary, ...]


def _json_safe(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result values must be finite")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    text = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_write_text(path, text + "\n")


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    _atomic_write_text(path, buffer.getvalue())


def _average_primary(values: Sequence[PrimaryMetrics]) -> PrimaryMetrics:
    if not values:
        raise ValueError("cannot average an empty metric collection")
    return PrimaryMetrics(
        **{
            field.name: float(
                np.mean([getattr(value, field.name) for value in values])
            )
            for field in fields(PrimaryMetrics)
        }
    )


def delta_to_reference(
    computed: PrimaryMetrics,
    reference: DatasetReference,
) -> PrimaryMetrics:
    return PrimaryMetrics(
        **{
            field.name: float(
                getattr(computed, field.name) - getattr(reference, field.name)
            )
            for field in fields(PrimaryMetrics)
        }
    )


def aggregate_dataset(
    dataset: str,
    results: Sequence[SequenceResult],
    *,
    expected_sequences: Sequence[str],
    paper_reference: DatasetReference,
) -> DatasetSummary:
    expected = tuple(expected_sequences)
    if len(set(expected)) != len(expected):
        raise ValueError(f"duplicate expected sequence names for {dataset}")
    relevant = tuple(result for result in results if result.dataset == dataset)
    names = tuple(result.sequence for result in relevant)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate sequence results for {dataset}")
    unexpected = sorted(set(names) - set(expected))
    missing = sorted(set(expected) - set(names))
    if unexpected or missing:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            f"dataset {dataset} sequence coverage mismatch: "
            + "; ".join(details)
        )
    primary = _average_primary([result.primary for result in relevant])
    return DatasetSummary(
        dataset=dataset,
        status="complete",
        expected_sequences=len(expected),
        selected_sequences=len(expected),
        completed_sequences=len(relevant),
        primary=primary,
        paper_reference=paper_reference,
        delta_to_paper=delta_to_reference(primary, paper_reference),
    )


def _partial_dataset_summary(
    dataset: str,
    results: Sequence[SequenceResult],
    expected_sequences: Sequence[str],
    full_sequence_count: int,
    paper_reference: DatasetReference,
    *,
    status: str,
) -> DatasetSummary:
    relevant = tuple(result for result in results if result.dataset == dataset)
    primary = (
        _average_primary([result.primary for result in relevant])
        if relevant
        else None
    )
    return DatasetSummary(
        dataset=dataset,
        status=status,
        expected_sequences=full_sequence_count,
        selected_sequences=len(tuple(expected_sequences)),
        completed_sequences=len(relevant),
        primary=primary,
        paper_reference=paper_reference,
        delta_to_paper=(
            delta_to_reference(primary, paper_reference)
            if primary is not None
            else None
        ),
    )


def _primary_from_payload(payload: Mapping[str, object]) -> PrimaryMetrics:
    return PrimaryMetrics(
        **{field.name: float(payload[field.name]) for field in fields(PrimaryMetrics)}
    )


def _reference_from_payload(
    payload: Mapping[str, object],
) -> DatasetReference:
    return DatasetReference(
        **{
            field.name: float(payload[field.name])
            for field in fields(DatasetReference)
        }
    )


def _diagnostics_from_payload(
    payload: Mapping[str, object],
) -> GeometryDiagnostics:
    normals_payload = payload["directional_normals"]
    if not isinstance(normals_payload, Mapping):
        raise ValueError("invalid stored directional normal metrics")
    threshold_payloads = payload["thresholds"]
    if not isinstance(threshold_payloads, list):
        raise ValueError("invalid stored threshold metrics")
    return GeometryDiagnostics(
        umeyama_scale=float(payload["umeyama_scale"]),
        icp_transformation=tuple(
            tuple(float(value) for value in row)
            for row in payload["icp_transformation"]
        ),
        icp_fitness=float(payload["icp_fitness"]),
        icp_inlier_rmse=float(payload["icp_inlier_rmse"]),
        predicted_point_count=int(payload["predicted_point_count"]),
        ground_truth_point_count=int(payload["ground_truth_point_count"]),
        directional_normals=DirectionalNormalMetrics(
            **{
                field.name: float(normals_payload[field.name])
                for field in fields(DirectionalNormalMetrics)
            }
        ),
        chamfer_l1_m=float(payload["chamfer_l1_m"]),
        thresholds=tuple(
            ThresholdMetrics(
                threshold_m=float(item["threshold_m"]),
                precision=float(item["precision"]),
                recall=float(item["recall"]),
                fscore=float(item["fscore"]),
            )
            for item in threshold_payloads
        ),
    )


def _sequence_from_payload(payload: Mapping[str, object]) -> SequenceResult:
    primary_payload = payload["primary"]
    diagnostics_payload = payload["diagnostics"]
    cache_payload = payload.get("cache_diagnostics", {})
    if not isinstance(primary_payload, Mapping) or not isinstance(
        diagnostics_payload, Mapping
    ) or not isinstance(cache_payload, Mapping):
        raise ValueError("invalid stored sequence result")
    return SequenceResult(
        dataset=str(payload["dataset"]),
        sequence=str(payload["sequence"]),
        frame_count=int(payload["frame_count"]),
        input_manifest_sha256=str(payload["input_manifest_sha256"]),
        ground_truth_sha256=str(payload["ground_truth_sha256"]),
        ordinary_prediction_key=str(payload["ordinary_prediction_key"]),
        primary=_primary_from_payload(primary_payload),
        diagnostics=_diagnostics_from_payload(diagnostics_payload),
        cache_diagnostics=dict(cache_payload),
    )


def _identity_from_payload(payload: Mapping[str, object]) -> RunIdentity:
    map_hashes = payload["sequence_map_sha256"]
    if not isinstance(map_hashes, Mapping):
        raise ValueError("invalid stored sequence-map identity")
    auxiliary_hashes = payload.get("auxiliary_checkpoint_sha256", {})
    if not isinstance(auxiliary_hashes, Mapping):
        raise ValueError("invalid stored auxiliary-checkpoint identity")
    return RunIdentity(
        protocol_identity_sha256=str(payload["protocol_identity_sha256"]),
        pipeline_sha256=str(payload["pipeline_sha256"]),
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        sequence_map_sha256={
            str(key): str(value) for key, value in map_hashes.items()
        },
        auxiliary_checkpoint_sha256={
            str(key): str(value) for key, value in auxiliary_hashes.items()
        },
        metric_version=str(payload["metric_version"]),
    )


def _failure_from_payload(payload: Mapping[str, object]) -> FailureRecord:
    return FailureRecord(
        dataset=str(payload["dataset"]),
        sequence=str(payload["sequence"]),
        category=str(payload["category"]),
        message=str(payload["message"]),
    )


class ResultStore:
    def __init__(
        self,
        output_dir: str | Path,
        identity: RunIdentity,
        *,
        resume: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.identity = identity
        self.resume = bool(resume)
        self._expected_sequences: dict[str, tuple[str, ...]] = {}
        self._full_sequence_counts: dict[str, int] = {}
        self._paper_reference: dict[str, DatasetReference] = {}
        self._subset = False
        self._state = "running"
        self._sequences: list[SequenceResult] = []
        self._failures: list[FailureRecord] = []
        self._protocol_manifest: dict[str, object] = {}

        if self.resume:
            self._load_existing()
        else:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                raise FileExistsError(
                    f"result directory is non-empty; enable protocol.resume "
                    f"to reuse it: {self.output_dir}"
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_existing(self) -> None:
        results_path = self.output_dir / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(
                f"resume requires canonical results.json: {results_path}"
            )
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError("resume results.json is unreadable") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("resume results.json must contain an object")
        if payload.get("schema_version") != METRIC_SCHEMA_VERSION:
            raise ValueError(
                "resume identity mismatch: metric result schema changed"
            )
        stored_identity_payload = payload.get("identity")
        if not isinstance(stored_identity_payload, Mapping):
            raise ValueError("resume results.json has no run identity")
        stored_identity = _identity_from_payload(stored_identity_payload)
        if stored_identity != self.identity:
            raise ValueError(
                "resume identity mismatch: protocol, pipeline, checkpoint, "
                "sequence map, or metric version changed"
            )
        self._state = str(payload.get("state", "incomplete"))
        self._subset = bool(payload.get("subset", False))
        expected_payload = payload.get("expected_sequences", {})
        counts_payload = payload.get("full_sequence_counts", {})
        references_payload = payload.get("paper_reference", {})
        if not all(
            isinstance(item, Mapping)
            for item in (expected_payload, counts_payload, references_payload)
        ):
            raise ValueError("resume results.json has invalid run metadata")
        self._expected_sequences = {
            str(dataset): tuple(str(name) for name in names)
            for dataset, names in expected_payload.items()
        }
        self._full_sequence_counts = {
            str(dataset): int(count)
            for dataset, count in counts_payload.items()
        }
        self._paper_reference = {
            str(dataset): _reference_from_payload(reference)
            for dataset, reference in references_payload.items()
        }
        sequence_payloads = payload.get("sequences", [])
        failure_payloads = payload.get("failures", [])
        if not isinstance(sequence_payloads, list) or not isinstance(
            failure_payloads, list
        ):
            raise ValueError("resume results.json has invalid record arrays")
        self._sequences = []
        for item in sequence_payloads:
            if not isinstance(item, Mapping):
                raise ValueError("invalid stored sequence result")
            sequence = _sequence_from_payload(item)
            _json_safe(sequence)
            self._sequences.append(sequence)
        self._failures = []
        for item in failure_payloads:
            if not isinstance(item, Mapping):
                raise ValueError("invalid stored failure record")
            failure = _failure_from_payload(item)
            _json_safe(failure)
            self._failures.append(failure)

    def initialize(
        self,
        *,
        expected_sequences: Mapping[str, Sequence[str]],
        full_sequence_counts: Mapping[str, int],
        paper_reference: Mapping[str, DatasetReference],
        subset: bool,
        preflight: bool = False,
    ) -> RunResults:
        normalized_expected = {
            str(dataset): tuple(str(name) for name in names)
            for dataset, names in expected_sequences.items()
        }
        normalized_counts = {
            str(dataset): int(count)
            for dataset, count in full_sequence_counts.items()
        }
        normalized_references = dict(paper_reference)
        if not normalized_expected or set(normalized_expected) != set(
            normalized_counts
        ) or set(normalized_expected) != set(normalized_references):
            raise ValueError(
                "expected sequences, full counts, and paper references must "
                "contain identical datasets"
            )
        for dataset, names in normalized_expected.items():
            if not names or len(set(names)) != len(names):
                raise ValueError(
                    f"selected sequence names for {dataset} must be unique"
                )
            if normalized_counts[dataset] < len(names):
                raise ValueError(
                    f"full sequence count for {dataset} is smaller than selection"
                )
        if self.resume and self._expected_sequences:
            compatible = (
                self._expected_sequences == normalized_expected
                and self._full_sequence_counts == normalized_counts
                and self._paper_reference == normalized_references
                and self._subset == bool(subset)
            )
            if not compatible:
                raise ValueError("resume run metadata mismatch")
        self._expected_sequences = normalized_expected
        self._full_sequence_counts = normalized_counts
        self._paper_reference = normalized_references
        self._subset = bool(subset)
        self._state = "preflight" if preflight else "running"
        self._write_failures()
        return self._write_results()

    def write_protocol_artifacts(
        self,
        *,
        resolved_protocol_yaml: str,
        resolved_pipeline_yaml: str,
        manifest: Mapping[str, object],
    ) -> None:
        _atomic_write_text(
            self.output_dir / "resolved_protocol.yaml",
            resolved_protocol_yaml,
        )
        _atomic_write_text(
            self.output_dir / "resolved_pipeline.yaml",
            resolved_pipeline_yaml,
        )
        self._protocol_manifest = dict(manifest)
        _atomic_write_json(
            self.output_dir / "protocol_manifest.json",
            self._protocol_manifest,
        )

    def update_protocol_manifest(
        self,
        updates: Mapping[str, object],
    ) -> None:
        self._protocol_manifest.update(dict(updates))
        _atomic_write_json(
            self.output_dir / "protocol_manifest.json",
            self._protocol_manifest,
        )

    def reusable_sequence(
        self,
        dataset: str,
        sequence: str,
        input_manifest_sha256: str,
        ground_truth_sha256: str,
    ) -> SequenceResult | None:
        for result in self._sequences:
            if (
                result.dataset == dataset
                and result.sequence == sequence
                and result.input_manifest_sha256 == input_manifest_sha256
                and result.ground_truth_sha256 == ground_truth_sha256
            ):
                return result
        return None

    def record_sequence(self, result: SequenceResult) -> None:
        if result.dataset not in self._expected_sequences:
            raise ValueError(f"unexpected result dataset: {result.dataset}")
        if result.sequence not in self._expected_sequences[result.dataset]:
            raise ValueError(
                f"unexpected sequence result: {result.dataset}/{result.sequence}"
            )
        _json_safe(result)
        existing_index = next(
            (
                index
                for index, item in enumerate(self._sequences)
                if item.dataset == result.dataset
                and item.sequence == result.sequence
            ),
            None,
        )
        if existing_index is not None:
            if not self.resume:
                raise ValueError(
                    f"duplicate sequence result: "
                    f"{result.dataset}/{result.sequence}"
                )
            self._sequences[existing_index] = result
        else:
            self._sequences.append(result)
        self._failures = [
            failure
            for failure in self._failures
            if not (
                failure.dataset == result.dataset
                and failure.sequence == result.sequence
            )
        ]
        self._state = "running"
        self._write_failures()
        self._write_results()

    def record_failure(self, failure: FailureRecord) -> None:
        _json_safe(failure)
        self._failures.append(failure)
        self._state = "incomplete" if self._sequences else "failed"
        self._write_failures()
        self._write_results()

    def _coverage_complete(self) -> bool:
        completed = {
            (result.dataset, result.sequence) for result in self._sequences
        }
        expected = {
            (dataset, sequence)
            for dataset, names in self._expected_sequences.items()
            for sequence in names
        }
        return completed == expected

    def _dataset_summaries(self, state: str) -> tuple[DatasetSummary, ...]:
        summaries = []
        for dataset, expected_names in self._expected_sequences.items():
            relevant_names = {
                result.sequence
                for result in self._sequences
                if result.dataset == dataset
            }
            dataset_complete = relevant_names == set(expected_names)
            if dataset_complete:
                summary = aggregate_dataset(
                    dataset,
                    self._sequences,
                    expected_sequences=expected_names,
                    paper_reference=self._paper_reference[dataset],
                )
                dataset_status = "subset" if self._subset else "complete"
                summary = replace(
                    summary,
                    status=dataset_status,
                    expected_sequences=self._full_sequence_counts[dataset],
                )
            else:
                summary = _partial_dataset_summary(
                    dataset,
                    self._sequences,
                    expected_names,
                    self._full_sequence_counts[dataset],
                    self._paper_reference[dataset],
                    status=state,
                )
            summaries.append(summary)
        return tuple(summaries)

    def _current_result(self, state: str | None = None) -> RunResults:
        selected_state = state or self._state
        if selected_state not in RUN_STATES:
            raise ValueError(f"invalid result state: {selected_state}")
        return RunResults(
            schema_version=METRIC_SCHEMA_VERSION,
            state=selected_state,
            identity=self.identity,
            expected_sequences=dict(self._expected_sequences),
            full_sequence_counts=dict(self._full_sequence_counts),
            subset=self._subset,
            paper_reference=dict(self._paper_reference),
            sequences=tuple(self._sequences),
            failures=tuple(self._failures),
            datasets=self._dataset_summaries(selected_state),
        )

    def _write_results(self, state: str | None = None) -> RunResults:
        result = self._current_result(state)
        _atomic_write_json(self.output_dir / "results.json", result)
        return result

    def _write_failures(self) -> None:
        lines = [
            json.dumps(
                _json_safe(failure),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            for failure in self._failures
        ]
        text = "\n".join(lines)
        if text:
            text += "\n"
        _atomic_write_text(self.output_dir / "failures.jsonl", text)

    def finalize(self) -> RunResults:
        if self._failures:
            state = "incomplete" if self._sequences else "failed"
        elif not self._coverage_complete():
            state = "incomplete"
        elif self._subset:
            state = "subset"
        else:
            state = "complete"
        self._state = state
        result = self._write_results()
        self._write_summary_csv(result)
        self._write_sequences_csv(result)
        self._write_failures()
        return result

    def _write_summary_csv(self, result: RunResults) -> None:
        fieldnames = (
            "dataset",
            "status",
            "expected_sequences",
            "selected_sequences",
            "completed_sequences",
            *(field.name for field in fields(PrimaryMetrics)),
        )
        rows = []
        for summary in result.datasets:
            row: dict[str, object] = {
                "dataset": summary.dataset,
                "status": summary.status,
                "expected_sequences": summary.expected_sequences,
                "selected_sequences": summary.selected_sequences,
                "completed_sequences": summary.completed_sequences,
            }
            if summary.primary is not None:
                row.update(asdict(summary.primary))
            rows.append(row)
        _atomic_write_csv(
            self.output_dir / "summary.csv",
            rows,
            fieldnames,
        )

    @staticmethod
    def _threshold_label(threshold_m: float) -> str:
        centimeters = threshold_m * 100.0
        if math.isclose(centimeters, round(centimeters), abs_tol=1e-9):
            return f"{int(round(centimeters))}cm"
        return f"{threshold_m:g}m".replace(".", "p")

    def _sequence_row(self, result: SequenceResult) -> dict[str, object]:
        row: dict[str, object] = {
            "dataset": result.dataset,
            "sequence": result.sequence,
            "frame_count": result.frame_count,
            "input_manifest_sha256": result.input_manifest_sha256,
            "ground_truth_sha256": result.ground_truth_sha256,
            "ordinary_prediction_key": result.ordinary_prediction_key,
            **asdict(result.primary),
            **asdict(result.diagnostics.directional_normals),
            "chamfer_l1_m": result.diagnostics.chamfer_l1_m,
            "icp_fitness": result.diagnostics.icp_fitness,
            "icp_inlier_rmse": result.diagnostics.icp_inlier_rmse,
            "umeyama_scale": result.diagnostics.umeyama_scale,
            "predicted_point_count": result.diagnostics.predicted_point_count,
            "ground_truth_point_count": (
                result.diagnostics.ground_truth_point_count
            ),
        }
        for threshold in result.diagnostics.thresholds:
            label = self._threshold_label(threshold.threshold_m)
            row[f"precision_at_{label}"] = threshold.precision
            row[f"recall_at_{label}"] = threshold.recall
            row[f"fscore_at_{label}"] = threshold.fscore
        for key, value in result.cache_diagnostics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"cache_{key}"] = value
        return row

    def _write_sequences_csv(self, result: RunResults) -> None:
        rows = [self._sequence_row(sequence) for sequence in result.sequences]
        leading = [
            "dataset",
            "sequence",
            "frame_count",
            "input_manifest_sha256",
            "ground_truth_sha256",
            "ordinary_prediction_key",
            *(field.name for field in fields(PrimaryMetrics)),
            *(field.name for field in fields(DirectionalNormalMetrics)),
            "chamfer_l1_m",
            "icp_fitness",
            "icp_inlier_rmse",
            "umeyama_scale",
            "predicted_point_count",
            "ground_truth_point_count",
        ]
        extras = sorted(
            {
                key
                for row in rows
                for key in row
                if key not in leading
            }
        )
        _atomic_write_csv(
            self.output_dir / "sequences.csv",
            rows,
            (*leading, *extras),
        )
