from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .geometry_metrics import PrimaryMetrics


@dataclass(frozen=True)
class SequencePointCloudResult:
    dataset: str
    sequence: str
    frame_count: int
    primary: PrimaryMetrics
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class PointCloudDatasetSummary:
    dataset: str
    expected_sequences: int
    completed_sequences: int
    primary: PrimaryMetrics


def _json_safe(value: object):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("point-cloud result values must be finite")
    return value


def _average(values: Sequence[PrimaryMetrics]) -> PrimaryMetrics:
    if not values:
        raise ValueError("cannot average empty point-cloud metrics")
    return PrimaryMetrics(
        **{
            item.name: float(np.mean([getattr(value, item.name) for value in values]))
            for item in fields(PrimaryMetrics)
        }
    )


def aggregate_pointcloud_results(
    dataset: str,
    results: Sequence[SequencePointCloudResult],
    *,
    expected_sequences: Sequence[str],
) -> PointCloudDatasetSummary:
    expected = tuple(expected_sequences)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected sequence names must be unique and non-empty")
    relevant = tuple(result for result in results if result.dataset == dataset)
    names = tuple(result.sequence for result in relevant)
    missing = sorted(set(expected) - set(names))
    unexpected = sorted(set(names) - set(expected))
    if len(set(names)) != len(names) or missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        if len(set(names)) != len(names):
            details.append("duplicate sequences")
        raise ValueError("point-cloud sequence coverage mismatch: " + "; ".join(details))
    return PointCloudDatasetSummary(
        dataset,
        len(expected),
        len(relevant),
        _average([result.primary for result in relevant]),
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_pointcloud_results(
    summary: PointCloudDatasetSummary,
    results: Sequence[SequencePointCloudResult],
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output / "pointcloud_metrics.json",
        json.dumps(
            _json_safe(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    fieldnames = (
        "dataset",
        "sequence",
        "frame_count",
        *(item.name for item in fields(PrimaryMetrics)),
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "dataset": result.dataset,
                "sequence": result.sequence,
                "frame_count": result.frame_count,
                **asdict(result.primary),
            }
        )
    _atomic_text(output / "pointcloud_sequences.csv", buffer.getvalue())
    return output
