"""Deterministic multi-signal diagnostic interval selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .schema import SelectedInterval


FAMILIES = {
    "trajectory": "trajectory_regret",
    "merge_atom": "merge_anomaly",
    "scale": "scale_dispersion",
    "temporal": "temporal_churn",
}
WEIGHTS = {"trajectory": .40, "merge_atom": .20, "scale": .25, "temporal": .15}
RECOVERY = {"02", "04", "10"}
GUARD = {"00", "05", "09"}


def robust_zscore(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if not array.size:
        return array
    finite = np.isfinite(array)
    result = np.zeros_like(array)
    if not finite.any():
        return result
    clean = array[finite]
    median = np.median(clean)
    mad = np.median(np.abs(clean - median))
    if mad > 1e-12:
        scale = 1.4826 * mad
    else:
        scale = np.std(clean)
    if scale > 1e-12:
        result[finite] = (clean - median) / scale
    result[~finite] = 0
    return result


def _stride(rows: list[dict]) -> int:
    starts = sorted({int(row["frame_start"]) for row in rows})
    differences = np.diff(starts)
    positive = differences[differences > 0]
    if positive.size:
        return int(np.median(positive))
    return max(1, int(rows[0]["frame_end"]) - int(rows[0]["frame_start"]) + 1)


def select_intervals(records: list[dict], *, limit: int = 48, context_windows: int = 2) -> list[SelectedInterval]:
    if limit <= 0:
        return []
    # Union configurations at the same global window so every method gets identical cases.
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(str(record["sequence_id"]), int(record["frame_start"]), int(record["frame_end"]))].append(record)
    windows = []
    for (sequence, start, end), rows in sorted(grouped.items()):
        merged = {"sequence_id": sequence, "frame_start": start, "frame_end": end}
        for family, field in FAMILIES.items():
            values = [float(row.get(field, np.nan)) for row in rows]
            finite = [value for value in values if np.isfinite(value)]
            merged[field] = max(finite) if finite else 0.0
        for field in ("gt_speed", "gt_turn", "confidence"):
            values = [float(row.get(field, np.nan)) for row in rows]
            finite = [value for value in values if np.isfinite(value)]
            merged[field] = float(np.mean(finite)) if finite else 0.0
        windows.append(merged)
    if not windows:
        return []
    for family, field in FAMILIES.items():
        z = robust_zscore(row[field] for row in windows)
        for row, value in zip(windows, z, strict=True):
            row[f"{family}_z"] = max(0.0, float(value))
    for row in windows:
        row["score"] = sum(WEIGHTS[family] * row[f"{family}_z"] for family in FAMILIES)

    candidates: dict[tuple[str, int, int], dict] = {}
    def add(row, reason, bonus=0.0):
        key = (row["sequence_id"], row["frame_start"], row["frame_end"])
        item = candidates.setdefault(key, {**row, "reasons": set(), "priority": row["score"]})
        item["reasons"].add(reason)
        item["priority"] = max(item["priority"], row["score"] + bonus)

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in windows:
        by_sequence[row["sequence_id"]].append(row)
    for sequence, rows in by_sequence.items():
        ordered = sorted(rows, key=lambda row: (-row["trajectory_z"], row["frame_start"]))
        count = 3 if sequence in RECOVERY else 2 if sequence in GUARD else 1
        for row in ordered[:count]:
            add(row, "trajectory", 3.0 if sequence in RECOVERY | GUARD else 0.0)
        # Matched low-anomaly control nearest in speed, turn and confidence to the strongest event.
        anchor = ordered[0]
        median_score = np.median([row["score"] for row in rows])
        pool = [row for row in rows if row["score"] <= median_score and row is not anchor]
        if pool:
            scales = {field: max(np.std([row[field] for row in rows]), 1e-6) for field in ("gt_speed", "gt_turn", "confidence")}
            control = min(pool, key=lambda row: (
                sum(abs(row[field] - anchor[field]) / scales[field] for field in scales), row["frame_start"]
            ))
            add(control, "control", 1.0)
        # Largest trajectory change point.
        temporal = sorted(rows, key=lambda row: row["frame_start"])
        if len(temporal) > 1:
            index = max(range(1, len(temporal)), key=lambda i: (abs(temporal[i]["trajectory_regret"] - temporal[i - 1]["trajectory_regret"]), -temporal[i]["frame_start"]))
            add(temporal[index], "trajectory_change")
    for family in FAMILIES:
        for row in sorted(windows, key=lambda row: (-row[f"{family}_z"], row["sequence_id"], row["frame_start"]))[:6]:
            add(row, family, 2.0)

    expanded = []
    for item in candidates.values():
        stride = _stride(by_sequence[item["sequence_id"]])
        expanded.append({
            **item,
            "start": max(0, item["frame_start"] - context_windows * stride),
            "end": item["frame_end"] + context_windows * stride,
            "stride": stride,
        })
    expanded.sort(key=lambda item: (item["sequence_id"], item["start"], item["end"]))
    merged_intervals: list[dict] = []
    for item in expanded:
        if merged_intervals and item["sequence_id"] == merged_intervals[-1]["sequence_id"] and item["start"] <= merged_intervals[-1]["end"] + item["stride"]:
            current = merged_intervals[-1]
            current["end"] = max(current["end"], item["end"])
            current["reasons"].update(item["reasons"])
            current["priority"] = max(current["priority"], item["priority"])
        else:
            merged_intervals.append({**item, "reasons": set(item["reasons"])})

    # Mandatory sequence coverage first, followed by the strongest remaining anomalies.
    mandatory = RECOVERY | GUARD
    ranked = sorted(merged_intervals, key=lambda item: (-(item["sequence_id"] in mandatory), -item["priority"], item["sequence_id"], item["start"]))
    chosen = ranked[:limit]
    chosen.sort(key=lambda item: (item["sequence_id"], item["start"]))
    return [
        SelectedInterval(
            sequence_id=item["sequence_id"], start_frame=int(item["start"]),
            end_frame=int(item["end"]), reasons=tuple(sorted(item["reasons"])),
            score=float(item["priority"]),
        )
        for item in chosen
    ]
