"""Deterministic multi-signal diagnostic interval selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .schema import SelectedInterval


FAMILIES = {
    "trajectory": "trajectory_regret",
    "merge": "merge_anomaly",
    "immutable_atom": "atom_anomaly",
    "scale": "scale_dispersion",
    "temporal": "temporal_churn",
}
WEIGHTS = {
    "trajectory": .40, "merge": .10, "immutable_atom": .10,
    "scale": .25, "temporal": .15,
}
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
        iqr = np.quantile(clean, .75) - np.quantile(clean, .25)
        scale = iqr / 1.349 if iqr > 1e-12 else np.std(clean)
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
        raise ValueError("interval limit must be positive")
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
        if sequence in GUARD:
            # Negative layer_atomic-minus-geometry regret is the stability
            # success we need to preserve on 00/05/09.
            ordered = sorted(rows, key=lambda row: (row["trajectory_regret"], row["frame_start"]))
        else:
            ordered = sorted(rows, key=lambda row: (-row["trajectory_z"], row["frame_start"]))
        count = 3 if sequence in RECOVERY else 2 if sequence in GUARD else 1
        for row in ordered[:count]:
            reason = "guard_success" if sequence in GUARD else "trajectory"
            add(row, reason, 3.0 if sequence in RECOVERY | GUARD else 0.0)
        # Matched low-anomaly control nearest in speed, turn and confidence to the strongest event.
        anchor = ordered[0]
        median_score = np.median([row["score"] for row in rows])
        median_regret = np.median([row["trajectory_regret"] for row in rows])
        pool = [
            row for row in rows
            if row["score"] <= median_score
            and row["trajectory_regret"] <= median_regret
            and row is not anchor
        ]
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
        is_control = "control" in item["reasons"]
        expanded.append({
            **item,
            "start": item["frame_start"] if is_control else max(0, item["frame_start"] - context_windows * stride),
            "end": item["frame_end"] if is_control else item["frame_end"] + context_windows * stride,
            "stride": stride,
        })
    expanded.sort(key=lambda item: (item["sequence_id"], item["start"], item["end"]))
    merged_intervals: list[dict] = []
    # Controls remain distinct comparison cases even when their context overlaps
    # an anomaly interval.  Anomaly candidates still obey the normal merge rule.
    controls = [item for item in expanded if "control" in item["reasons"]]
    for item in (item for item in expanded if "control" not in item["reasons"]):
        max_span = (
            int(item["frame_end"]) - int(item["frame_start"]) + 1
            + 2 * context_windows * int(item["stride"])
        )
        proposed_end = max(
            item["end"],
            merged_intervals[-1]["end"] if merged_intervals else item["end"],
        )
        merge_is_bounded = (
            bool(merged_intervals)
            and proposed_end - merged_intervals[-1]["start"] + 1 <= max_span
        )
        if (
            merged_intervals
            and item["sequence_id"] == merged_intervals[-1]["sequence_id"]
            and item["start"] <= merged_intervals[-1]["end"] + item["stride"]
            and merge_is_bounded
        ):
            current = merged_intervals[-1]
            current["end"] = proposed_end
            current["reasons"].update(item["reasons"])
            current["priority"] = max(current["priority"], item["priority"])
        else:
            merged_intervals.append({**item, "reasons": set(item["reasons"])})
    merged_intervals.extend({**item, "reasons": set(item["reasons"])} for item in controls)

    # Reserve mandatory sequence, signal-family, and control coverage before
    # filling the remaining weighted-ranking slots.
    mandatory = RECOVERY | GUARD
    ranked = sorted(
        merged_intervals,
        key=lambda item: (-item["priority"], item["sequence_id"], item["start"]),
    )
    chosen: list[dict] = []
    chosen_ids: set[tuple[str, int, int]] = set()
    def reserve(items, count=1):
        if count <= 0:
            return 0
        added = 0
        for item in items:
            identity = (item["sequence_id"], item["start"], item["end"])
            if identity not in chosen_ids and len(chosen) < limit:
                chosen.append(item); chosen_ids.add(identity)
                added += 1
                if added == count:
                    break
        return added
    for sequence in sorted(mandatory):
        reserve(
            item for item in ranked
            if item["sequence_id"] == sequence and "control" not in item["reasons"]
        )
        reserve(
            item for item in ranked
            if item["sequence_id"] == sequence and "control" in item["reasons"]
        )
    for reason in ("merge", "immutable_atom", "scale", "temporal"):
        already = sum(reason in item["reasons"] for item in chosen)
        reserve((item for item in ranked if reason in item["reasons"]), max(0, 2 - already))
    for item in ranked:
        identity = (item["sequence_id"], item["start"], item["end"])
        if identity not in chosen_ids and len(chosen) < limit:
            chosen.append(item); chosen_ids.add(identity)
    missing = []
    for sequence in sorted(mandatory & set(by_sequence)):
        sequence_items = [item for item in chosen if item["sequence_id"] == sequence]
        if not any("control" not in item["reasons"] for item in sequence_items):
            missing.append(f"{sequence}:event")
        if not any("control" in item["reasons"] for item in sequence_items):
            missing.append(f"{sequence}:control")
    for reason in ("merge", "immutable_atom", "scale", "temporal"):
        available = sum(reason in item["reasons"] for item in ranked)
        required = min(2, available)
        actual = sum(reason in item["reasons"] for item in chosen)
        if actual < required:
            missing.append(f"{reason}:{actual}/{required}")
    if missing:
        raise ValueError(
            f"interval limit {limit} cannot satisfy mandatory diagnostic diversity: "
            + ", ".join(missing)
        )
    chosen.sort(key=lambda item: (item["sequence_id"], item["start"]))
    return [
        SelectedInterval(
            sequence_id=item["sequence_id"], start_frame=int(item["start"]),
            end_frame=int(item["end"]), reasons=tuple(sorted(item["reasons"])),
            score=float(item["priority"]),
        )
        for item in chosen
    ]
