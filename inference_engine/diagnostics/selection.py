"""Deterministic multi-signal diagnostic interval selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .schema import SelectedInterval


FAMILIES = {
    "merge": "merge_anomaly",
    "immutable_atom": "atom_anomaly",
    "scale": "scale_dispersion",
    "temporal": "temporal_churn",
}
REGRET_FIELDS = (
    "split_minus_depth_regret",
    "split_minus_geometry_regret",
)
WEIGHTS = {
    "trajectory": .35,
    "split": .20,
    "merge": .10,
    "immutable_atom": .10,
    "scale": .15,
    "temporal": .10,
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


def _finite_values(rows: list[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(field, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _number(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _stride(rows: list[dict]) -> int:
    starts = sorted({int(row["frame_start"]) for row in rows})
    differences = np.diff(starts)
    positive = differences[differences > 0]
    if positive.size:
        return int(np.median(positive))
    return max(1, int(rows[0]["frame_end"]) - int(rows[0]["frame_start"]) + 1)


def select_intervals(
    records: list[dict], *, limit: int = 48, context_windows: int = 2,
) -> list[SelectedInterval]:
    if limit <= 0:
        raise ValueError("interval limit must be positive")

    # Union configurations at the same global window so every method gets
    # identical cases. Split activity is sourced exclusively from the split
    # profile; baseline structural maxima must not manufacture split events.
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(
            str(record["sequence_id"]),
            int(record["frame_start"]),
            int(record["frame_end"]),
        )].append(record)
    windows = []
    for (sequence, start, end), rows in sorted(grouped.items()):
        merged = {"sequence_id": sequence, "frame_start": start, "frame_end": end}
        for field in REGRET_FIELDS:
            values = _finite_values(rows, field)
            merged[field] = float(np.mean(values)) if values else 0.0
        for field in FAMILIES.values():
            values = _finite_values(rows, field)
            merged[field] = max(values) if values else 0.0
        for field in ("gt_speed", "gt_turn", "confidence"):
            values = _finite_values(rows, field)
            merged[field] = float(np.mean(values)) if values else 0.0
        split_rows = [row for row in rows if row.get("config_id") == "layer_atomic_split"]
        accepted = _finite_values(split_rows, "split_accepted_count")
        changed = _finite_values(split_rows, "split_changed_pixel_ratio")
        merged["split_accepted_count"] = max(accepted) if accepted else None
        merged["split_changed_pixel_ratio"] = max(changed) if changed else None
        merged["split_activity"] = (
            _number(merged["split_accepted_count"], 0.0)
            + _number(merged["split_changed_pixel_ratio"], 0.0)
            if accepted or changed else None
        )
        windows.append(merged)
    if not windows:
        return []

    regret_zscores = [
        np.abs(robust_zscore(row[field] for row in windows))
        for field in REGRET_FIELDS
    ]
    accepted_z = robust_zscore(
        _number(row["split_accepted_count"]) for row in windows
    )
    changed_z = robust_zscore(
        _number(row["split_changed_pixel_ratio"]) for row in windows
    )
    for index, row in enumerate(windows):
        row["trajectory_z"] = max(float(values[index]) for values in regret_zscores)
        row["split_z"] = max(0.0, float(accepted_z[index]), float(changed_z[index]))
    for family, field in FAMILIES.items():
        zscores = robust_zscore(row[field] for row in windows)
        for row, value in zip(windows, zscores, strict=True):
            row[f"{family}_z"] = max(0.0, float(value))
    for row in windows:
        row["score"] = (
            WEIGHTS["trajectory"] * row["trajectory_z"]
            + WEIGHTS["split"] * row["split_z"]
            + sum(WEIGHTS[family] * row[f"{family}_z"] for family in FAMILIES)
        )

    candidates: dict[tuple[str, int, int], dict] = {}

    def add(row, reason, bonus=0.0):
        key = (row["sequence_id"], row["frame_start"], row["frame_end"])
        item = candidates.setdefault(
            key, {**row, "reasons": set(), "priority": row["score"]},
        )
        item["reasons"].add(reason)
        item["priority"] = max(item["priority"], row["score"] + bonus)

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in windows:
        by_sequence[row["sequence_id"]].append(row)
    for sequence, rows in by_sequence.items():
        count = 3 if sequence in RECOVERY else 2 if sequence in GUARD else 1
        degradation = sorted(
            (row for row in rows if max(row[field] for field in REGRET_FIELDS) > 0),
            key=lambda row: (
                -max(row[field] for field in REGRET_FIELDS), row["frame_start"],
            ),
        )
        improvement = sorted(
            (row for row in rows if min(row[field] for field in REGRET_FIELDS) < 0),
            key=lambda row: (
                min(row[field] for field in REGRET_FIELDS), row["frame_start"],
            ),
        )
        focus_bonus = 3.0 if sequence in RECOVERY | GUARD else 0.0
        for row in degradation[:count]:
            add(row, "trajectory_degradation", focus_bonus + 2.0)
        for row in improvement[:count]:
            add(row, "trajectory_improvement", focus_bonus + 2.0)

        temporal = sorted(rows, key=lambda row: row["frame_start"])
        if len(temporal) > 1:
            deltas = [
                max(
                    abs(temporal[index][field] - temporal[index - 1][field])
                    for field in REGRET_FIELDS
                )
                for index in range(1, len(temporal))
            ]
            index = max(
                range(1, len(temporal)),
                key=lambda value: (
                    deltas[value - 1], -temporal[value]["frame_start"],
                ),
            )
            add(temporal[index], "trajectory_change", 1.0 + deltas[index - 1])

        active = [
            row for row in rows if _number(row["split_accepted_count"], -1.0) > 0
        ]
        if active:
            observed_activity = [
                _number(row["split_activity"])
                for row in rows if np.isfinite(_number(row["split_activity"]))
            ]
            median_activity = float(np.median(observed_activity))
            median_regret = float(np.median([
                max(abs(row[field]) for field in REGRET_FIELDS) for row in rows
            ]))
            no_effect = [
                row for row in active
                if row["split_activity"] > median_activity
                and max(abs(row[field]) for field in REGRET_FIELDS) <= median_regret
            ]
            if no_effect:
                row = min(
                    no_effect,
                    key=lambda item: (
                        max(abs(item[field]) for field in REGRET_FIELDS),
                        -item["split_activity"], item["frame_start"],
                    ),
                )
                add(row, "split_no_trajectory_effect", 1.5 + row["split_z"])

        anchor = max(
            rows,
            key=lambda row: (
                row["split_z"], _number(row["split_accepted_count"], -1.0),
                _number(row["split_changed_pixel_ratio"], -1.0),
                -row["frame_start"],
            ),
        )
        pool = [
            row for row in rows
            if _number(row["split_accepted_count"], -1.0) == 0 and row is not anchor
        ]
        if pool:
            scales = {
                field: max(float(np.std([row[field] for row in rows])), 1e-6)
                for field in ("gt_speed", "gt_turn", "confidence")
            }
            control = min(pool, key=lambda row: (
                sum(abs(row[field] - anchor[field]) / scales[field] for field in scales),
                row["frame_start"],
            ))
            add(control, "matched_control", 1.0)

    for row in sorted(
        windows,
        key=lambda row: (-row["split_z"], row["sequence_id"], row["frame_start"]),
    )[:6]:
        if (
            _number(row["split_accepted_count"], -1.0) > 0
            or _number(row["split_changed_pixel_ratio"], -1.0) > 0
        ):
            add(row, "split_anomaly", 2.0 + row["split_z"])
    for family in FAMILIES:
        for row in sorted(
            windows,
            key=lambda row: (-row[f"{family}_z"], row["sequence_id"], row["frame_start"]),
        )[:6]:
            add(row, family, 2.0)

    expanded = []
    for item in candidates.values():
        stride = _stride(by_sequence[item["sequence_id"]])
        is_control = "matched_control" in item["reasons"]
        expanded.append({
            **item,
            "start": item["frame_start"] if is_control else max(
                0, item["frame_start"] - context_windows * stride,
            ),
            "end": item["frame_end"] if is_control else (
                item["frame_end"] + context_windows * stride
            ),
            "stride": stride,
        })
    expanded.sort(key=lambda item: (item["sequence_id"], item["start"], item["end"]))
    merged_intervals: list[dict] = []
    controls = [item for item in expanded if "matched_control" in item["reasons"]]
    for item in (item for item in expanded if "matched_control" not in item["reasons"]):
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
    merged_intervals.extend(
        {**item, "reasons": set(item["reasons"])} for item in controls
    )

    # Reserve required sequence and signal-family coverage before filling the
    # remaining weighted-ranking slots.
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
                chosen.append(item)
                chosen_ids.add(identity)
                added += 1
                if added == count:
                    break
        return added

    for sequence in sorted(mandatory):
        reserve(
            item for item in ranked
            if item["sequence_id"] == sequence
            and "matched_control" not in item["reasons"]
        )
        reserve(
            item for item in ranked
            if item["sequence_id"] == sequence
            and "matched_control" in item["reasons"]
        )
    for reason in (
        "trajectory_degradation", "trajectory_improvement", "trajectory_change",
        "split_anomaly", "split_no_trajectory_effect", "matched_control",
    ):
        reserve(item for item in ranked if reason in item["reasons"])
    for reason in FAMILIES:
        already = sum(reason in item["reasons"] for item in chosen)
        reserve(
            (item for item in ranked if reason in item["reasons"]),
            max(0, 2 - already),
        )
    for item in ranked:
        identity = (item["sequence_id"], item["start"], item["end"])
        if identity not in chosen_ids and len(chosen) < limit:
            chosen.append(item)
            chosen_ids.add(identity)

    missing = []
    for sequence in sorted(mandatory & set(by_sequence)):
        sequence_items = [item for item in chosen if item["sequence_id"] == sequence]
        if not any("matched_control" not in item["reasons"] for item in sequence_items):
            missing.append(f"{sequence}:event")
        if not any("matched_control" in item["reasons"] for item in sequence_items):
            missing.append(f"{sequence}:control")
    for reason in FAMILIES:
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
            sequence_id=item["sequence_id"],
            start_frame=int(item["start"]),
            end_frame=int(item["end"]),
            reasons=tuple(sorted(item["reasons"])),
            score=float(item["priority"]),
        )
        for item in chosen
    ]
