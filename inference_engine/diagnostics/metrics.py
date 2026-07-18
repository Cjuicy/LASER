"""Official comparison, stability guard, and recovery summaries."""

from __future__ import annotations

from typing import Any

import numpy as np


OFFICIAL_CONFIGS = ("depth", "geometry_baseline", "layer_atomic_split")
GUARD_SEQUENCES = ("00", "05", "09")
RECOVERY_SEQUENCES = ("02", "04", "10")


def recovery_score(
    depth_ate: float,
    split_ate: float,
    geometry_ate: float,
    *,
    eps: float = 1e-9,
) -> dict:
    denominator = float(depth_ate - geometry_ate)
    if not np.isfinite(denominator) or denominator <= eps:
        return {"valid": False, "score": None, "invalid_reason": "non_positive_recovery_gap"}
    score = (float(depth_ate) - float(split_ate)) / denominator
    return {
        "valid": bool(np.isfinite(score)),
        "score": float(score),
        "invalid_reason": None,
    }


def evaluate_stability_guard(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    mean_regression_limit: float = .03,
    median_regression_limit: float = 0.0,
    guard_sequence_limit: float = .10,
    expected_sequences: tuple[str, ...] | list[str] | None = None,
) -> dict:
    sequences = sorted(set(candidate) & set(baseline))
    reasons: list[str] = []
    if expected_sequences is not None:
        reasons.extend(
            f"sequence_{sequence}_missing"
            for sequence in sorted(set(expected_sequences) - set(candidate))
        )
    if set(baseline) - set(candidate):
        reasons.append("missing_sequences")
    valid = [seq for seq in sequences if np.isfinite(candidate[seq]) and np.isfinite(baseline[seq]) and baseline[seq] > 0]
    if len(valid) != len(sequences):
        reasons.append("invalid_sequence_metric")
    if valid:
        candidate_values = np.asarray([candidate[seq] for seq in valid], dtype=float)
        baseline_values = np.asarray([baseline[seq] for seq in valid], dtype=float)
        mean_regression = float(candidate_values.mean() / baseline_values.mean() - 1)
        median_regression = float(np.median(candidate_values) / np.median(baseline_values) - 1)
        if mean_regression > mean_regression_limit:
            reasons.append("mean_ate_regression")
        if median_regression > median_regression_limit:
            reasons.append("median_ate_regression")
    else:
        mean_regression = median_regression = None
        reasons.append("no_valid_sequences")
    per_guard = {}
    for seq in GUARD_SEQUENCES:
        if seq not in candidate or seq not in baseline or baseline.get(seq, 0) <= 0:
            per_guard[seq] = None
            reasons.append(f"guard_{seq}_missing")
            continue
        regression = float(candidate[seq] / baseline[seq] - 1)
        per_guard[seq] = regression
        if regression > guard_sequence_limit:
            reasons.append(f"guard_{seq}_regression")
    return {
        "passed": not reasons,
        "baseline_config": "depth",
        "failure_reasons": sorted(set(reasons)),
        "mean_regression": mean_regression,
        "median_regression": median_regression,
        "guard_sequence_regression": per_guard,
        "thresholds": {"mean": mean_regression_limit, "median": median_regression_limit, "per_guard": guard_sequence_limit},
    }


def build_sequence_summary(results: dict[str, dict[str, dict[str, Any]]]) -> dict:
    sequences = sorted({sequence for config in results.values() for sequence in config})
    ranking: dict[str, list[dict[str, Any]]] = {}
    for sequence in sequences:
        rows = []
        for config_id in OFFICIAL_CONFIGS:
            metric = results.get(config_id, {}).get(sequence, {})
            ate = metric.get("ate_rmse")
            if metric.get("valid") and ate is not None and np.isfinite(ate):
                rows.append({"config_id": config_id, "ate_rmse": float(ate)})
        ranking[sequence] = sorted(rows, key=lambda row: (row["ate_rmse"], row["config_id"]))
    aggregates = {}
    for config_id in OFFICIAL_CONFIGS:
        values = [row["ate_rmse"] for sequence in sequences for row in ranking[sequence] if row["config_id"] == config_id]
        aggregates[config_id] = {
            "mean_ate": float(np.mean(values)) if values else None,
            "median_ate": float(np.median(values)) if values else None,
            "valid_sequences": len(values),
            "wins": sum(bool(ranking[seq]) and ranking[seq][0]["config_id"] == config_id for seq in sequences),
        }
    recovery = {}
    for sequence in RECOVERY_SEQUENCES:
        depth = results.get("depth", {}).get(sequence, {}).get("ate_rmse")
        split = results.get("layer_atomic_split", {}).get(sequence, {}).get("ate_rmse")
        geometry = results.get("geometry_baseline", {}).get(sequence, {}).get("ate_rmse")
        if depth is None or split is None or geometry is None:
            recovery[sequence] = {
                "valid": False,
                "score": None,
                "invalid_reason": "missing_ate",
            }
        else:
            recovery[sequence] = recovery_score(depth, split, geometry)
    return {
        "official_ranking": ranking,
        "official_aggregate": aggregates,
        "recovery": recovery,
    }
