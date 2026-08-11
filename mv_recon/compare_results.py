from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

from mv_recon.geometry_metrics import PrimaryMetrics
from mv_recon.protocol import (
    EXPECTED_DATASET_SEQUENCE_COUNTS,
    PAPER_REFERENCE_VALUES,
    PAPER_SEQUENCE_MAP_PATHS,
    load_sequence_map,
)
from mv_recon.results import METRIC_SCHEMA_VERSION


DATASET = "NRGBD-dense"
METHODS = ("depth", "geometry", "atomic")
PRIMARY_FIELDS = tuple(field.name for field in fields(PrimaryMetrics))
DISTANCE_FIELDS = frozenset(
    {
        "accuracy_mean_m",
        "accuracy_median_m",
        "completion_mean_m",
        "completion_median_m",
        "chamfer_l1_m",
    }
)


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
        value,
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
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"JSON values must be finite; got {value}")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"comparison input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid comparison JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"comparison JSON root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _finite_tree(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    _finite_number(value, label)


def _primary(value: object, label: str) -> dict[str, float]:
    mapping = _mapping(value, label)
    if set(mapping) != set(PRIMARY_FIELDS):
        raise ValueError(
            f"{label} metric fields must be exactly {PRIMARY_FIELDS}"
        )
    return {
        field: _finite_number(mapping[field], f"{label}.{field}")
        for field in PRIMARY_FIELDS
    }


def _nrgbd_sequence_names() -> tuple[str, ...]:
    root = Path(__file__).resolve().parents[1]
    path = root / PAPER_SEQUENCE_MAP_PATHS[DATASET]
    specs = load_sequence_map(
        path,
        expected_count=EXPECTED_DATASET_SEQUENCE_COUNTS[DATASET],
    )
    return tuple(spec.name for spec in specs)


def _load_run(run_dir: Path, expected_method: str) -> dict[str, object]:
    if expected_method not in METHODS:
        raise ValueError(f"unsupported comparison method: {expected_method}")
    selected_dir = Path(run_dir)
    results = _read_object(selected_dir / "results.json")
    manifest = _read_object(selected_dir / "protocol_manifest.json")
    if results.get("state") != "subset" or results.get("subset") is not True:
        raise ValueError("comparison run is not a successful subset")
    failures = _list(results.get("failures"), "results.failures")
    if failures:
        raise ValueError("comparison run contains failures")
    if results.get("schema_version") != METRIC_SCHEMA_VERSION:
        raise ValueError("metric schema mismatch in results")
    identity = _mapping(results.get("identity"), "results.identity")
    if identity.get("metric_version") != METRIC_SCHEMA_VERSION:
        raise ValueError("metric schema mismatch in result identity")
    if manifest.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
        raise ValueError("metric schema mismatch in protocol manifest")
    if manifest.get("run_state") != results.get("state"):
        raise ValueError("results and manifest run-state identity disagree")
    if manifest.get("evaluation_mode") != "comparison":
        raise ValueError("run is not a comparison profile")
    if manifest.get("segmentation_method") != expected_method:
        raise ValueError(
            f"comparison method mismatch: expected {expected_method}"
        )
    pipeline = _mapping(manifest.get("pipeline"), "manifest.pipeline")
    segmentation = _mapping(
        pipeline.get("segmentation"), "manifest.pipeline.segmentation"
    )
    if segmentation.get("method") != expected_method:
        raise ValueError("pipeline segmentation method mismatch")
    prediction_cache = _mapping(
        pipeline.get("prediction_cache"),
        "manifest.pipeline.prediction_cache",
    )
    expected_cache_mode = "auto" if expected_method == "depth" else "readonly"
    if prediction_cache.get("mode") != expected_cache_mode:
        raise ValueError(
            f"prediction-cache mode mismatch for {expected_method}"
        )
    result_checkpoint = _digest(
        identity.get("checkpoint_sha256"),
        "results.identity.checkpoint_sha256",
    )
    manifest_checkpoint = _digest(
        manifest.get("checkpoint_sha256"),
        "manifest.checkpoint_sha256",
    )
    if result_checkpoint != manifest_checkpoint:
        raise ValueError("checkpoint identity disagreement between results and manifest")
    result_maps = _mapping(
        identity.get("sequence_map_sha256"),
        "results.identity.sequence_map_sha256",
    )
    manifest_maps = _mapping(
        manifest.get("sequence_map_sha256"),
        "manifest.sequence_map_sha256",
    )
    if result_maps != manifest_maps:
        raise ValueError("sequence-map identity disagreement between results and manifest")
    for dataset, digest in result_maps.items():
        _digest(digest, f"sequence map {dataset}")
    protocol_identity = _digest(
        identity.get("protocol_identity_sha256"),
        "results.identity.protocol_identity_sha256",
    )
    if protocol_identity != _digest(
        manifest.get("protocol_identity_sha256"),
        "manifest.protocol_identity_sha256",
    ):
        raise ValueError("protocol identity disagreement between results and manifest")
    pipeline_identity = _digest(
        identity.get("pipeline_sha256"),
        "results.identity.pipeline_sha256",
    )
    if pipeline_identity != _digest(
        manifest.get("resolved_pipeline_sha256"),
        "manifest.resolved_pipeline_sha256",
    ):
        raise ValueError("pipeline identity disagreement between results and manifest")
    resolved_protocol = _digest(
        manifest.get("resolved_protocol_sha256"),
        "manifest.resolved_protocol_sha256",
    )
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit:
        raise ValueError("manifest.git_commit must be a non-empty string")
    runtime = dict(_mapping(manifest.get("runtime"), "manifest.runtime"))
    if not runtime:
        raise ValueError("manifest.runtime must not be empty")
    return {
        "run_dir": selected_dir,
        "method": expected_method,
        "results": results,
        "manifest": manifest,
        "checkpoint_sha256": result_checkpoint,
        "sequence_map_sha256": dict(result_maps),
        "protocol_identity_sha256": protocol_identity,
        "resolved_protocol_sha256": resolved_protocol,
        "pipeline_sha256": pipeline_identity,
        "git_commit": git_commit,
        "runtime": runtime,
    }


def _sequence_thresholds(
    sequence: Mapping[str, object],
    label: str,
) -> tuple[dict[str, float], ...]:
    diagnostics = _mapping(sequence.get("diagnostics"), f"{label}.diagnostics")
    _finite_tree(diagnostics, f"{label}.diagnostics")
    thresholds = _list(
        diagnostics.get("thresholds"),
        f"{label}.diagnostics.thresholds",
    )
    parsed = []
    for index, value in enumerate(thresholds):
        threshold = _mapping(value, f"{label}.thresholds[{index}]")
        expected_fields = {"threshold_m", "precision", "recall", "fscore"}
        if set(threshold) != expected_fields:
            raise ValueError("threshold metric fields mismatch")
        parsed.append(
            {
                name: _finite_number(
                    threshold[name], f"{label}.thresholds[{index}].{name}"
                )
                for name in expected_fields
            }
        )
    if not parsed:
        raise ValueError("threshold metrics must not be empty")
    return tuple(parsed)


def _validate_cache_diagnostics(
    sequence: Mapping[str, object],
    method: str,
    label: str,
) -> tuple[int, int]:
    cache = _mapping(
        sequence.get("cache_diagnostics"),
        f"{label}.cache_diagnostics",
    )
    hits = _nonnegative_integer(cache.get("ordinary_hits"), f"{label}.ordinary_hits")
    misses = _nonnegative_integer(
        cache.get("ordinary_misses"), f"{label}.ordinary_misses"
    )
    if method != "depth" and (misses != 0 or hits < 1):
        raise ValueError(
            f"{method} readonly ordinary cache must have hits and zero misses"
        )
    cache_key = cache.get("ordinary_prediction_key")
    if cache_key is not None and cache_key != sequence.get("ordinary_prediction_key"):
        raise ValueError("ordinary prediction key disagrees with cache diagnostics")
    return hits, misses


def _validate_coverage(
    run: Mapping[str, object],
    expected_sequences: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    expected = tuple(str(name) for name in expected_sequences)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected comparison sequences must be unique")
    results = _mapping(run["results"], "results")
    manifest = _mapping(run["manifest"], "manifest")
    expected_mapping = _mapping(
        results.get("expected_sequences"), "results.expected_sequences"
    )
    selected_mapping = _mapping(
        manifest.get("selected_sequences"), "manifest.selected_sequences"
    )
    if (
        set(expected_mapping) != {DATASET}
        or tuple(_list(expected_mapping.get(DATASET), "expected sequences"))
        != expected
        or set(selected_mapping) != {DATASET}
        or tuple(_list(selected_mapping.get(DATASET), "selected sequences"))
        != expected
    ):
        raise ValueError("comparison sequence coverage mismatch")
    raw_sequences = _list(results.get("sequences"), "results.sequences")
    sequences = tuple(
        _mapping(value, f"results.sequences[{index}]")
        for index, value in enumerate(raw_sequences)
    )
    observed = tuple(
        (sequence.get("dataset"), sequence.get("sequence"))
        for sequence in sequences
    )
    wanted = tuple((DATASET, name) for name in expected)
    if observed != wanted:
        raise ValueError("comparison sequence coverage or order mismatch")
    full_counts = _mapping(
        results.get("full_sequence_counts"), "results.full_sequence_counts"
    )
    selected_counts = _mapping(
        manifest.get("selected_sequence_counts"),
        "manifest.selected_sequence_counts",
    )
    expected_counts = _mapping(
        manifest.get("expected_sequence_counts"),
        "manifest.expected_sequence_counts",
    )
    full_count = EXPECTED_DATASET_SEQUENCE_COUNTS[DATASET]
    if (
        full_counts != {DATASET: full_count}
        or selected_counts != {DATASET: len(expected)}
        or expected_counts != {DATASET: full_count}
    ):
        raise ValueError("comparison sequence count metadata mismatch")
    datasets = _list(results.get("datasets"), "results.datasets")
    if len(datasets) != 1:
        raise ValueError("comparison must contain exactly one dataset summary")
    summary = _mapping(datasets[0], "results.datasets[0]")
    if (
        summary.get("dataset") != DATASET
        or summary.get("status") != "subset"
        or summary.get("expected_sequences") != full_count
        or summary.get("selected_sequences") != len(expected)
        or summary.get("completed_sequences") != len(expected)
    ):
        raise ValueError("comparison dataset summary coverage mismatch")
    method = str(run["method"])
    sequence_primaries = []
    for index, sequence in enumerate(sequences):
        label = f"results.sequences[{index}]"
        _nonnegative_integer(sequence.get("frame_count"), f"{label}.frame_count")
        _digest(sequence.get("input_manifest_sha256"), f"{label}.input hash")
        _digest(sequence.get("ground_truth_sha256"), f"{label}.GT hash")
        _digest(
            sequence.get("ordinary_prediction_key"),
            f"{label}.ordinary prediction key",
        )
        sequence_primaries.append(_primary(sequence.get("primary"), f"{label}.primary"))
        _sequence_thresholds(sequence, label)
        _validate_cache_diagnostics(sequence, method, label)
    dataset_primary = _primary(summary.get("primary"), "dataset.primary")
    for field in PRIMARY_FIELDS:
        aggregate = math.fsum(primary[field] for primary in sequence_primaries) / len(
            sequence_primaries
        )
        if not math.isclose(
            dataset_primary[field], aggregate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"dataset primary aggregation mismatch for {field}")
    return sequences


def _dataset_primary(run: Mapping[str, object]) -> dict[str, float]:
    results = _mapping(run["results"], "results")
    datasets = _list(results.get("datasets"), "results.datasets")
    summary = _mapping(datasets[0], "results.datasets[0]")
    return _primary(summary.get("primary"), "dataset.primary")


def _validate_depth_run(
    run: Mapping[str, object],
    expected_sequences: Sequence[str],
) -> dict[str, float]:
    _validate_coverage(run, expected_sequences)
    primary = _dataset_primary(run)
    reference = PAPER_REFERENCE_VALUES[DATASET]
    mismatches = {
        name: (format(primary[name], ".3f"), format(reference[name], ".3f"))
        for name in PRIMARY_FIELDS
        if format(primary[name], ".3f") != format(reference[name], ".3f")
    }
    if mismatches:
        raise ValueError(f"Depth reproduction gate failed: {mismatches}")
    return primary


def validate_depth_gate(
    run_dir: Path,
    *,
    expected_sequences: Sequence[str] | None = None,
) -> dict[str, float]:
    expected = tuple(expected_sequences or _nrgbd_sequence_names())
    run = _load_run(Path(run_dir), "depth")
    return _validate_depth_run(run, expected)


def _threshold_signature(
    sequence: Mapping[str, object], label: str
) -> tuple[float, ...]:
    return tuple(
        threshold["threshold_m"]
        for threshold in _sequence_thresholds(sequence, label)
    )


def _assert_comparable(
    runs: Sequence[Mapping[str, object]],
    sequences_by_method: Sequence[Sequence[Mapping[str, object]]],
) -> None:
    reference_run = runs[0]
    reference_sequences = sequences_by_method[0]
    for run, sequences in zip(runs[1:], sequences_by_method[1:]):
        method = str(run["method"])
        if run["checkpoint_sha256"] != reference_run["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch for {method}")
        if run["sequence_map_sha256"] != reference_run["sequence_map_sha256"]:
            raise ValueError(f"sequence map mismatch for {method}")
        if run["git_commit"] != reference_run["git_commit"]:
            raise ValueError(f"git commit mismatch for {method}")
        if run["runtime"] != reference_run["runtime"]:
            raise ValueError(f"runtime metadata mismatch for {method}")
        for index, (reference, candidate) in enumerate(
            zip(reference_sequences, sequences)
        ):
            for field, label in (
                ("frame_count", "frame count"),
                ("input_manifest_sha256", "input manifest"),
                ("ground_truth_sha256", "ground truth"),
                ("ordinary_prediction_key", "ordinary prediction"),
            ):
                if candidate.get(field) != reference.get(field):
                    raise ValueError(
                        f"{label} mismatch for {method} sequence {index}"
                    )
            reference_thresholds = _threshold_signature(
                reference, f"depth.sequences[{index}]"
            )
            candidate_thresholds = _threshold_signature(
                candidate, f"{method}.sequences[{index}]"
            )
            if candidate_thresholds != reference_thresholds:
                raise ValueError(
                    f"threshold mismatch for {method} sequence {index}"
                )


def _macro_diagnostics(
    sequences: Sequence[Mapping[str, object]],
    method: str,
) -> dict[str, object]:
    chamfer_values = []
    parsed_thresholds = []
    cache_hits = 0
    cache_misses = 0
    for index, sequence in enumerate(sequences):
        label = f"{method}.sequences[{index}]"
        diagnostics = _mapping(sequence.get("diagnostics"), f"{label}.diagnostics")
        chamfer_values.append(
            _finite_number(
                diagnostics.get("chamfer_l1_m"), f"{label}.chamfer_l1_m"
            )
        )
        parsed_thresholds.append(_sequence_thresholds(sequence, label))
        hits, misses = _validate_cache_diagnostics(sequence, method, label)
        cache_hits += hits
        cache_misses += misses
    signature = tuple(item["threshold_m"] for item in parsed_thresholds[0])
    if any(
        tuple(item["threshold_m"] for item in thresholds) != signature
        for thresholds in parsed_thresholds[1:]
    ):
        raise ValueError(f"threshold mismatch within {method} run")
    macro_thresholds = []
    for threshold_index, threshold_m in enumerate(signature):
        row = {"threshold_m": threshold_m}
        for metric in ("precision", "recall", "fscore"):
            row[metric] = math.fsum(
                thresholds[threshold_index][metric]
                for thresholds in parsed_thresholds
            ) / len(parsed_thresholds)
        macro_thresholds.append(row)
    return {
        "chamfer_l1_m": math.fsum(chamfer_values) / len(chamfer_values),
        "thresholds": macro_thresholds,
        "cache": {
            "ordinary_hits": cache_hits,
            "ordinary_misses": cache_misses,
        },
    }


def _threshold_label(threshold_m: float) -> str:
    centimeters = threshold_m * 100.0
    if math.isclose(centimeters, round(centimeters), abs_tol=1e-9):
        return f"{int(round(centimeters))}cm"
    return f"{threshold_m:g}m".replace(".", "p")


def _flatten_metrics(
    primary: Mapping[str, float], diagnostics: Mapping[str, object]
) -> dict[str, float]:
    flattened = dict(primary)
    flattened["chamfer_l1_m"] = float(diagnostics["chamfer_l1_m"])
    for threshold in diagnostics["thresholds"]:
        label = _threshold_label(float(threshold["threshold_m"]))
        for metric in ("precision", "recall", "fscore"):
            flattened[f"{metric}_at_{label}"] = float(threshold[metric])
    return flattened


def _metric_directions(metric_names: Sequence[str]) -> dict[str, str]:
    return {
        name: (
            "lower"
            if name in DISTANCE_FIELDS
            or name.startswith("accuracy_")
            or name.startswith("completion_")
            else "higher"
        )
        for name in metric_names
    }


def _depth_gate_report(
    run_dir: Path,
    expected: Sequence[str],
    primary: Mapping[str, float],
) -> dict[str, object]:
    reference = PAPER_REFERENCE_VALUES[DATASET]
    return {
        "schema_version": 1,
        "passed": True,
        "dataset": DATASET,
        "sequence_count": len(expected),
        "run_dir": str(Path(run_dir).resolve()),
        "comparison_precision_decimals": 3,
        "primary": dict(primary),
        "paper_reference": dict(reference),
        "displayed_primary": {
            key: format(primary[key], ".3f") for key in PRIMARY_FIELDS
        },
    }


def compare_run_directories(
    depth_dir: Path,
    geometry_dir: Path,
    atomic_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    expected = _nrgbd_sequence_names()
    runs = (
        _load_run(Path(depth_dir), "depth"),
        _load_run(Path(geometry_dir), "geometry"),
        _load_run(Path(atomic_dir), "atomic"),
    )
    depth_primary = _validate_depth_run(runs[0], expected)
    sequences_by_method = tuple(
        _validate_coverage(run, expected) for run in runs
    )
    _assert_comparable(runs, sequences_by_method)
    method_payloads = []
    flattened_by_method = []
    for run, sequences in zip(runs, sequences_by_method):
        method = str(run["method"])
        primary = _dataset_primary(run)
        diagnostics = _macro_diagnostics(sequences, method)
        flattened = _flatten_metrics(primary, diagnostics)
        flattened_by_method.append(flattened)
        method_payloads.append(
            {
                "method": method,
                "run_dir": str(Path(run["run_dir"]).resolve()),
                "state": "subset",
                "sequence_count": len(sequences),
                "primary": primary,
                "diagnostics": diagnostics,
                "delta_to_depth": {},
            }
        )
    metric_names = tuple(flattened_by_method[0])
    if any(tuple(metrics) != metric_names for metrics in flattened_by_method[1:]):
        raise ValueError("comparison metric field mismatch")
    depth_metrics = flattened_by_method[0]
    for payload, metrics in zip(method_payloads, flattened_by_method):
        payload["delta_to_depth"] = {
            name: metrics[name] - depth_metrics[name] for name in metric_names
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "dataset": DATASET,
        "sequence_count": len(expected),
        "sequences": list(expected),
        "checkpoint_sha256": runs[0]["checkpoint_sha256"],
        "sequence_map_sha256": runs[0]["sequence_map_sha256"],
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "provenance": {
            "git_commit": runs[0]["git_commit"],
            "runtime": runs[0]["runtime"],
            "methods": {
                str(run["method"]): {
                    "resolved_protocol_sha256": run[
                        "resolved_protocol_sha256"
                    ],
                    "protocol_identity_sha256": run[
                        "protocol_identity_sha256"
                    ],
                    "pipeline_sha256": run["pipeline_sha256"],
                }
                for run in runs
            },
        },
        "depth_gate": {
            "passed": True,
            "primary": depth_primary,
            "paper_reference": dict(PAPER_REFERENCE_VALUES[DATASET]),
        },
        "metric_directions": _metric_directions(metric_names),
        "methods": method_payloads,
    }
    rows = []
    for payload, metrics in zip(method_payloads, flattened_by_method):
        row: dict[str, object] = {"method": payload["method"], **metrics}
        row.update(
            {
                f"delta_to_depth_{name}": value
                for name, value in payload["delta_to_depth"].items()
            }
        )
        rows.append(row)
    output = Path(output_dir)
    _atomic_write_json(output / "comparison.json", report)
    _atomic_write_csv(output / "comparison.csv", rows, tuple(rows[0]))
    _atomic_write_json(
        output / "depth_gate.json",
        _depth_gate_report(Path(depth_dir), expected, depth_primary),
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and compare LASER NeuralRGBD point-map runs."
    )
    parser.add_argument("--depth-run", type=Path, required=True)
    parser.add_argument("--geometry-run", type=Path)
    parser.add_argument("--atomic-run", type=Path)
    parser.add_argument("--depth-gate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.depth_gate_only:
        if args.geometry_run is not None or args.atomic_run is not None:
            parser.error("--depth-gate-only does not accept later method runs")
        expected = _nrgbd_sequence_names()
        primary = validate_depth_gate(
            args.depth_run, expected_sequences=expected
        )
        _atomic_write_json(
            args.output_dir / "depth_gate.json",
            _depth_gate_report(args.depth_run, expected, primary),
        )
        return 0
    if args.geometry_run is None or args.atomic_run is None:
        parser.error("full comparison requires --geometry-run and --atomic-run")
    compare_run_directories(
        args.depth_run,
        args.geometry_run,
        args.atomic_run,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
