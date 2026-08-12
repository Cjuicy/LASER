from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from omegaconf import OmegaConf

from mv_recon.compare_results import (
    DATASET,
    METRIC_SCHEMA_VERSION,
    _atomic_write_csv,
    _atomic_write_json,
    _dataset_primary,
    _digest,
    _flatten_metrics,
    _list,
    _load_run,
    _macro_diagnostics,
    _mapping,
    _metric_directions,
    _nonnegative_integer,
    _nrgbd_sequence_names,
    _read_object,
    _threshold_signature,
    _validate_coverage,
)


NO_LOOP_MODE = "comparison"
NO_LOOP_ASSEMBLY = "laser-incremental-global-map-v1"
LOOP_MODE = "experiment"
LOOP_ASSEMBLY = "pipeline-loop-aggregate-v1"
INTERPRETATION_WARNING = (
    "Point-map assembly differs between runs; deltas combine traditional "
    "loop closure with the pipeline's delayed Sim(3) aggregation semantics "
    "and are not a pure loop-only causal estimate."
)
EXPECTED_GEOMETRY = {
    "center_crop_size": 224,
    "alignment": "umeyama_sim3_then_icp",
    "icp_type": "point_to_point",
    "icp_threshold_m": 0.1,
    "normal_estimation": "open3d_default",
    "fscore_thresholds_m": [0.01, 0.02, 0.05],
}


def _require_geometry(
    manifest: Mapping[str, object],
    run_dir: Path,
) -> None:
    raw_geometry = manifest.get("geometry")
    if raw_geometry is None:
        protocol_path = Path(run_dir) / "resolved_protocol.yaml"
        try:
            protocol_text = protocol_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "comparison input lacks geometry provenance: "
                f"{protocol_path}"
            ) from exc
        expected_digest = _digest(
            manifest.get("resolved_protocol_sha256"),
            "manifest.resolved_protocol_sha256",
        )
        observed_digest = hashlib.sha256(
            protocol_text.encode("utf-8")
        ).hexdigest()
        if observed_digest != expected_digest:
            raise ValueError("resolved protocol file hash mismatch")
        protocol = OmegaConf.to_container(
            OmegaConf.create(protocol_text),
            resolve=True,
        )
        raw_geometry = _mapping(protocol, "resolved protocol").get("geometry")
    geometry = dict(_mapping(raw_geometry, "protocol.geometry"))
    if geometry != EXPECTED_GEOMETRY:
        raise ValueError("point-map geometry evaluation configuration mismatch")


def _require_pipeline_settings(
    pipeline: Mapping[str, object],
    *,
    loop_enabled: bool,
) -> None:
    window = _mapping(pipeline.get("window"), "manifest.pipeline.window")
    if (window.get("size"), window.get("overlap")) != (20, 5):
        raise ValueError("point-map window configuration mismatch")

    prediction_cache = _mapping(
        pipeline.get("prediction_cache"),
        "manifest.pipeline.prediction_cache",
    )
    if prediction_cache.get("mode") != "auto":
        raise ValueError("point-map prediction-cache mode must be auto")

    segmentation = _mapping(
        pipeline.get("segmentation"),
        "manifest.pipeline.segmentation",
    )
    expected_segmentation = {
        "method": "depth",
        "confidence_keep_ratio": 0.5,
        "depth_merge_threshold": 0.1,
        "temporal_iou_threshold": 0.3,
    }
    for field, expected in expected_segmentation.items():
        if segmentation.get(field) != expected:
            raise ValueError(f"point-map segmentation {field} mismatch")
    felzenszwalb = _mapping(
        segmentation.get("felzenszwalb"),
        "manifest.pipeline.segmentation.felzenszwalb",
    )
    if (
        felzenszwalb.get("scale"),
        felzenszwalb.get("sigma"),
        felzenszwalb.get("min_size"),
    ) != (300, 1.1, 500):
        raise ValueError("point-map Felzenszwalb configuration mismatch")

    anchor = _mapping(
        pipeline.get("anchor_propagation"),
        "manifest.pipeline.anchor_propagation",
    )
    if (
        anchor.get("enabled"),
        anchor.get("correspondence_iou_threshold"),
    ) != (True, 0.4):
        raise ValueError("point-map anchor propagation mismatch")

    loop = _mapping(pipeline.get("loop"), "manifest.pipeline.loop")
    registration = _mapping(
        loop.get("registration"),
        "manifest.pipeline.loop.registration",
    )
    if loop.get("enabled") is not loop_enabled:
        raise ValueError("point-map loop enabled setting mismatch")
    if loop.get("method") != "traditional":
        raise ValueError("run does not use the traditional loop method")
    if registration.get("confidence_keep_ratio") != 0.5:
        raise ValueError("point-map loop confidence configuration mismatch")


def _load_traditional_loop_run(run_dir: Path) -> dict[str, object]:
    selected_dir = Path(run_dir)
    results = _read_object(selected_dir / "results.json")
    manifest = _read_object(selected_dir / "protocol_manifest.json")
    if results.get("state") != "subset" or results.get("subset") is not True:
        raise ValueError("traditional loop run is not a successful subset")
    if _list(results.get("failures"), "results.failures"):
        raise ValueError("traditional loop run contains failures")
    if results.get("schema_version") != METRIC_SCHEMA_VERSION:
        raise ValueError("metric schema mismatch in results")
    identity = _mapping(results.get("identity"), "results.identity")
    if identity.get("metric_version") != METRIC_SCHEMA_VERSION:
        raise ValueError("metric schema mismatch in result identity")
    if manifest.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
        raise ValueError("metric schema mismatch in protocol manifest")
    if manifest.get("run_state") != results.get("state"):
        raise ValueError("results and manifest run-state identity disagree")
    if manifest.get("evaluation_mode") != LOOP_MODE:
        raise ValueError("traditional loop run is not an experiment profile")
    if manifest.get("pointmap_assembly") != LOOP_ASSEMBLY:
        raise ValueError("traditional loop run uses the wrong point-map assembly")
    if manifest.get("segmentation_method") != "depth":
        raise ValueError("traditional loop run is not Depth segmentation")
    if manifest.get("loop_enabled") is not True:
        raise ValueError("traditional loop run does not enable loop closure")
    if manifest.get("loop_method") != "traditional":
        raise ValueError("run does not use the traditional loop method")
    _require_geometry(manifest, selected_dir)
    pipeline = _mapping(manifest.get("pipeline"), "manifest.pipeline")
    _require_pipeline_settings(pipeline, loop_enabled=True)

    result_checkpoint = _digest(
        identity.get("checkpoint_sha256"),
        "results.identity.checkpoint_sha256",
    )
    manifest_checkpoint = _digest(
        manifest.get("checkpoint_sha256"),
        "manifest.checkpoint_sha256",
    )
    if result_checkpoint != manifest_checkpoint:
        raise ValueError(
            "checkpoint identity disagreement between results and manifest"
        )
    result_maps = dict(
        _mapping(
            identity.get("sequence_map_sha256"),
            "results.identity.sequence_map_sha256",
        )
    )
    manifest_maps = dict(
        _mapping(
            manifest.get("sequence_map_sha256"),
            "manifest.sequence_map_sha256",
        )
    )
    if result_maps != manifest_maps:
        raise ValueError(
            "sequence-map identity disagreement between results and manifest"
        )
    for dataset, value in result_maps.items():
        _digest(value, f"sequence map {dataset}")
    protocol_identity = _digest(
        identity.get("protocol_identity_sha256"),
        "results.identity.protocol_identity_sha256",
    )
    if protocol_identity != _digest(
        manifest.get("protocol_identity_sha256"),
        "manifest.protocol_identity_sha256",
    ):
        raise ValueError(
            "protocol identity disagreement between results and manifest"
        )
    pipeline_identity = _digest(
        identity.get("pipeline_sha256"),
        "results.identity.pipeline_sha256",
    )
    if pipeline_identity != _digest(
        manifest.get("resolved_pipeline_sha256"),
        "manifest.resolved_pipeline_sha256",
    ):
        raise ValueError(
            "pipeline identity disagreement between results and manifest"
        )
    auxiliary_result = dict(
        _mapping(
            identity.get("auxiliary_checkpoint_sha256"),
            "results.identity.auxiliary_checkpoint_sha256",
        )
    )
    auxiliary_manifest = dict(
        _mapping(
            manifest.get("auxiliary_checkpoint_sha256"),
            "manifest.auxiliary_checkpoint_sha256",
        )
    )
    if auxiliary_result != auxiliary_manifest:
        raise ValueError(
            "auxiliary checkpoint identity disagreement between results and "
            "manifest"
        )
    if set(auxiliary_result) != {"salad", "dino"}:
        raise ValueError("traditional loop run must identify SALAD and DINO")
    for label, value in auxiliary_result.items():
        _digest(value, f"auxiliary checkpoint {label}")
    git_commit = manifest.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit:
        raise ValueError("manifest.git_commit must be a non-empty string")
    runtime = dict(_mapping(manifest.get("runtime"), "manifest.runtime"))
    if not runtime:
        raise ValueError("manifest.runtime must not be empty")
    return {
        "run_dir": selected_dir,
        "method": "depth",
        "results": results,
        "manifest": manifest,
        "checkpoint_sha256": result_checkpoint,
        "sequence_map_sha256": result_maps,
        "protocol_identity_sha256": protocol_identity,
        "resolved_protocol_sha256": _digest(
            manifest.get("resolved_protocol_sha256"),
            "manifest.resolved_protocol_sha256",
        ),
        "pipeline_sha256": pipeline_identity,
        "git_commit": git_commit,
        "runtime": runtime,
        "auxiliary_checkpoint_sha256": auxiliary_result,
    }


def _validate_no_loop_settings(run: Mapping[str, object]) -> None:
    manifest = _mapping(run["manifest"], "manifest")
    if manifest.get("evaluation_mode") != NO_LOOP_MODE:
        raise ValueError("no-loop run is not a comparison profile")
    if manifest.get("pointmap_assembly") != NO_LOOP_ASSEMBLY:
        raise ValueError("no-loop run uses the wrong point-map assembly")
    _require_geometry(manifest, Path(run["run_dir"]))
    pipeline = _mapping(manifest.get("pipeline"), "manifest.pipeline")
    _require_pipeline_settings(pipeline, loop_enabled=False)


def _loop_diagnostics(
    sequences: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    totals = {
        "candidate_count": 0,
        "constraint_count": 0,
        "rejected_candidate_count": 0,
        "joint_forward_count": 0,
        "used_no_loop_path_count": 0,
    }
    for index, sequence in enumerate(sequences):
        cache = _mapping(
            sequence.get("cache_diagnostics"),
            f"traditional_loop.sequences[{index}].cache_diagnostics",
        )
        candidate_count = _nonnegative_integer(
            cache.get("candidate_count"),
            f"traditional_loop.sequences[{index}].candidate_count",
        )
        constraint_count = _nonnegative_integer(
            cache.get("constraint_count"),
            f"traditional_loop.sequences[{index}].constraint_count",
        )
        rejected_count = _nonnegative_integer(
            cache.get("rejected_candidate_count"),
            f"traditional_loop.sequences[{index}].rejected_candidate_count",
        )
        joint_count = _nonnegative_integer(
            cache.get("joint_forward_count"),
            f"traditional_loop.sequences[{index}].joint_forward_count",
        )
        fallback = cache.get("used_no_loop_path")
        if not isinstance(fallback, bool):
            raise ValueError("used_no_loop_path must be a boolean")
        if candidate_count != constraint_count + rejected_count:
            raise ValueError(
                "candidate count must equal constraint count plus rejected "
                "candidate count"
            )
        if cache.get("loop_method") != "traditional":
            raise ValueError("sequence does not use the traditional loop method")
        totals["candidate_count"] += candidate_count
        totals["constraint_count"] += constraint_count
        totals["rejected_candidate_count"] += rejected_count
        totals["joint_forward_count"] += joint_count
        totals["used_no_loop_path_count"] += int(fallback)
    return totals


def _assert_loop_comparable(
    no_loop: Mapping[str, object],
    traditional_loop: Mapping[str, object],
    no_loop_sequences: Sequence[Mapping[str, object]],
    loop_sequences: Sequence[Mapping[str, object]],
) -> None:
    if traditional_loop["checkpoint_sha256"] != no_loop["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch for traditional loop")
    if traditional_loop["sequence_map_sha256"] != no_loop["sequence_map_sha256"]:
        raise ValueError("sequence map mismatch for traditional loop")
    if traditional_loop["runtime"] != no_loop["runtime"]:
        raise ValueError("runtime metadata mismatch for traditional loop")
    for index, (reference, candidate) in enumerate(
        zip(no_loop_sequences, loop_sequences)
    ):
        for field, label in (
            ("frame_count", "frame count"),
            ("input_manifest_sha256", "input manifest"),
            ("ground_truth_sha256", "ground truth"),
            ("ordinary_prediction_key", "ordinary prediction"),
        ):
            if candidate.get(field) != reference.get(field):
                raise ValueError(
                    f"{label} mismatch for traditional loop sequence {index}"
                )
        no_loop_thresholds = _threshold_signature(
            reference,
            f"no_loop.sequences[{index}]",
        )
        loop_thresholds = _threshold_signature(
            candidate,
            f"traditional_loop.sequences[{index}]",
        )
        if loop_thresholds != no_loop_thresholds:
            raise ValueError(
                f"threshold mismatch for traditional loop sequence {index}"
            )


def compare_loop_run_directories(
    no_loop_dir: Path,
    traditional_loop_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    expected = _nrgbd_sequence_names()
    no_loop = _load_run(Path(no_loop_dir), "depth")
    traditional_loop = _load_traditional_loop_run(Path(traditional_loop_dir))
    _validate_no_loop_settings(no_loop)
    no_loop_sequences = _validate_coverage(no_loop, expected)
    loop_sequences = _validate_coverage(traditional_loop, expected)
    _assert_loop_comparable(
        no_loop,
        traditional_loop,
        no_loop_sequences,
        loop_sequences,
    )
    loop_counts = _loop_diagnostics(loop_sequences)

    runs = (no_loop, traditional_loop)
    sequences_by_method = (no_loop_sequences, loop_sequences)
    labels = ("no_loop", "traditional_loop")
    method_payloads = []
    flattened_by_method = []
    for label, run, sequences in zip(labels, runs, sequences_by_method):
        primary = _dataset_primary(run)
        diagnostics = _macro_diagnostics(sequences, "depth")
        flattened = _flatten_metrics(primary, diagnostics)
        flattened_by_method.append(flattened)
        method_payloads.append(
            {
                "method": label,
                "run_dir": str(Path(run["run_dir"]).resolve()),
                "state": "subset",
                "sequence_count": len(sequences),
                "primary": primary,
                "diagnostics": diagnostics,
                "delta_to_no_loop": {},
            }
        )
    metric_names = tuple(flattened_by_method[0])
    if tuple(flattened_by_method[1]) != metric_names:
        raise ValueError("loop comparison metric field mismatch")
    baseline = flattened_by_method[0]
    for payload, metrics in zip(method_payloads, flattened_by_method):
        payload["delta_to_no_loop"] = {
            name: metrics[name] - baseline[name] for name in metric_names
        }

    report: dict[str, object] = {
        "schema_version": 1,
        "dataset": DATASET,
        "sequence_count": len(expected),
        "sequences": list(expected),
        "checkpoint_sha256": no_loop["checkpoint_sha256"],
        "sequence_map_sha256": no_loop["sequence_map_sha256"],
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "assemblies": {
            "no_loop": NO_LOOP_ASSEMBLY,
            "traditional_loop": LOOP_ASSEMBLY,
        },
        "interpretation_warning": INTERPRETATION_WARNING,
        "metric_directions": _metric_directions(metric_names),
        "loop_diagnostics": loop_counts,
        "provenance": {
            "runtime": no_loop["runtime"],
            "git_commits": {
                "no_loop": no_loop["git_commit"],
                "traditional_loop": traditional_loop["git_commit"],
            },
            "auxiliary_checkpoint_sha256": traditional_loop[
                "auxiliary_checkpoint_sha256"
            ],
            "methods": {
                label: {
                    "resolved_protocol_sha256": run[
                        "resolved_protocol_sha256"
                    ],
                    "protocol_identity_sha256": run[
                        "protocol_identity_sha256"
                    ],
                    "pipeline_sha256": run["pipeline_sha256"],
                }
                for label, run in zip(labels, runs)
            },
        },
        "methods": method_payloads,
    }
    rows = []
    for payload, metrics in zip(method_payloads, flattened_by_method):
        row: dict[str, object] = {"method": payload["method"], **metrics}
        row.update(
            {
                f"delta_to_no_loop_{name}": value
                for name, value in payload["delta_to_no_loop"].items()
            }
        )
        rows.append(row)
    output = Path(output_dir)
    _atomic_write_json(output / "loop_comparison.json", report)
    _atomic_write_csv(output / "loop_comparison.csv", rows, tuple(rows[0]))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare preserved LASER NeuralRGBD Depth point maps with a "
            "traditional-loop experiment."
        )
    )
    parser.add_argument("--no-loop-run", type=Path, required=True)
    parser.add_argument("--traditional-loop-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    compare_loop_run_directories(
        args.no_loop_run,
        args.traditional_loop_run,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
