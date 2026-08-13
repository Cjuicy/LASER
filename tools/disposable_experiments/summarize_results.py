#!/usr/bin/env python3
"""Create strict CSV and screenshot-style XLSX experiment summaries."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping, Sequence
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


def _load_utility():
    path = SCRIPT_DIR / "preflight_experiments.py"
    spec = importlib.util.spec_from_file_location("disposable_preflight_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UTILITY = _load_utility()

POINTCLOUD_FIELDS = (
    "dataset",
    "scene",
    "method",
    "segmentation_method",
    "split_mode",
    "reconstruction_mode",
    "sample_stride",
    "window_size",
    "overlap",
    "registration_confidence_keep_ratio",
    "prediction_key",
    "frame_count",
    "accuracy_mean_m",
    "accuracy_median_m",
    "completion_mean_m",
    "completion_median_m",
    "normal_consistency_mean",
    "normal_consistency_median",
    "chamfer_l1_m",
    "precision_1cm",
    "recall_1cm",
    "fscore_1cm",
    "precision_2cm",
    "recall_2cm",
    "fscore_2cm",
    "precision_5cm",
    "recall_5cm",
    "fscore_5cm",
)

ATE_FIELDS = (
    "dataset",
    "sequence",
    "method",
    "segmentation_method",
    "split_mode",
    "reconstruction_mode",
    "sample_stride",
    "window_size",
    "overlap",
    "registration_confidence_keep_ratio",
    "prediction_key",
    "frame_count",
    "ate_rmse_m",
    "rpe_translation_rmse_m",
    "rpe_rotation_rmse_deg",
    "matched_frame_count",
)

POINTCLOUD_NUMERIC_FIELDS = frozenset(POINTCLOUD_FIELDS[6:10] + POINTCLOUD_FIELDS[11:])
ATE_NUMERIC_FIELDS = frozenset(ATE_FIELDS[6:10] + ATE_FIELDS[11:])

POINTCLOUD_METHOD_LABELS = {
    "depth": "LASER-Depth\n(基线)",
    "geometry": "LASER-Geometry",
    "atomic-original": "LASER-Atomic\n(原始 / split off)",
    "atomic-split-assisted": "LASER-Atomic-split\n(有辅助判断)",
    "atomic-split-no-assisted": "LASER-Atomic-split\n(无辅助判断)",
}

ATE_METHOD_LABELS = {
    "depth-traditional": "LASER-Depth / Traditional\n(基线)",
    "depth-corrected": "LASER-Depth / Corrected",
    "geometry-corrected": "LASER-Geometry / Corrected",
    "atomic-original-corrected": "LASER-Atomic / Corrected\n(原始)",
    "atomic-split-assisted-corrected": "LASER-Atomic-split / Corrected\n(有辅助判断)",
    "atomic-split-no-assisted-corrected": "LASER-Atomic-split / Corrected\n(无辅助判断)",
}


def _load_map(path: str | Path) -> dict[str, list[int]]:
    return UTILITY._load_sequence_map(path)


def _safe_scene(scene: str) -> str:
    return scene.replace("/", "__")


def _pointcloud_row(payload: Mapping[str, object]) -> dict[str, object]:
    identity = payload["identity"]
    metrics = payload["metrics"]
    row = {
        "dataset": identity["dataset"],
        "scene": identity["scene"],
        "method": identity["method"],
        "segmentation_method": identity["segmentation_method"],
        "split_mode": identity["split_mode"],
        "reconstruction_mode": identity["reconstruction_mode"],
        "sample_stride": identity["sample_stride"],
        "window_size": identity["window_size"],
        "overlap": identity["overlap"],
        "registration_confidence_keep_ratio": identity[
            "registration_confidence_keep_ratio"
        ],
        "prediction_key": payload["prediction_key"],
        "frame_count": payload["frame_count"],
    }
    for name in UTILITY.POINTCLOUD_METRICS:
        row[name] = metrics[name]
    thresholds = {
        round(float(record["threshold_m"]), 3): record
        for record in metrics["thresholds"]
    }
    for threshold, suffix in ((0.01, "1cm"), (0.02, "2cm"), (0.05, "5cm")):
        record = thresholds[threshold]
        for name in ("precision", "recall", "fscore"):
            row[f"{name}_{suffix}"] = record[name]
    return row


def collect_pointcloud_rows(
    metric_root: str | Path,
    seven_map: Mapping[str, Sequence[int]],
    nrgbd_map: Mapping[str, Sequence[int]],
) -> list[dict[str, object]]:
    root = Path(metric_root) / "pointcloud"
    rows = []
    for dataset, sequence_map in (("7scenes", seven_map), ("nrgbd", nrgbd_map)):
        for scene in sequence_map:
            scene_prediction_key = None
            for method in UTILITY.POINTCLOUD_METHODS:
                identity = UTILITY.method_identity(
                    evaluation="pointcloud",
                    dataset=dataset,
                    scene=scene,
                    method=method,
                )
                path = root / dataset / _safe_scene(scene) / f"{method}.json"
                if not path.is_file():
                    raise ValueError(f"missing point-cloud result: {path}")
                payload = UTILITY.load_valid_result(path, identity)
                prediction_key = payload["prediction_key"]
                if scene_prediction_key is None:
                    scene_prediction_key = prediction_key
                elif prediction_key != scene_prediction_key:
                    raise ValueError(
                        f"point-cloud prediction key mismatch: {dataset}/{scene}"
                    )
                rows.append(_pointcloud_row(payload))
    return rows


def collect_ate_rows(metric_root: str | Path) -> list[dict[str, object]]:
    root = Path(metric_root) / "ate" / "kitti"
    rows = []
    for sequence in UTILITY.KITTI_SEQUENCES:
        sequence_prediction_key = None
        for method in UTILITY.ATE_METHODS:
            identity = UTILITY.method_identity(
                evaluation="ate",
                dataset="kitti",
                scene=sequence,
                method=method,
            )
            path = root / sequence / f"{method}.json"
            if not path.is_file():
                raise ValueError(f"missing ATE result: {path}")
            payload = UTILITY.load_valid_result(path, identity)
            prediction_key = payload["prediction_key"]
            if sequence_prediction_key is None:
                sequence_prediction_key = prediction_key
            elif prediction_key != sequence_prediction_key:
                raise ValueError(f"KITTI prediction key mismatch: {sequence}")
            metrics = payload["metrics"]
            rows.append(
                {
                    "dataset": "kitti",
                    "sequence": sequence,
                    "method": method,
                    "segmentation_method": identity["segmentation_method"],
                    "split_mode": identity["split_mode"],
                    "reconstruction_mode": identity["reconstruction_mode"],
                    "sample_stride": identity["sample_stride"],
                    "window_size": identity["window_size"],
                    "overlap": identity["overlap"],
                    "registration_confidence_keep_ratio": identity[
                        "registration_confidence_keep_ratio"
                    ],
                    "prediction_key": prediction_key,
                    "frame_count": payload["frame_count"],
                    **{name: metrics[name] for name in UTILITY.ATE_METRICS},
                }
            )
    return rows


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


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]):
    buffer = io.StringIO(newline="")
    buffer.write("\ufeff")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _pointcloud_cell(row: Mapping[str, object]) -> str:
    return "\n".join(
        (
            f"Acc mean : {row['accuracy_mean_m']:.6f} m",
            f"Acc med  : {row['accuracy_median_m']:.6f} m",
            f"Comp mean: {row['completion_mean_m']:.6f} m",
            f"Comp med : {row['completion_median_m']:.6f} m",
            f"NC mean  : {row['normal_consistency_mean']:.6f}",
            f"NC med   : {row['normal_consistency_median']:.6f}",
            f"Chamfer  : {row['chamfer_l1_m']:.6f} m",
            "F@1/2/5 : "
            f"{row['fscore_1cm']:.6f} / {row['fscore_2cm']:.6f} / "
            f"{row['fscore_5cm']:.6f}",
        )
    )


def _ate_cell(row: Mapping[str, object]) -> str:
    return "\n".join(
        (
            f"KITTI sequence : {row['sequence']}",
            f"ATE RMSE : {row['ate_rmse_m']:.6f} m",
            f"RPE translation : {row['rpe_translation_rmse_m']:.6f} m",
            f"RPE rotation : {row['rpe_rotation_rmse_deg']:.6f} deg",
        )
    )


_STRING_CELL = re.compile(
    r'<(?P<prefix>[A-Za-z][A-Za-z0-9]*:)?c'
    r'(?P<before>[^>]*)t="str"(?P<after>[^>]*)>'
    r'<(?P=prefix)v>(?P<token>__[A-Z0-9_]+__)</(?P=prefix)v>'
    r'</(?P=prefix)c>'
)


def _populate_template(
    template: Path,
    output: Path,
    replacements: Mapping[str, object],
    numeric_tokens: frozenset[str],
) -> None:
    if not template.is_file():
        raise FileNotFoundError(f"workbook template does not exist: {template}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output.with_name(f".{output.name}.tmp")
    found = set()
    try:
        with zipfile.ZipFile(template, "r") as source, zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename.startswith("xl/worksheets/") and info.filename.endswith(
                    ".xml"
                ):
                    text = data.decode("utf-8")

                    def replace_cell(match):
                        token = match.group("token")
                        if token not in replacements:
                            return match.group(0)
                        found.add(token)
                        prefix = match.group("prefix") or ""
                        attributes = f"{match.group('before')}{match.group('after')}"
                        if token in numeric_tokens:
                            value = replacements[token]
                            if (
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or not math.isfinite(float(value))
                            ):
                                raise ValueError(f"numeric workbook token is invalid: {token}")
                            return (
                                f"<{prefix}c{attributes}><{prefix}v>{value}"
                                f"</{prefix}v></{prefix}c>"
                            )
                        return (
                            f'<{prefix}c{match.group("before")}t="inlineStr"'
                            f'{match.group("after")}><{prefix}is>'
                            f'<{prefix}t xml:space="preserve">'
                            f"{escape(str(replacements[token]))}"
                            f"</{prefix}t></{prefix}is></{prefix}c>"
                        )

                    text = _STRING_CELL.sub(replace_cell, text)
                    data = text.encode("utf-8")
                target.writestr(info, data)
        missing = sorted(set(replacements) - found)
        if missing:
            raise ValueError(
                "workbook template is missing replacement tokens: "
                + ", ".join(missing[:5])
            )
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _raw_replacements(
    prefix: str,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    numeric_fields: frozenset[str],
):
    replacements = {}
    numeric_tokens = set()
    for row_index, row in enumerate(rows):
        for column_index, field in enumerate(fields):
            token = f"__{prefix}_{row_index:03d}_{column_index:02d}__"
            replacements[token] = row[field]
            if field in numeric_fields:
                numeric_tokens.add(token)
    return replacements, numeric_tokens


def _pointcloud_workbook_replacements(
    rows: Sequence[Mapping[str, object]],
    seven_map: Mapping[str, Sequence[int]],
    nrgbd_map: Mapping[str, Sequence[int]],
):
    replacements, numeric_tokens = _raw_replacements(
        "PCR", rows, POINTCLOUD_FIELDS, POINTCLOUD_NUMERIC_FIELDS
    )
    lookup = {(row["dataset"], row["scene"], row["method"]): row for row in rows}
    token_index = 0
    for dataset, mapping in (("7scenes", seven_map), ("nrgbd", nrgbd_map)):
        for method in UTILITY.POINTCLOUD_METHODS:
            for scene in mapping:
                replacements[f"__PCP_{token_index:03d}__"] = _pointcloud_cell(
                    lookup[(dataset, scene, method)]
                )
                token_index += 1
    return replacements, frozenset(numeric_tokens)


def _ate_workbook_replacements(rows: Sequence[Mapping[str, object]]):
    replacements, numeric_tokens = _raw_replacements(
        "ATER", rows, ATE_FIELDS, ATE_NUMERIC_FIELDS
    )
    lookup = {(row["sequence"], row["method"]): row for row in rows}
    token_index = 0
    for method in UTILITY.ATE_METHODS:
        for sequence in UTILITY.KITTI_SEQUENCES:
            replacements[f"__ATEP_{token_index:03d}__"] = _ate_cell(
                lookup[(sequence, method)]
            )
            token_index += 1
    return replacements, frozenset(numeric_tokens)


def build_parser() -> argparse.ArgumentParser:
    repository = SCRIPT_DIR.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-root",
        default=str(repository / "outputs/disposable_experiments/metrics"),
    )
    parser.add_argument(
        "--seven-map",
        default=str(repository / "datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json"),
    )
    parser.add_argument(
        "--nrgbd-map",
        default=str(repository / "datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repository / "outputs/disposable_experiments/summary"),
    )
    parser.add_argument(
        "--pointcloud-template",
        default=str(TEMPLATE_DIR / "pointcloud_summary_template.xlsx"),
    )
    parser.add_argument(
        "--ate-template",
        default=str(TEMPLATE_DIR / "kitti_ate_summary_template.xlsx"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    seven_map = _load_map(arguments.seven_map)
    nrgbd_map = _load_map(arguments.nrgbd_map)
    pointcloud_rows = collect_pointcloud_rows(
        arguments.metric_root, seven_map, nrgbd_map
    )
    ate_rows = collect_ate_rows(arguments.metric_root)
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "pointcloud_long.csv", POINTCLOUD_FIELDS, pointcloud_rows)
    _write_csv(output / "kitti_ate_long.csv", ATE_FIELDS, ate_rows)
    replacements, numeric_tokens = _pointcloud_workbook_replacements(
        pointcloud_rows, seven_map, nrgbd_map
    )
    _populate_template(
        Path(arguments.pointcloud_template),
        output / "pointcloud_summary.xlsx",
        replacements,
        numeric_tokens,
    )
    replacements, numeric_tokens = _ate_workbook_replacements(ate_rows)
    _populate_template(
        Path(arguments.ate_template),
        output / "kitti_ate_summary.xlsx",
        replacements,
        numeric_tokens,
    )
    print(output / "pointcloud_long.csv")
    print(output / "pointcloud_summary.xlsx")
    print(output / "kitti_ate_long.csv")
    print(output / "kitti_ate_summary.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
