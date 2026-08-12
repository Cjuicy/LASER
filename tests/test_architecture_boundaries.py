from __future__ import annotations

import ast
import inspect
from pathlib import Path

from evaluation.pointcloud.evaluator import evaluate_point_maps
from evaluation.trajectory.evaluator import evaluate_trajectory


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "evaluation": (
        "reconstruction",
        "inference_engine.segmentation",
        "inference_engine.anchor_propagation",
        "loop_closure",
        "pipeline.runner",
    ),
    "reconstruction.modes.no_loop": ("loop_closure",),
}


def _module_for(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


def collect_forbidden_imports():
    violations = []
    for source, forbidden in FORBIDDEN.items():
        source_path = ROOT / Path(*source.split("."))
        paths = (
            sorted(source_path.rglob("*.py"))
            if source_path.is_dir()
            else [source_path.with_suffix(".py")]
        )
        for path in paths:
            for line, imported in _imports(path):
                if any(
                    imported == prefix or imported.startswith(prefix + ".")
                    for prefix in forbidden
                ):
                    violations.append(
                        f"{_module_for(path)}:{line} imports {imported}"
                    )
    return violations


def test_forbidden_import_edges_are_absent():
    assert collect_forbidden_imports() == []


def test_evaluator_signatures_accept_views_not_pipeline_config():
    assert tuple(inspect.signature(evaluate_trajectory).parameters) == (
        "estimate",
        "ground_truth",
        "config",
    )
    assert tuple(inspect.signature(evaluate_point_maps).parameters)[:3] == (
        "estimate",
        "ground_truth",
        "config",
    )
