from scripts import verify_pipeline_matrix as matrix_module
from scripts.verify_pipeline_matrix import build_matrix


def test_matrix_contains_exactly_ten_unique_configurations():
    matrix = build_matrix()
    assert len(matrix) == 10
    assert len({entry.name for entry in matrix}) == 10
    assert {
        (
            entry.segmentation_method,
            entry.atomic_split_mode,
            entry.loop_method,
        )
        for entry in matrix
    } == {
        ("depth", None, "traditional"),
        ("depth", None, "corrected"),
        ("geometry", None, "traditional"),
        ("geometry", None, "corrected"),
        ("atomic", "none", "traditional"),
        ("atomic", "none", "corrected"),
        ("atomic", "conservative", "traditional"),
        ("atomic", "conservative", "corrected"),
        ("atomic", "normal_only", "traditional"),
        ("atomic", "normal_only", "corrected"),
    }


def test_entries_apply_unique_output_suffixes():
    for entry in build_matrix():
        overrides = entry.overrides()
        assert f"output.scene_name=matrix_{entry.name}" in overrides
        assert f"output.cache_dir=matrix_runs/cache/{entry.name}" in overrides
        assert f"output.result_dir=matrix_runs/results/{entry.name}" in overrides
        if entry.segmentation_method != "atomic":
            assert not any(
                override.startswith(
                    "segmentation.atomic.split_mode="
                )
                for override in overrides
            )


def test_dry_run_validates_all_configs_without_loading_models(monkeypatch):
    calls = []
    monkeypatch.setattr(
        matrix_module,
        "run_from_config",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert matrix_module.main(
        [
            "--config",
            "configs/pipeline/test.yaml",
            "--dry-run",
        ]
    ) == 0
    assert calls == []
