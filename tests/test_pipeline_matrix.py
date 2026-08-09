from pipeline.config import ModelName, PredictionCacheMode
from scripts import verify_pipeline_matrix as matrix_module
from scripts.verify_pipeline_matrix import (
    build_matrix,
    effective_matrix_cache_mode,
)


def test_pi3_matrix_contains_exactly_ten_unique_configurations():
    matrix = build_matrix(ModelName.PI3)
    assert len(matrix) == 10
    assert len({entry.name for entry in matrix}) == 10
    assert all(entry.name.startswith("pi3_") for entry in matrix)
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


def test_entries_keep_prediction_root_shared_and_outputs_unique():
    for entry in build_matrix(ModelName.PI3):
        overrides = entry.overrides()
        assert f"output.scene_name=matrix_{entry.name}" in overrides
        assert f"output.cache_dir=matrix_runs/cache/{entry.name}" in overrides
        assert f"output.result_dir=matrix_runs/results/{entry.name}" in overrides
        assert not any(
            override.startswith("prediction_cache.root=")
            for override in overrides
        )
        if entry.segmentation_method != "atomic":
            assert not any(
                override.startswith(
                    "segmentation.atomic.split_mode="
                )
                for override in overrides
            )


def test_matrix_cache_mode_mapping():
    for requested in (
        PredictionCacheMode.AUTO,
        PredictionCacheMode.READONLY,
        PredictionCacheMode.OFF,
    ):
        assert [
            effective_matrix_cache_mode(requested, index)
            for index in range(10)
        ] == [requested] * 10

    assert [
        effective_matrix_cache_mode(
            PredictionCacheMode.REFRESH,
            index,
        )
        for index in range(10)
    ] == [
        PredictionCacheMode.REFRESH,
        *([PredictionCacheMode.READONLY] * 9),
    ]


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
