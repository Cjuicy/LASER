from importlib import import_module


_EXPORTS = {
    "EvaluationKind": ("experiments.config", "EvaluationKind"),
    "ExperimentConfig": ("experiments.config", "ExperimentConfig"),
    "ExperimentDatasetConfig": ("experiments.config", "ExperimentDatasetConfig"),
    "load_experiment_config": ("experiments.config", "load_experiment_config"),
    "ArtifactRepository": ("experiments.matrix", "ArtifactRepository"),
    "MatrixEntry": ("experiments.matrix", "MatrixEntry"),
    "build_matrix": ("experiments.matrix", "build_matrix"),
    "reconstruction_identity": ("experiments.matrix", "reconstruction_identity"),
    "ExperimentRunRecord": ("experiments.runner", "ExperimentRunRecord"),
    "matrix_cache_mode": ("experiments.runner", "matrix_cache_mode"),
    "run_matrix": ("experiments.runner", "run_matrix"),
}

__all__ = [
    "EvaluationKind",
    "ExperimentRunRecord",
    "ArtifactRepository",
    "ExperimentConfig",
    "ExperimentDatasetConfig",
    "MatrixEntry",
    "build_matrix",
    "load_experiment_config",
    "reconstruction_identity",
    "matrix_cache_mode",
    "run_matrix",
]


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
