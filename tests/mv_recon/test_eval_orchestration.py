import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch

from mv_recon.eval import (
    EvaluationDependencies,
    EvaluationFailed,
    InferenceOutput,
    run_evaluation,
)
from mv_recon.geometry_metrics import (
    DirectionalNormalMetrics,
    GeometryDiagnostics,
    GeometryEvaluation,
    PrimaryMetrics,
    ThresholdMetrics,
)


ROOT = Path(__file__).resolve().parents[2]


def _config(
    tmp_path: Path,
    *,
    profile: str = "mv_recon_laser_paper",
    max_sequences: int | None = 1,
    preflight_only: bool = False,
    resume: bool = False,
):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.2",
    ):
        return compose(
            config_name="eval_mv_recon_dense",
            overrides=[
                f"evaluation={profile}",
                "device=cpu",
                "protocol.max_sequences="
                + ("null" if max_sequences is None else str(max_sequences)),
                f"protocol.preflight_only={str(preflight_only).lower()}",
                f"protocol.resume={str(resume).lower()}",
                f"output_dir={tmp_path / 'results'}",
                f"pi3.checkpoint={checkpoint}",
            ],
        )


@dataclass
class State:
    model_constructions: int = 0
    inference_sequences: list[str] = field(default_factory=list)
    geometry_calls: int = 0
    empty_cache_calls: int = 0
    fail_geometry_call: int | None = None
    interrupt_geometry_call: int | None = None
    ground_truth_offset: float = 0.0


class FakeDataset:
    def __init__(self, dataset_name: str, root: Path, state: State):
        self.dataset_name = dataset_name
        self.root = root
        self.state = state
        self.sequence_list = (
            ["chess/seq-03"]
            if dataset_name == "7scenes-dense"
            else [
                "breakfast_room",
                "complete_kitchen",
                "green_room",
                "grey_white_room",
                "kitchen",
                "morning_apartment",
                "staircase",
                "thin_geometry",
                "whiteroom",
            ]
        )

    def get_seq_framenum(self, sequence_name):
        assert sequence_name in self.sequence_list
        return 2000

    def get_data(self, sequence_name, ids):
        assert sequence_name in self.sequence_list
        sequence_dir = (
            self.root
            / self.dataset_name
            / sequence_name.replace("/", "-")
        )
        sequence_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for frame_id in ids:
            path = sequence_dir / f"frame-{frame_id:06d}.png"
            path.write_bytes(f"{self.dataset_name}:{frame_id}".encode("utf-8"))
            image_paths.append(str(path))
        frame_count = len(ids)
        pointclouds = np.zeros((frame_count, 2, 2, 3), dtype=np.float64)
        pointclouds[..., 2] = 1.0 + self.state.ground_truth_offset
        return {
            "image_paths": image_paths,
            "images": torch.zeros((frame_count, 3, 2, 2)),
            "pointclouds": pointclouds,
            "valid_mask": np.ones((frame_count, 2, 2), dtype=bool),
        }


def _geometry_result() -> GeometryEvaluation:
    primary = PrimaryMetrics(
        accuracy_mean_m=0.01,
        accuracy_median_m=0.005,
        completion_mean_m=0.02,
        completion_median_m=0.006,
        normal_consistency_mean=0.7,
        normal_consistency_median=0.8,
    )
    return GeometryEvaluation(
        primary=primary,
        diagnostics=GeometryDiagnostics(
            umeyama_scale=1.0,
            icp_transformation=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            icp_fitness=1.0,
            icp_inlier_rmse=0.0,
            predicted_point_count=4,
            ground_truth_point_count=4,
            directional_normals=DirectionalNormalMetrics(
                nc1_mean=0.7,
                nc1_median=0.8,
                nc2_mean=0.7,
                nc2_median=0.8,
            ),
            chamfer_l1_m=0.015,
            thresholds=(
                ThresholdMetrics(0.01, 0.5, 0.5, 0.5),
                ThresholdMetrics(0.02, 0.8, 0.8, 0.8),
                ThresholdMetrics(0.05, 1.0, 1.0, 1.0),
            ),
        ),
    )


def _dependencies(tmp_path: Path, state: State) -> EvaluationDependencies:
    def instantiate_dataset(config):
        target = str(config._target_)
        dataset_name = (
            "7scenes-dense" if target.endswith("SevenScenes") else "NRGBD-dense"
        )
        return FakeDataset(dataset_name, tmp_path / "fake-data", state)

    def model_factory(config):
        state.model_constructions += 1
        return object()

    def infer_point_maps(paths, model, hydra_cfg, data_size):
        state.inference_sequences.append(Path(paths[0]).parent.name)
        frame_count = len(paths)
        points = np.zeros((frame_count, 2, 2, 3), dtype=np.float64)
        points[..., 2] = 1.0
        return InferenceOutput(
            points=points,
            confidence=np.full((frame_count, 2, 2), np.nan),
            ordinary_prediction_key=(
                "a" if "7scenes" in paths[0] else "b"
            )
            * 64,
            cache_diagnostics={"ordinary_hits": 0, "ordinary_misses": 1},
        )

    def evaluate_geometry(predicted, ground_truth, mask, geometry, backend):
        state.geometry_calls += 1
        if state.interrupt_geometry_call == state.geometry_calls:
            raise KeyboardInterrupt("injected user interruption")
        if state.fail_geometry_call == state.geometry_calls:
            raise RuntimeError("injected geometry failure")
        assert predicted.shape == ground_truth.shape
        assert mask.shape == ground_truth.shape[:-1]
        return _geometry_result()

    def empty_cache():
        state.empty_cache_calls += 1

    return EvaluationDependencies(
        instantiate_dataset=instantiate_dataset,
        model_factory=model_factory,
        infer_point_maps=infer_point_maps,
        evaluate_geometry=evaluate_geometry,
        geometry_backend_factory=lambda: object(),
        checkpoint_digest=lambda path: "c" * 64,
        git_commit=lambda: "test-commit",
        runtime_metadata=lambda: {
            "python": "test",
            "open3d": "test",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": "test",
            "cuda_available": False,
            "cuda_capability": None,
            "gpu_name": None,
        },
        empty_cuda_cache=empty_cache,
    )


def test_preflight_only_never_constructs_streaming_model(tmp_path):
    state = State()

    result = run_evaluation(
        _config(tmp_path, preflight_only=True),
        dependencies=_dependencies(tmp_path, state),
        repository_root=ROOT,
    )

    assert result.state == "preflight"
    assert state.model_constructions == 0
    assert state.inference_sequences == []
    assert (tmp_path / "results" / "resolved_protocol.yaml").is_file()
    assert (tmp_path / "results" / "resolved_pipeline.yaml").is_file()
    assert (tmp_path / "results" / "protocol_manifest.json").is_file()


def test_two_dataset_smoke_builds_one_model_and_returns_subset(tmp_path):
    state = State()

    result = run_evaluation(
        _config(tmp_path),
        dependencies=_dependencies(tmp_path, state),
        repository_root=ROOT,
    )

    assert result.state == "subset"
    assert state.model_constructions == 1
    assert state.inference_sequences == ["chess-seq-03", "breakfast_room"]
    assert state.geometry_calls == 2
    assert state.empty_cache_calls == 2
    assert [(item.dataset, item.sequence) for item in result.sequences] == [
        ("7scenes-dense", "chess/seq-03"),
        ("NRGBD-dense", "breakfast_room"),
    ]
    assert result.datasets[0].primary.normal_consistency_median == pytest.approx(
        0.8
    )

    output = tmp_path / "results"
    assert {
        "resolved_protocol.yaml",
        "resolved_pipeline.yaml",
        "protocol_manifest.json",
        "results.json",
        "summary.csv",
        "sequences.csv",
        "failures.jsonl",
    } <= {path.name for path in output.iterdir()}
    canonical = json.loads(
        (output / "results.json").read_text(encoding="utf-8")
    )
    assert set(canonical["datasets"][0]["primary"]) == {
        "accuracy_mean_m",
        "accuracy_median_m",
        "completion_mean_m",
        "completion_median_m",
        "normal_consistency_mean",
        "normal_consistency_median",
    }
    assert canonical["sequences"][0]["ordinary_prediction_key"] == "a" * 64
    manifest = json.loads(
        (output / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_state"] == "subset"
    assert manifest["attempted_sequences"] == 2
    assert manifest["successful_sequences"] == 2
    assert len(manifest["sequence_cache"]) == 2
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    with (output / "sequences.csv").open(newline="", encoding="utf-8") as stream:
        sequence_rows = list(csv.DictReader(stream))
    assert [row["dataset"] for row in summary_rows] == [
        "7scenes-dense",
        "NRGBD-dense",
    ]
    assert float(summary_rows[0]["accuracy_mean_m"]) == pytest.approx(
        canonical["datasets"][0]["primary"]["accuracy_mean_m"]
    )
    assert len(sequence_rows) == 2
    assert (output / "failures.jsonl").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("profile", "method"),
    (
        ("mv_recon_laser_nrgbd_depth", "depth"),
        ("mv_recon_laser_nrgbd_geometry", "geometry"),
        ("mv_recon_laser_nrgbd_atomic", "atomic"),
    ),
)
def test_full_nrgbd_comparison_is_subset_with_method_manifest(
    tmp_path, profile, method
):
    result = run_evaluation(
        _config(tmp_path, profile=profile, max_sequences=None),
        dependencies=_dependencies(tmp_path, State()),
        repository_root=ROOT,
    )

    assert result.state == "subset"
    assert result.datasets[0].status == "subset"
    assert [item.sequence for item in result.sequences] == [
        "breakfast_room",
        "complete_kitchen",
        "green_room",
        "grey_white_room",
        "kitchen",
        "morning_apartment",
        "staircase",
        "thin_geometry",
        "whiteroom",
    ]
    manifest = json.loads(
        (tmp_path / "results/protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["evaluation_mode"] == "comparison"
    assert manifest["segmentation_method"] == method


def test_sequence_failure_preserves_prior_result_and_raises_nonzero_error(
    tmp_path,
):
    state = State(fail_geometry_call=2)

    with pytest.raises(EvaluationFailed, match="NRGBD-dense/breakfast_room"):
        run_evaluation(
            _config(tmp_path),
            dependencies=_dependencies(tmp_path, state),
            repository_root=ROOT,
        )

    payload = json.loads(
        (tmp_path / "results" / "results.json").read_text(encoding="utf-8")
    )
    assert payload["state"] == "incomplete"
    assert [item["sequence"] for item in payload["sequences"]] == [
        "chess/seq-03"
    ]
    assert payload["failures"][0]["sequence"] == "breakfast_room"
    assert state.empty_cache_calls == 2
    manifest = json.loads(
        (tmp_path / "results" / "protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["run_state"] == "incomplete"


def test_keyboard_interrupt_preserves_prior_result_and_incomplete_state(
    tmp_path,
):
    state = State(interrupt_geometry_call=2)

    with pytest.raises(KeyboardInterrupt, match="user interruption"):
        run_evaluation(
            _config(tmp_path),
            dependencies=_dependencies(tmp_path, state),
            repository_root=ROOT,
        )

    payload = json.loads(
        (tmp_path / "results" / "results.json").read_text(encoding="utf-8")
    )
    assert payload["state"] == "incomplete"
    assert [item["sequence"] for item in payload["sequences"]] == [
        "chess/seq-03"
    ]
    assert payload["failures"][0]["category"] == "KeyboardInterrupt"
    manifest = json.loads(
        (tmp_path / "results" / "protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["run_state"] == "incomplete"
    assert manifest["attempted_sequences"] == 2
    assert state.empty_cache_calls == 2


def test_exact_resume_skips_inference_and_geometry(tmp_path):
    first_state = State()
    first = run_evaluation(
        _config(tmp_path),
        dependencies=_dependencies(tmp_path, first_state),
        repository_root=ROOT,
    )
    assert first.state == "subset"

    resumed_state = State()
    resumed = run_evaluation(
        _config(tmp_path, resume=True),
        dependencies=_dependencies(tmp_path, resumed_state),
        repository_root=ROOT,
    )

    assert resumed.state == "subset"
    assert resumed_state.model_constructions == 1
    assert resumed_state.inference_sequences == []
    assert resumed_state.geometry_calls == 0


def test_changed_ground_truth_invalidates_resumed_sequence_results(tmp_path):
    first_state = State()
    run_evaluation(
        _config(tmp_path),
        dependencies=_dependencies(tmp_path, first_state),
        repository_root=ROOT,
    )

    changed_state = State(ground_truth_offset=0.25)
    result = run_evaluation(
        _config(tmp_path, resume=True),
        dependencies=_dependencies(tmp_path, changed_state),
        repository_root=ROOT,
    )

    assert result.state == "subset"
    assert changed_state.inference_sequences == [
        "chess-seq-03",
        "breakfast_room",
    ]
    assert changed_state.geometry_calls == 2
    assert all(
        "ground_truth_sha256" in item
        for item in json.loads(
            (tmp_path / "results" / "results.json").read_text(
                encoding="utf-8"
            )
        )["sequences"]
    )
