import numpy as np
import pytest
import warnings

from mv_recon.eval_utils import accuracy, completion
from mv_recon.geometry_metrics import (
    BackendResult,
    combine_normal_consistency,
    compute_directional_metrics,
    evaluate_point_maps,
)
from mv_recon.protocol import GeometryProtocol


def _geometry(crop_size: int = 224) -> GeometryProtocol:
    return GeometryProtocol(
        center_crop_size=crop_size,
        alignment="umeyama_sim3_then_icp",
        icp_type="point_to_point",
        icp_threshold_m=0.1,
        normal_estimation="open3d_default",
        fscore_thresholds_m=(0.01, 0.02, 0.05),
    )


def _grid_points(frames: int, height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.5, 1.5, width),
        indexing="ij",
    )
    z = 0.2 * x + 0.1 * y + 1.0
    grid = np.stack((x, y, z), axis=-1)
    return np.repeat(grid[None], frames, axis=0)


class RecordingBackend:
    def __init__(self):
        self.thresholds = []

    def refine_and_estimate_normals(
        self,
        predicted,
        ground_truth,
        threshold_m,
    ):
        self.thresholds.append(threshold_m)
        return BackendResult(
            predicted_points=np.asarray(predicted),
            ground_truth_points=np.asarray(ground_truth),
            predicted_normals=np.tile(
                np.array([1.0, 0.0, 0.0]),
                (len(predicted), 1),
            ),
            ground_truth_normals=np.tile(
                np.array([1.0, 0.0, 0.0]),
                (len(ground_truth), 1),
            ),
            transformation=np.eye(4),
            fitness=1.0,
            inlier_rmse=0.0,
        )


def test_directional_names_match_official_accuracy_and_completion():
    predicted = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    ground_truth = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    predicted_normals = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    gt_normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    result = compute_directional_metrics(
        predicted,
        ground_truth,
        predicted_normals,
        gt_normals,
        (0.01, 0.02, 0.05),
    )

    assert result.primary.accuracy_mean_m == pytest.approx(1.0)
    assert result.primary.accuracy_median_m == pytest.approx(1.0)
    assert result.primary.completion_mean_m == pytest.approx(0.0)
    assert result.primary.completion_median_m == pytest.approx(0.0)
    assert result.primary.normal_consistency_mean == pytest.approx(0.75)
    assert result.primary.normal_consistency_median == pytest.approx(0.75)
    assert result.chamfer_l1_m == pytest.approx(0.5)


def test_nc_median_averages_directional_medians_not_concatenated_samples():
    mean, median = combine_normal_consistency(
        nc1=np.array([0.0, 0.0, 1.0]),
        nc2=np.array([0.5]),
    )

    assert mean == pytest.approx(0.4166666667)
    assert median == pytest.approx(0.25)


def test_directional_metrics_match_existing_official_helpers():
    predicted = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    ground_truth = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    predicted_normals = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    gt_normals = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    current = compute_directional_metrics(
        predicted,
        ground_truth,
        predicted_normals,
        gt_normals,
        (0.05,),
    )
    acc, acc_med, nc1, nc1_med = accuracy(
        ground_truth,
        predicted,
        gt_normals,
        predicted_normals,
    )
    comp, comp_med, nc2, nc2_med = completion(
        ground_truth,
        predicted,
        gt_normals,
        predicted_normals,
    )

    assert current.primary.accuracy_mean_m == pytest.approx(acc)
    assert current.primary.accuracy_median_m == pytest.approx(acc_med)
    assert current.primary.completion_mean_m == pytest.approx(comp)
    assert current.primary.completion_median_m == pytest.approx(comp_med)
    assert current.directional_normals.nc1_mean == pytest.approx(nc1)
    assert current.directional_normals.nc1_median == pytest.approx(nc1_med)
    assert current.directional_normals.nc2_mean == pytest.approx(nc2)
    assert current.directional_normals.nc2_median == pytest.approx(nc2_med)


def test_evaluate_point_maps_uses_crop_gt_mask_umeyama_and_fixed_icp():
    ground_truth = _grid_points(frames=1, height=226, width=228)
    predicted = ground_truth * 2.0 + np.array([3.0, -1.0, 0.5])
    valid_mask = np.ones(ground_truth.shape[:-1], dtype=bool)
    valid_mask[:, 0, :] = False
    backend = RecordingBackend()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = evaluate_point_maps(
            predicted,
            ground_truth,
            valid_mask,
            _geometry(),
            backend=backend,
        )

    assert result.diagnostics.predicted_point_count == 224 * 224
    assert result.diagnostics.ground_truth_point_count == 224 * 224
    assert result.diagnostics.umeyama_scale == pytest.approx(0.5)
    assert backend.thresholds == [0.1]
    assert result.primary.accuracy_mean_m == pytest.approx(0.0, abs=1e-8)
    assert result.primary.completion_mean_m == pytest.approx(0.0, abs=1e-8)


def test_evaluate_point_maps_rejects_nonfinite_selected_points():
    ground_truth = _grid_points(frames=1, height=4, width=4)
    predicted = ground_truth.copy()
    predicted[0, 1, 1, 0] = np.nan
    valid_mask = np.ones(ground_truth.shape[:-1], dtype=bool)

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_point_maps(
            predicted,
            ground_truth,
            valid_mask,
            _geometry(crop_size=4),
            backend=RecordingBackend(),
        )


def test_evaluate_point_maps_rejects_degenerate_umeyama_points():
    ground_truth = _grid_points(frames=1, height=3, width=3)
    predicted = np.ones_like(ground_truth)
    valid_mask = np.ones(ground_truth.shape[:-1], dtype=bool)

    with pytest.raises(ValueError, match="degenerate Umeyama"):
        evaluate_point_maps(
            predicted,
            ground_truth,
            valid_mask,
            _geometry(crop_size=3),
            backend=RecordingBackend(),
        )


def test_evaluate_point_maps_requires_three_valid_correspondences():
    ground_truth = _grid_points(frames=1, height=3, width=3)
    valid_mask = np.zeros(ground_truth.shape[:-1], dtype=bool)
    valid_mask.reshape(-1)[:2] = True

    with pytest.raises(ValueError, match="at least three"):
        evaluate_point_maps(
            ground_truth.copy(),
            ground_truth,
            valid_mask,
            _geometry(crop_size=3),
            backend=RecordingBackend(),
        )


def test_evaluate_point_maps_rejects_crop_larger_than_input():
    ground_truth = _grid_points(frames=1, height=3, width=4)

    with pytest.raises(ValueError, match="crop size 5"):
        evaluate_point_maps(
            ground_truth.copy(),
            ground_truth,
            np.ones(ground_truth.shape[:-1], dtype=bool),
            _geometry(crop_size=5),
            backend=RecordingBackend(),
        )
