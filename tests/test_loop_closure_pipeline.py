from pathlib import Path
import importlib.util
import sys
import types

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compose_sim3(first, second):
    scale_1, rotation_1, translation_1 = first
    scale_2, rotation_2, translation_2 = second
    return (
        scale_1 * scale_2,
        rotation_1 @ rotation_2,
        scale_1 * rotation_1 @ translation_2 + translation_1,
    )


def inverse_sim3(scale, rotation=None, translation=None):
    if rotation is None and translation is None:
        scale, rotation, translation = scale
    inverse_scale = 1.0 / scale
    inverse_rotation = rotation.T
    inverse_translation = (
        -inverse_scale * inverse_rotation @ translation
    )
    return inverse_scale, inverse_rotation, inverse_translation


def compute_sim3_ab(sim3_a, sim3_b):
    scale_a, rotation_a, translation_a = sim3_a
    scale_b, rotation_b, translation_b = sim3_b
    scale_ab = scale_b / scale_a
    rotation_ab = rotation_b @ rotation_a.T
    translation_ab = (
        translation_b
        - scale_ab * rotation_ab @ translation_a
    )
    return scale_ab, rotation_ab, translation_ab


def make_sim3(scale):
    return scale, torch.eye(3), torch.zeros(3)


def assert_sim3_close(actual, expected):
    actual_scale, actual_rotation, actual_translation = actual
    expected_scale, expected_rotation, expected_translation = expected
    assert torch.as_tensor(actual_scale).item() == pytest.approx(
        torch.as_tensor(expected_scale).item()
    )
    torch.testing.assert_close(actual_rotation, expected_rotation)
    torch.testing.assert_close(actual_translation, expected_translation)


def make_cache(value, sim3_abs, sim3_edge=None):
    cache = {
        "local_points": torch.full((2, 1, 1, 3), value),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
        "conf": torch.ones((2, 1, 1)),
        "sim3_abs": sim3_abs,
    }
    if sim3_edge is not None:
        cache["sim3_edge"] = sim3_edge
    return cache


def engine_config():
    return {
        "Model": {"loop_chunk_size": 2},
        "Loop": {
            "SIM3_Optimizer": {
                "lang_version": "python",
                "max_iterations": 3,
                "lambda_init": "1e-6",
            },
        },
    }


def load_loop_engine_module(monkeypatch):
    detector_instances = []
    optimizer_instances = []

    fake_loop_model = types.ModuleType("loop_closure.loop_model")

    class RecordingDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loop_list = []
            self.run_calls = 0
            self.image_paths = kwargs.get("image_paths")
            detector_instances.append(self)

        def run(self):
            self.run_calls += 1

        def get_loop_list(self):
            return self.loop_list

    fake_loop_model.LoopDetector = RecordingDetector
    monkeypatch.setitem(sys.modules, "loop_closure.loop_model", fake_loop_model)

    fake_optimizer_module = types.ModuleType("loop_closure.utils.sim3loop")

    class RecordingOptimizer:
        def __init__(self, config):
            self.config = config
            self.calls = []
            self.optimized_edges = None
            optimizer_instances.append(self)

        def optimize(self, edges, constraints):
            self.calls.append((edges, constraints))
            if self.optimized_edges is None:
                return edges
            return self.optimized_edges

    fake_optimizer_module.Sim3LoopOptimizer = RecordingOptimizer
    monkeypatch.setitem(
        sys.modules,
        "loop_closure.utils.sim3loop",
        fake_optimizer_module,
    )

    fake_sim3utils = types.ModuleType("loop_closure.utils.sim3utils")
    fake_sim3utils.compute_sim3_ab = compute_sim3_ab
    fake_sim3utils.process_loop_list = lambda *args, **kwargs: []
    monkeypatch.setitem(
        sys.modules,
        "loop_closure.utils.sim3utils",
        fake_sim3utils,
    )

    fake_load = types.ModuleType("utils.load_fn")
    fake_load.load_and_preprocess_images = lambda paths: torch.ones(
        (len(paths), 3, 1, 1)
    )
    monkeypatch.setitem(sys.modules, "utils.load_fn", fake_load)

    fake_pi3 = types.ModuleType("pi3.models.pi3")
    fake_pi3.Pi3 = object
    monkeypatch.setitem(sys.modules, "pi3.models.pi3", fake_pi3)

    fake_engine = types.ModuleType("inference_engine")
    fake_engine.__path__ = []
    fake_engine.StreamingWindowEngine = object
    fake_engine.StreamingWindowEngineLC = object
    monkeypatch.setitem(sys.modules, "inference_engine", fake_engine)

    fake_inference = types.ModuleType("inference_engine.inference_utils")
    fake_inference.register_adjacent_windows = lambda *args: make_sim3(1.0)
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.inference_utils",
        fake_inference,
    )

    fake_engine_utils = types.ModuleType("inference_engine.utils")
    fake_engine_utils.__path__ = []
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils",
        fake_engine_utils,
    )

    fake_confidence = types.ModuleType(
        "inference_engine.utils.registration_confidence"
    )
    fake_confidence.validate_confidence_keep_ratio = float
    fake_confidence.select_top_confidence_mask = (
        lambda confidence, keep_ratio: torch.ones_like(
            confidence,
            dtype=torch.bool,
        )
    )

    def fake_intersect_confidence_masks(
        source_mask,
        target_mask,
        *,
        context="registration",
    ):
        mutual_mask = source_mask & target_mask
        if not torch.any(mutual_mask):
            raise ValueError(
                f"{context} has no shared high-confidence pixels"
            )
        return mutual_mask

    fake_confidence.intersect_confidence_masks = (
        fake_intersect_confidence_masks
    )
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils.registration_confidence",
        fake_confidence,
    )

    fake_geometry = types.ModuleType("inference_engine.utils.geometry")
    fake_geometry.accumulate_sim3 = compose_sim3
    fake_geometry.closed_form_inverse_sim3 = inverse_sim3
    monkeypatch.setitem(
        sys.modules,
        "inference_engine.utils.geometry",
        fake_geometry,
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))

    module_name = "loop_closure._pipeline_test_module"
    source_path = (
        Path(__file__).resolve().parents[1]
        / "loop_closure"
        / "loop_closure.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "loop_closure"
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, detector_instances, optimizer_instances


def make_engine(monkeypatch, tmp_path):
    module, detector_instances, optimizer_instances = (
        load_loop_engine_module(monkeypatch)
    )
    image_paths = [
        str(tmp_path / "frame1.png"),
        str(tmp_path / "frame2.png"),
        str(tmp_path / "frame3.png"),
    ]
    engine = module.LoopClosureEngine(
        engine_config(),
        tmp_path,
        tmp_path / "output",
        object(),
        window_size=2,
        overlap=1,
        image_paths=image_paths,
    )
    return (
        module,
        engine,
        detector_instances[0],
        optimizer_instances[0],
    )


def make_two_caches():
    return [
        make_cache(10.0, make_sim3(1.0)),
        make_cache(
            20.0,
            make_sim3(2.0),
            sim3_edge=make_sim3(2.0),
        ),
    ]


def test_no_loop_returns_original_absolute_transforms(
    monkeypatch,
    tmp_path,
):
    _, engine, detector, optimizer = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()
    detector.loop_list = []

    optimized_absolute = engine.run(caches)

    assert len(optimized_absolute) == len(caches)
    for actual, cache in zip(optimized_absolute, caches):
        assert_sim3_close(actual, cache["sim3_abs"])
    assert optimizer.calls == []


def test_invalid_loop_candidates_do_not_invoke_optimizer(
    monkeypatch,
    tmp_path,
):
    _, engine, detector, optimizer = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()
    detector.loop_list = [(0, 2)]
    engine.process_loops = lambda predictions: []

    optimized_absolute = engine.run(caches)

    assert len(optimized_absolute) == len(caches)
    assert optimizer.calls == []


def test_optimizer_receives_edges_and_returns_reaccumulated_absolutes(
    monkeypatch,
    tmp_path,
):
    _, engine, detector, optimizer = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()
    detector.loop_list = [(0, 2)]
    constraint = (0, 1, make_sim3(1.0))
    engine.process_loops = lambda predictions: [constraint]
    optimizer.optimized_edges = [make_sim3(3.0)]

    optimized_absolute = engine.run(caches)

    assert len(optimizer.calls) == 1
    input_edges, input_constraints = optimizer.calls[0]
    assert_sim3_close(input_edges[0], caches[1]["sim3_edge"])
    assert input_constraints == [constraint]
    assert len(optimized_absolute) == 2
    assert_sim3_close(optimized_absolute[0], make_sim3(1.0))
    assert_sim3_close(optimized_absolute[1], make_sim3(3.0))


def test_loop_constraints_convert_global_alignments_to_local_measurement(
    monkeypatch,
    tmp_path,
):
    module, engine, _, optimizer = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()
    engine.loop_list = [(0, 2)]
    monkeypatch.setattr(
        module,
        "process_loop_list",
        lambda *args, **kwargs: [
            (0, (0, 1), 1, (1, 2)),
        ],
    )
    joint_prediction = {
        "local_points": torch.ones((2, 1, 1, 3)),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
        "conf": torch.ones((2, 1, 1)),
    }
    engine.process_single_chunk = (
        lambda first_range, range_2=None: joint_prediction
    )
    registered_sources = []
    local_alignments = iter((make_sim3(2.0), make_sim3(6.0)))

    def fake_register(source_points, *args):
        registered_sources.append(source_points.clone())
        return next(local_alignments)

    monkeypatch.setattr(module, "register_adjacent_windows", fake_register)

    constraints = engine.process_loops(caches)

    torch.testing.assert_close(
        registered_sources[0],
        caches[0]["local_points"][:1],
    )
    torch.testing.assert_close(
        registered_sources[1],
        caches[1]["local_points"][:1],
    )
    assert len(constraints) == 1
    chunk_a, chunk_b, constraint_ab = constraints[0]
    assert (chunk_a, chunk_b) == (0, 1)
    assert_sim3_close(constraint_ab, make_sim3(1.5))
    assert optimizer.calls == []


def rotation_z(degrees):
    radians = torch.deg2rad(torch.tensor(float(degrees)))
    cosine = torch.cos(radians)
    sine = torch.sin(radians)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack((
        torch.stack((cosine, -sine, zero)),
        torch.stack((sine, cosine, zero)),
        torch.stack((zero, zero, one)),
    ))


def test_build_local_loop_constraint_recovers_joint_local_measurement(
    monkeypatch,
    tmp_path,
):
    module, _, _, _ = make_engine(monkeypatch, tmp_path)
    sim3_abs_a = (
        1.4,
        rotation_z(25.0),
        torch.tensor([3.0, -2.0, 1.0]),
    )
    sim3_abs_b = (
        0.8,
        rotation_z(-35.0),
        torch.tensor([-4.0, 1.5, 0.5]),
    )
    local_alignment_a = (
        1.1,
        rotation_z(12.0),
        torch.tensor([0.4, -0.2, 0.7]),
    )
    local_alignment_b = (
        0.9,
        rotation_z(-18.0),
        torch.tensor([-0.3, 0.8, -0.1]),
    )
    global_alignment_a = compose_sim3(
        sim3_abs_a,
        local_alignment_a,
    )
    global_alignment_b = compose_sim3(
        sim3_abs_b,
        local_alignment_b,
    )
    expected = compose_sim3(
        local_alignment_b,
        inverse_sim3(local_alignment_a),
    )

    actual = module.build_local_loop_constraint(
        sim3_abs_a,
        sim3_abs_b,
        global_alignment_a,
        global_alignment_b,
    )

    assert_sim3_close(actual, expected)


def test_loop_candidate_with_disjoint_confidence_is_skipped(
    monkeypatch,
    tmp_path,
):
    module, engine, _, optimizer = make_engine(monkeypatch, tmp_path)
    caches = make_two_caches()
    engine.loop_list = [(0, 2)]
    monkeypatch.setattr(
        module,
        "process_loop_list",
        lambda *args, **kwargs: [
            (0, (0, 1), 1, (1, 2)),
        ],
    )
    joint_prediction = {
        "local_points": torch.ones((2, 1, 1, 3)),
        "camera_poses": torch.eye(4).repeat(2, 1, 1),
        "conf": torch.ones((2, 1, 1)),
    }
    engine.process_single_chunk = (
        lambda first_range, range_2=None: joint_prediction
    )
    selections = iter((
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    ))
    monkeypatch.setattr(
        module,
        "select_top_confidence_mask",
        lambda *args: next(selections),
    )
    monkeypatch.setattr(
        module,
        "register_adjacent_windows",
        lambda *args: pytest.fail(
            "disjoint loop candidate must not be registered"
        ),
    )

    constraints = engine.process_loops(caches)

    assert constraints == []
    assert optimizer.calls == []


def test_cache_count_must_match_canonical_manifest(
    monkeypatch,
    tmp_path,
):
    _, engine, detector, _ = make_engine(monkeypatch, tmp_path)

    with pytest.raises(
        ValueError,
        match="cache count does not match image manifest",
    ):
        engine.run(make_two_caches()[:1])

    assert detector.run_calls == 0


@pytest.mark.parametrize("image_count", [1, 2, 3])
def test_chunk_indices_match_streaming_for_short_first_window(
    monkeypatch,
    tmp_path,
    image_count,
):
    _, engine, _, _ = make_engine(monkeypatch, tmp_path)
    engine.window_size = 4
    engine.overlap = 2
    engine.img_list = [
        str(tmp_path / f"frame{index}.png")
        for index in range(image_count)
    ]

    assert engine._build_chunk_indices() == [(0, image_count)]
