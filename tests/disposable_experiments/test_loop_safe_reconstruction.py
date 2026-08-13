from __future__ import annotations

from tools.disposable_experiments.preflight_experiments import (
    build_loop_detector_outside_artifact,
)


def test_loop_detector_output_does_not_create_final_artifact_directory(tmp_path):
    requested = (
        tmp_path
        / "method"
        / "artifact"
        / "depth-traditional"
        / "loop_candidates.json"
    )
    observed = {}

    def fake_factory(config, *, output_path):
        observed["config"] = config
        observed["output_path"] = output_path
        output_path.parent.mkdir(parents=True)
        output_path.write_text("[]", encoding="utf-8")
        return object()

    detector = build_loop_detector_outside_artifact(
        "detection-config",
        output_path=requested,
        detector_factory=fake_factory,
    )

    assert detector is not None
    assert observed == {
        "config": "detection-config",
        "output_path": requested.parent.with_suffix(".loop_candidates.json"),
    }
    assert observed["output_path"].is_file()
    assert not requested.parent.exists()
