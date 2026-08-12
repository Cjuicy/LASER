from __future__ import annotations

import pytest

from pipeline.config import ReconstructionMode, load_pipeline_config
from reconstruction.modes.corrected import CorrectedReconstructionMode
from reconstruction.modes.no_loop import NoLoopReconstructionMode
from reconstruction.modes.traditional import TraditionalReconstructionMode
from reconstruction.registry import (
    ReconstructionServices,
    build_reconstruction_mode,
)


class Detector:
    def detect(self, manifest, images):
        return ()


class Evidence:
    def estimate(self, window_a, window_b, candidate):
        raise AssertionError("fixture has no candidates")


def _optimizer_config():
    return load_pipeline_config(
        "configs/reconstruction/pi3_laser.yaml",
        ("loop.optimizer.implementation=python",),
    ).config.loop.optimizer


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (ReconstructionMode.NO_LOOP, NoLoopReconstructionMode),
        (ReconstructionMode.TRADITIONAL, TraditionalReconstructionMode),
        (ReconstructionMode.CORRECTED, CorrectedReconstructionMode),
    ),
)
def test_registry_builds_exact_mode(mode, expected):
    services = (
        ReconstructionServices()
        if mode is ReconstructionMode.NO_LOOP
        else ReconstructionServices(
            detector=Detector(),
            evidence=Evidence(),
            optimizer_config=_optimizer_config(),
        )
    )

    assert isinstance(build_reconstruction_mode(mode, services), expected)


def test_no_loop_registry_rejects_accidental_loop_services():
    with pytest.raises(ValueError, match="no_loop.*loop services"):
        build_reconstruction_mode(
            ReconstructionMode.NO_LOOP,
            ReconstructionServices(
                detector=Detector(),
                evidence=Evidence(),
                optimizer_config=_optimizer_config(),
            ),
        )


@pytest.mark.parametrize(
    "mode",
    (ReconstructionMode.TRADITIONAL, ReconstructionMode.CORRECTED),
)
def test_loop_registry_requires_complete_services(mode):
    with pytest.raises(ValueError, match=f"{mode.value}.*loop services"):
        build_reconstruction_mode(mode, ReconstructionServices())
