from __future__ import annotations

from pathlib import Path

from pipeline.config import DetectionConfig
from pipeline.manifest import ImageManifest

from .base import LoopCandidate


def detect_loop_candidates(
    detection_config: DetectionConfig,
    image_manifest: ImageManifest,
    output_path: str | Path,
) -> tuple[LoopCandidate, ...]:
    from loop_closure.loop_model import LoopDetector

    detector = LoopDetector(
        detection_config=detection_config,
        image_manifest=image_manifest,
        output_path=Path(output_path),
    )
    candidates = detector.run()
    if candidates is None:
        return ()
    if not isinstance(candidates, tuple) or not all(
        isinstance(candidate, LoopCandidate) for candidate in candidates
    ):
        raise TypeError(
            "LoopDetector.run() must return tuple[LoopCandidate, ...]"
        )
    return candidates
