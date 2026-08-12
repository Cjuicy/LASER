from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from loop_closure.types import LoopCandidate
from pipeline.config import DetectionConfig
from pipeline.manifest import ImageManifest


class LoopDetector(Protocol):
    def detect(
        self,
        manifest: ImageManifest,
        images: torch.Tensor,
    ) -> tuple[LoopCandidate, ...]:
        raise NotImplementedError


class _LegacySaladBackend:
    def detect(self, config, manifest, images, output_path):
        del images
        from loop_closure.loop_model import LoopDetector as LegacyLoopDetector

        return LegacyLoopDetector(
            detection_config=config,
            image_manifest=manifest,
            output_path=output_path,
        ).run()


class SaladLoopDetector:
    def __init__(
        self,
        config: DetectionConfig,
        *,
        output_path: str | Path,
        backend: object | None = None,
    ) -> None:
        if not isinstance(config, DetectionConfig):
            raise ValueError("SALAD detector requires DetectionConfig")
        self.config = config
        self.output_path = Path(output_path)
        self.backend = backend or _LegacySaladBackend()

    def detect(
        self,
        manifest: ImageManifest,
        images: torch.Tensor,
    ) -> tuple[LoopCandidate, ...]:
        if not isinstance(manifest, ImageManifest):
            raise ValueError("loop detector manifest must be ImageManifest")
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("loop detector images must have shape (N,C,H,W)")
        if images.shape[0] != len(manifest):
            raise ValueError("loop detector image and manifest lengths must match")
        raw = self.backend.detect(
            self.config,
            manifest,
            images,
            self.output_path,
        )
        candidates = []
        for item in raw or ():
            if isinstance(item, LoopCandidate):
                candidates.append(item)
            else:
                try:
                    frame_a, frame_b, similarity = item
                except (TypeError, ValueError) as exc:
                    raise ValueError("SALAD backend candidate is invalid") from exc
                candidates.append(
                    LoopCandidate(
                        frame_a=int(frame_a),
                        frame_b=int(frame_b),
                        similarity=float(similarity),
                    )
                )
        return tuple(candidates)
