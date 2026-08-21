from __future__ import annotations

from collections.abc import Mapping

import torch

from inference_engine.inference_utils import (
    estimate_pseudo_depth_and_intrinsics,
    unproject_depth_to_local_points,
)
from inference_engine.models.lazy import (
    LazyModelHandle,
    ModelForwardKind,
)
from pipeline.config import PredictionCacheMode

from .store import (
    OrdinaryPredictionStore,
    PredictionCacheCorruptError,
    PredictionCacheMissError,
)
from .types import (
    OrdinaryWindowArtifact,
    SequenceArtifact,
    WindowSpec,
)


class OrdinaryPredictionProvider(torch.nn.Module):
    def __init__(
        self,
        *,
        store: OrdinaryPredictionStore,
        model: LazyModelHandle,
    ) -> None:
        super().__init__()
        if not isinstance(store, OrdinaryPredictionStore):
            raise ValueError(
                "ordinary prediction provider store is invalid"
            )
        if not isinstance(model, LazyModelHandle):
            raise ValueError(
                "ordinary prediction provider model is invalid"
            )
        self.store = store
        self.model = model
        self._reference_intrinsic: torch.Tensor | None = None
        self._sequence_checked = False

    def forward(
        self,
        images: torch.Tensor,
        *,
        window_spec: WindowSpec,
    ) -> dict[str, torch.Tensor]:
        return self.get(window_spec, images)

    @property
    def reference_intrinsic(self) -> torch.Tensor | None:
        if self._reference_intrinsic is None:
            return None
        return self._reference_intrinsic.clone()

    @property
    def prediction_key(self) -> str:
        return self.store.fingerprint.key

    def _load_sequence_once(self) -> None:
        if self._sequence_checked:
            return
        sequence = self.store.read_sequence()
        if sequence is not None:
            self._reference_intrinsic = (
                sequence.reference_intrinsic.detach().cpu().clone()
            )
        self._sequence_checked = True

    @staticmethod
    def _validate_images(
        spec: WindowSpec,
        images: torch.Tensor,
    ) -> None:
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[0] != spec.frame_count
            or images.shape[1] != 3
        ):
            raise ValueError(
                "ordinary images must have shape (window_frames,3,H,W)"
            )
        if not torch.isfinite(images).all():
            raise ValueError("ordinary images must contain finite values")

    @staticmethod
    def _prediction_tensors(
        prediction: Mapping[str, object],
        spec: WindowSpec,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = {
            "local_points": (1, spec.frame_count, height, width, 3),
            "camera_poses": (1, spec.frame_count, 4, 4),
            "conf": (1, spec.frame_count, height, width),
        }
        values = []
        for key in ("local_points", "camera_poses", "conf"):
            value = prediction.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"ordinary prediction is missing tensor {key!r}"
                )
            if tuple(value.shape) != expected[key]:
                raise ValueError(
                    f"ordinary prediction {key!r} has shape "
                    f"{tuple(value.shape)}, expected {expected[key]}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(
                    f"ordinary prediction {key!r} contains non-finite values"
                )
            values.append(value.squeeze(0))
        return values[0], values[1], values[2]

    def _prediction_from_artifact(
        self,
        artifact: OrdinaryWindowArtifact,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self._reference_intrinsic is None:
            raise RuntimeError(
                "reference intrinsic is unavailable for cache replay"
            )
        local_points = unproject_depth_to_local_points(
            artifact.depth.clone(),
            self._reference_intrinsic.clone(),
        )
        if not torch.isfinite(local_points).all():
            raise ValueError(
                "cached depth produced non-finite reconstructed local points"
            )
        return self._with_reference_intrinsic(
            {
                "local_points": local_points.unsqueeze(0),
                "camera_poses": artifact.camera_poses.clone().unsqueeze(0),
                "conf": artifact.confidence.clone().unsqueeze(0),
                "images": images.detach().clone().unsqueeze(0),
            }
        )

    def _with_reference_intrinsic(
        self,
        prediction: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self._reference_intrinsic is None:
            raise RuntimeError("reference intrinsic is unavailable")
        return {
            **prediction,
            "reference_intrinsic": self._reference_intrinsic.clone(),
        }

    def get(
        self,
        spec: WindowSpec,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if spec not in self.store.expected_specs:
            raise ValueError("WindowSpec does not belong to this provider")
        self._validate_images(spec, images)
        try:
            self._load_sequence_once()
        except (
            PredictionCacheMissError,
            PredictionCacheCorruptError,
        ) as exc:
            raise self.store.contextualize_read_error(spec, exc) from exc

        artifact = self.store.read_window(spec)
        if artifact is not None and self._reference_intrinsic is not None:
            return self._prediction_from_artifact(artifact, images)

        if self._reference_intrinsic is None and spec.index != 0:
            raise ValueError(
                "first window must establish the reference intrinsic "
                "before later windows"
            )

        prediction = self.model.predict(
            images,
            kind=ModelForwardKind.ORDINARY,
        )
        height, width = int(images.shape[-2]), int(images.shape[-1])
        local_points, camera_poses, confidence = (
            self._prediction_tensors(
                prediction,
                spec,
                height,
                width,
            )
        )
        if self._reference_intrinsic is None:
            _, intrinsics = estimate_pseudo_depth_and_intrinsics(
                local_points
            )
            reference = intrinsics[0].detach().cpu()
            if not torch.isfinite(reference).all():
                raise ValueError(
                    "first ordinary prediction produced invalid intrinsic"
                )
            sequence = SequenceArtifact(reference.clone())
            self._reference_intrinsic = reference.clone()
            self.store.write_sequence(sequence)

        compact = OrdinaryWindowArtifact(
            spec=spec,
            depth=local_points[..., -1].detach().cpu(),
            confidence=confidence.detach().cpu(),
            camera_poses=camera_poses.detach().cpu(),
        )
        self.store.write_window(compact)
        if (
            self.store.mode is not PredictionCacheMode.OFF
            and spec == self.store.expected_specs[-1]
        ):
            self.store.finalize()
        return self._prediction_from_artifact(compact, images)
