from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class ImageManifest:
    paths: tuple[Path, ...]

    def __len__(self) -> int:
        return len(self.paths)

    def as_strings(self) -> list[str]:
        return [str(path) for path in self.paths]


def natural_sort_key(path: str | Path) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", Path(path).name.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def discover_image_manifest(
    image_dir: str | Path,
    sample_stride: int,
) -> ImageManifest:
    if sample_stride < 1:
        raise ValueError("input.sample_stride must be at least 1")

    directory = Path(image_dir)
    if not directory.exists():
        raise FileNotFoundError(f"input.image_dir does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(
            f"input.image_dir is not a directory: {directory}"
        )

    candidates = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=natural_sort_key,
    )
    sampled = tuple(path.resolve() for path in candidates[::sample_stride])
    if not sampled:
        raise ValueError(
            f"input.image_dir contains no supported images: {directory}"
        )
    return ImageManifest(paths=sampled)

