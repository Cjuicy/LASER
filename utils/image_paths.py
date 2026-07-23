import os
from pathlib import Path
import re


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def natural_sort_key(
    path: str | os.PathLike[str],
) -> list[tuple[int, str | int]]:
    parts = re.split(r"(\d+)", Path(path).name.lower())
    return [
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
    ]


def discover_images(
    data_path: str | os.PathLike[str],
    sample_interval: int = 1,
) -> list[str]:
    if sample_interval < 1:
        raise ValueError("sample_interval must be at least 1")

    image_dir = Path(data_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_paths.sort(key=natural_sort_key)
    return [str(path) for path in image_paths[::sample_interval]]
