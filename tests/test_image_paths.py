from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.image_paths import discover_images, natural_sort_key


def test_natural_sort_key_orders_embedded_numbers_numerically():
    names = ["frame10.jpg", "Frame2.jpg", "frame1.jpg"]

    assert sorted(names, key=natural_sort_key) == [
        "frame1.jpg",
        "Frame2.jpg",
        "frame10.jpg",
    ]


def test_discover_images_filters_sorts_then_samples(tmp_path):
    for name in [
        "frame10.jpg",
        "frame2.PNG",
        "frame1.jpeg",
        "notes.txt",
        "frame3.JPG",
    ]:
        (tmp_path / name).touch()

    image_names = discover_images(tmp_path, sample_interval=2)

    assert [Path(path).name for path in image_names] == [
        "frame1.jpeg",
        "frame3.JPG",
    ]


def test_discover_images_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        discover_images(tmp_path / "missing")


@pytest.mark.parametrize("sample_interval", [0, -1])
def test_discover_images_rejects_non_positive_interval(tmp_path, sample_interval):
    with pytest.raises(ValueError, match="sample_interval must be at least 1"):
        discover_images(tmp_path, sample_interval=sample_interval)
