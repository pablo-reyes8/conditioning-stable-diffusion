from pathlib import Path

from PIL import Image

from src.evaluation.io import list_image_paths


def test_list_image_paths_filters_non_image_files(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(images_dir / "a.png")
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(images_dir / "b.jpg")
    (images_dir / "ignore.txt").write_text("not-an-image", encoding="utf-8")

    image_paths = list_image_paths(images_dir)

    assert [path.name for path in image_paths] == ["a.png", "b.jpg"]
