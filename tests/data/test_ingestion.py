import json
import zipfile
from pathlib import Path

import pandas as pd

from src.data.ingestion import build_balanced_manifest, download_file, filter_archive_by_manifest


def _write_source_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "Filename": [f"n0001/img_{idx:03d}.jpg" for idx in range(8)],
            "Identity": [1, 1, 2, 2, 3, 3, 4, 4],
            "Male": [1, 0, 1, 0, 1, 0, 1, 0],
            "Young": [1, 1, 0, 0, 1, 1, 0, 0],
            "Senior": [0, 0, 1, 1, 0, 0, 1, 1],
            "Asian": [1, 0, 1, 0, 1, 0, 1, 0],
            "Black": [0, 1, 0, 1, 0, 1, 0, 1],
            "Smiling": [1, 1, 1, 0, 0, 0, 1, 0],
            "Blond_Hair": [0, 1, 0, 1, 0, 1, 0, 1],
            "Chubby": [0, 0, 1, 1, 0, 0, 1, 1],
            "Heavy_Makeup": [1, 0, 0, 1, 1, 0, 0, 1],
            "Black_Hair": [1, 1, 0, 0, 1, 1, 0, 0],
            "Big_Nose": [0, 1, 1, 0, 0, 1, 1, 0],
        }
    )
    df.to_csv(path, index=False)


def test_build_balanced_manifest_creates_csv_and_report(tmp_path: Path):
    source_path = tmp_path / "source.csv"
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"
    _write_source_csv(source_path)

    artifacts = build_balanced_manifest(
        source_path=source_path,
        manifest_path=manifest_path,
        report_path=report_path,
        sample_size=4,
    )

    manifest = pd.read_csv(manifest_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(manifest) == 4
    assert artifacts.selected_samples == 4
    assert report["requested_samples"] == 4
    assert "Smiling" in report["positive_rate_pct"]


def test_filter_archive_by_manifest_only_keeps_matching_files(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    archive_path = tmp_path / "source.zip"
    output_archive = tmp_path / "filtered.zip"
    report_path = tmp_path / "archive_report.json"

    pd.DataFrame({"Filename": ["n0001/img_000.jpg", "n0001/img_003.jpg"]}).to_csv(
        manifest_path, index=False
    )

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train/n0001/img_000.jpg", b"image-0")
        archive.writestr("train/n0001/img_003.jpg", b"image-3")
        archive.writestr("train/n0001/img_999.jpg", b"image-999")

    artifacts = filter_archive_by_manifest(
        source_archive=archive_path,
        manifest_path=manifest_path,
        output_archive=output_archive,
        report_path=report_path,
    )

    with zipfile.ZipFile(output_archive, "r") as archive:
        names = sorted(archive.namelist())

    assert names == ["train/n0001/img_000.jpg", "train/n0001/img_003.jpg"]
    assert artifacts.matched_files == 2


def test_download_file_supports_local_file_urls(tmp_path: Path):
    source_path = tmp_path / "source.bin"
    target_path = tmp_path / "downloads" / "copied.bin"
    source_path.write_bytes(b"stable-diffusion")

    artifacts = download_file(source_path.resolve().as_uri(), target_path)

    assert target_path.read_bytes() == b"stable-diffusion"
    assert artifacts.bytes_downloaded == len(b"stable-diffusion")
    assert len(artifacts.sha256) == 64
