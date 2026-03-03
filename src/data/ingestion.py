from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.constants import DEFAULT_ATTRIBUTES


@dataclass(slots=True)
class ManifestArtifacts:
    source_path: str
    manifest_path: str
    report_path: str
    requested_samples: int
    selected_samples: int
    positive_rate_pct: dict[str, float]
    flagged_correlations: list[dict[str, float | str]]


@dataclass(slots=True)
class ArchiveArtifacts:
    source_archive: str
    manifest_path: str
    output_archive: str
    report_path: str
    matched_files: int
    manifest_entries: int
    zip_prefix: str


@dataclass(slots=True)
class DownloadArtifacts:
    source_url: str
    output_path: str
    bytes_downloaded: int
    sha256: str


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix in {".csv", ".tsv"}:
        sep = "\t" if table_path.suffix == ".tsv" else ","
        return pd.read_csv(table_path, sep=sep)
    raise ValueError(f"Unsupported table format: {table_path}")


def validate_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> list[str]:
    return [column for column in required_columns if column not in df.columns]


def build_balanced_manifest(
    source_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    *,
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    sample_size: int = 100_000,
    random_state: int = 42,
    filename_column: str = "Filename",
    identity_column: str = "Identity",
    correlation_threshold: float = 0.25,
) -> ManifestArtifacts:
    df = load_table(source_path)
    required_columns = [filename_column, identity_column, *attributes]
    missing = validate_columns(df, required_columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if sample_size > len(df):
        raise ValueError(
            f"Requested sample_size={sample_size} exceeds dataset size={len(df)}."
        )

    work = df[[filename_column, identity_column, *attributes]].copy()
    for attr in attributes:
        work[attr] = (work[attr] == 1).astype(np.int8)

    positive_frequency = work[list(attributes)].mean()
    if ((positive_frequency <= 0.0) | (positive_frequency >= 1.0)).any():
        invalid = positive_frequency[
            (positive_frequency <= 0.0) | (positive_frequency >= 1.0)
        ]
        raise ValueError(
            "Balancing requires attributes with both positive and negative examples. "
            f"Invalid attributes: {invalid.to_dict()}"
        )

    weights = np.zeros(len(work), dtype=np.float64)
    for attr in attributes:
        pos_weight = 1.0 / float(positive_frequency[attr])
        neg_weight = 1.0 / float(1.0 - positive_frequency[attr])
        weights += np.where(work[attr] == 1, pos_weight, neg_weight)

    weights /= weights.sum()
    balanced = work.sample(
        n=sample_size,
        weights=weights,
        random_state=random_state,
        replace=False,
    )

    corr_matrix = balanced[list(attributes)].corr()
    np.fill_diagonal(corr_matrix.values, 0.0)
    red_flags: list[dict[str, float | str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for left in attributes:
        for right in attributes:
            if left == right:
                continue
            pair = tuple(sorted((left, right)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            correlation = float(corr_matrix.loc[left, right])
            if abs(correlation) >= correlation_threshold:
                red_flags.append(
                    {
                        "left": pair[0],
                        "right": pair[1],
                        "correlation": round(correlation, 6),
                    }
                )

    manifest_output = Path(manifest_path)
    report_output = Path(report_path)
    _ensure_parent(manifest_output)
    _ensure_parent(report_output)
    balanced.to_csv(manifest_output, index=False)

    artifacts = ManifestArtifacts(
        source_path=str(Path(source_path)),
        manifest_path=str(manifest_output),
        report_path=str(report_output),
        requested_samples=int(sample_size),
        selected_samples=int(len(balanced)),
        positive_rate_pct={
            attr: round(float(value * 100.0), 4)
            for attr, value in balanced[list(attributes)].mean().items()
        },
        flagged_correlations=sorted(
            red_flags,
            key=lambda item: abs(float(item["correlation"])),
            reverse=True,
        ),
    )

    report_payload = asdict(artifacts)
    report_payload["attributes"] = list(attributes)
    report_payload["random_state"] = int(random_state)
    report_payload["correlation_threshold"] = float(correlation_threshold)
    report_output.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return artifacts


def _candidate_names(member_name: str, zip_prefix: str) -> Iterable[str]:
    normalized = member_name.replace("\\", "/")
    yield normalized
    if zip_prefix:
        prefix = zip_prefix.rstrip("/") + "/"
        if normalized.startswith(prefix):
            yield normalized[len(prefix) :]
        else:
            yield prefix + normalized
    parts = normalized.split("/")
    if len(parts) >= 2:
        yield "/".join(parts[-2:])
    if len(parts) >= 1:
        yield parts[-1]


def filter_archive_by_manifest(
    source_archive: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    output_archive: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    *,
    filename_column: str = "Filename",
    zip_prefix: str = "train/",
) -> ArchiveArtifacts:
    manifest_df = load_table(manifest_path)
    if filename_column not in manifest_df.columns:
        raise ValueError(f"Manifest missing required column: {filename_column}")

    wanted_files = {
        str(name).replace("\\", "/") for name in manifest_df[filename_column].astype(str)
    }
    output_path = Path(output_archive)
    report_output = Path(report_path)
    _ensure_parent(output_path)
    _ensure_parent(report_output)

    matched_files = 0
    with zipfile.ZipFile(source_archive, "r") as source_zip, zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_zip:
        for info in source_zip.infolist():
            if info.is_dir():
                continue

            normalized = info.filename.replace("\\", "/")
            if not normalized.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            if any(candidate in wanted_files for candidate in _candidate_names(normalized, zip_prefix)):
                output_zip.writestr(info, source_zip.read(info))
                matched_files += 1

    artifacts = ArchiveArtifacts(
        source_archive=str(Path(source_archive)),
        manifest_path=str(Path(manifest_path)),
        output_archive=str(output_path),
        report_path=str(report_output),
        matched_files=matched_files,
        manifest_entries=int(len(wanted_files)),
        zip_prefix=zip_prefix,
    )
    report_output.write_text(json.dumps(asdict(artifacts), indent=2), encoding="utf-8")
    return artifacts


def compute_sha256(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    source_url: str,
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    expected_sha256: str | None = None,
) -> DownloadArtifacts:
    destination = Path(output_path)
    _ensure_parent(destination)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}. Use overwrite=True to replace it."
            )
        if destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)

    with urllib.request.urlopen(source_url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    digest = compute_sha256(destination)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch for {destination}: expected {expected_sha256}, got {digest}"
        )

    return DownloadArtifacts(
        source_url=source_url,
        output_path=str(destination),
        bytes_downloaded=destination.stat().st_size,
        sha256=digest,
    )
