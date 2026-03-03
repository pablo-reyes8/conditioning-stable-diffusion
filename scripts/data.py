#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import build_balanced_manifest, download_file, filter_archive_by_manifest
from src.utils.config import load_yaml, resolve_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data ingestion CLI for the Stable Diffusion project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest", help="Create a balanced manifest CSV and JSON report.")
    manifest_parser.add_argument("--config", default="config/data/maad_face.yaml")
    manifest_parser.add_argument("--input", dest="input_path", default=None)
    manifest_parser.add_argument("--output", dest="output_path", default=None)
    manifest_parser.add_argument("--report", dest="report_path", default=None)
    manifest_parser.add_argument("--sample-size", type=int, default=None)

    archive_parser = subparsers.add_parser("filter-archive", help="Filter a source archive using a manifest.")
    archive_parser.add_argument("--config", default="config/data/maad_face.yaml")
    archive_parser.add_argument("--input-archive", default=None)
    archive_parser.add_argument("--manifest", default=None)
    archive_parser.add_argument("--output-archive", default=None)
    archive_parser.add_argument("--report", dest="report_path", default=None)

    download_parser = subparsers.add_parser("download", help="Download a dataset artifact from a remote URL.")
    download_parser.add_argument("--config", default="config/data/maad_face.yaml")
    download_parser.add_argument("--url", default=None)
    download_parser.add_argument("--output", default=None)
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument("--sha256", default=None)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_yaml(args.config)
    data_cfg = config.get("dataset", {})

    if args.command == "build-manifest":
        artifacts = build_balanced_manifest(
            source_path=resolve_path(args.input_path or data_cfg["metadata_path"]),
            manifest_path=resolve_path(args.output_path or data_cfg["manifest_path"]),
            report_path=resolve_path(args.report_path or data_cfg["manifest_report_path"]),
            attributes=data_cfg.get("attributes", []),
            sample_size=args.sample_size or int(data_cfg.get("sample_size", 100_000)),
            random_state=int(data_cfg.get("random_state", 42)),
            filename_column=str(data_cfg.get("filename_column", "Filename")),
            identity_column=str(data_cfg.get("identity_column", "Identity")),
            correlation_threshold=float(data_cfg.get("correlation_threshold", 0.25)),
        )
        print(json.dumps(asdict(artifacts), indent=2))
        return 0

    if args.command == "filter-archive":
        artifacts = filter_archive_by_manifest(
            source_archive=resolve_path(args.input_archive or data_cfg["source_archive_path"]),
            manifest_path=resolve_path(args.manifest or data_cfg["manifest_path"]),
            output_archive=resolve_path(args.output_archive or data_cfg["filtered_archive_path"]),
            report_path=resolve_path(args.report_path or data_cfg["archive_report_path"]),
            filename_column=str(data_cfg.get("filename_column", "Filename")),
            zip_prefix=str(data_cfg.get("zip_prefix", "train/")),
        )
        print(json.dumps(asdict(artifacts), indent=2))
        return 0

    artifacts = download_file(
        source_url=str(args.url or data_cfg["download_url"]),
        output_path=resolve_path(args.output or data_cfg["download_output_path"]),
        overwrite=bool(args.overwrite),
        expected_sha256=args.sha256 or data_cfg.get("download_sha256"),
    )
    print(json.dumps(asdict(artifacts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
