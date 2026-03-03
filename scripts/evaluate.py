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

from src.evaluation.pipeline import run_evaluation_from_config
from src.utils.config import load_yaml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation CLI for generative quality and face detectability.")
    parser.add_argument("--config", default="config/evaluation/maad_face_eval.yaml")
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--disable-distribution-metrics", action="store_true")
    parser.add_argument("--disable-face-detection", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_yaml(args.config)
    evaluation_block = config.setdefault("evaluation", {})

    if args.generated_dir is not None:
        evaluation_block["generated_dir"] = args.generated_dir
    if args.real_dir is not None:
        evaluation_block["real_dir"] = args.real_dir
    if args.output is not None:
        evaluation_block["output_path"] = args.output

    distribution_block = evaluation_block.setdefault("distribution_metrics", {})
    face_block = evaluation_block.setdefault("face_detection", {})
    if args.disable_distribution_metrics:
        distribution_block["enabled"] = False
    if args.disable_face_detection:
        face_block["enabled"] = False

    summary = run_evaluation_from_config(config)
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
