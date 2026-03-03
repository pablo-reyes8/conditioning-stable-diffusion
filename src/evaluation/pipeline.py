from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.evaluation.face_detection import FaceDetectionConfig, FaceDetectionEvaluator
from src.evaluation.io import list_image_paths
from src.evaluation.metrics import DistributionMetricsConfig, TorchMetricsDistributionEvaluator
from src.utils.config import resolve_path


@dataclass(slots=True)
class EvaluationSummary:
    generated_dir: str
    real_dir: str | None
    generated_image_count: int
    real_image_count: int
    distribution_metrics: dict[str, float] | None
    face_detection_generated: dict[str, Any] | None
    face_detection_real: dict[str, Any] | None


def evaluate_generation_run(
    *,
    generated_dir: str | Path,
    real_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    recursive: bool = True,
    distribution_config: DistributionMetricsConfig | None = None,
    face_detection_config: FaceDetectionConfig | None = None,
    distribution_evaluator=None,
    face_evaluator=None,
) -> EvaluationSummary:
    generated_paths = list_image_paths(generated_dir, recursive=recursive)
    real_paths = list_image_paths(real_dir, recursive=recursive) if real_dir is not None else []

    if not generated_paths:
        raise ValueError("No generated images were found for evaluation.")
    if distribution_config and distribution_config.enabled and not real_paths:
        raise ValueError("Distribution metrics require a non-empty real image directory.")

    distribution_results = None
    if distribution_config and distribution_config.enabled:
        evaluator = distribution_evaluator or TorchMetricsDistributionEvaluator(distribution_config)
        evaluator.update_real_paths([str(path) for path in real_paths])
        evaluator.update_generated_paths([str(path) for path in generated_paths])
        distribution_results = evaluator.compute()

    face_generated = None
    face_real = None
    if face_detection_config and face_detection_config.enabled:
        detector = face_evaluator or FaceDetectionEvaluator(config=face_detection_config)
        face_generated = asdict(detector.evaluate_paths(generated_paths))
        if real_paths:
            face_real = asdict(detector.evaluate_paths(real_paths))

    summary = EvaluationSummary(
        generated_dir=str(Path(generated_dir)),
        real_dir=(str(Path(real_dir)) if real_dir is not None else None),
        generated_image_count=len(generated_paths),
        real_image_count=len(real_paths),
        distribution_metrics=distribution_results,
        face_detection_generated=face_generated,
        face_detection_real=face_real,
    )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    return summary


def run_evaluation_from_config(config: dict[str, Any]) -> EvaluationSummary:
    evaluation_config = config["evaluation"]
    top_level_device = str(config.get("device", "cpu"))

    distribution_block = evaluation_config.get("distribution_metrics", {})
    distribution_config = DistributionMetricsConfig(
        enabled=bool(distribution_block.get("enabled", True)),
        device=str(distribution_block.get("device", top_level_device)),
        batch_size=int(distribution_block.get("batch_size", 16)),
        image_size=int(distribution_block.get("image_size", 299)),
        fid=bool(distribution_block.get("fid", True)),
        kid=bool(distribution_block.get("kid", True)),
        inception_score=bool(distribution_block.get("inception_score", True)),
        kid_subset_size=int(distribution_block.get("kid_subset_size", 50)),
    )

    face_block = evaluation_config.get("face_detection", {})
    face_config = FaceDetectionConfig(
        enabled=bool(face_block.get("enabled", True)),
        detector=str(face_block.get("detector", "mtcnn")),
        device=str(face_block.get("device", top_level_device)),
        min_face_size=int(face_block.get("min_face_size", 20)),
        thresholds=tuple(face_block.get("thresholds", [0.6, 0.7, 0.7])),
        keep_all=bool(face_block.get("keep_all", True)),
        max_images=face_block.get("max_images"),
        sample_failures=int(face_block.get("sample_failures", 10)),
    )

    return evaluate_generation_run(
        generated_dir=resolve_path(evaluation_config["generated_dir"]),
        real_dir=(
            resolve_path(evaluation_config["real_dir"])
            if evaluation_config.get("real_dir")
            else None
        ),
        output_path=resolve_path(
            evaluation_config.get("output_path", "artifacts/evaluation/evaluation_report.json")
        ),
        recursive=bool(evaluation_config.get("recursive", True)),
        distribution_config=distribution_config,
        face_detection_config=face_config,
    )
