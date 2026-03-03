from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.evaluation.io import list_image_paths, load_pil_image


class FaceDetector(Protocol):
    def detect(self, image): ...


@dataclass(slots=True)
class FaceDetectionConfig:
    enabled: bool = True
    detector: str = "mtcnn"
    device: str = "cpu"
    min_face_size: int = 20
    thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7)
    keep_all: bool = True
    max_images: int | None = None
    sample_failures: int = 10


@dataclass(slots=True)
class FaceDetectionSummary:
    detector: str
    total_images: int
    images_with_faces: int
    detection_rate: float
    total_faces_detected: int
    mean_faces_per_image: float
    mean_confidence: float | None
    sampled_missed_images: list[str]


class MTCNNFaceDetector:
    def __init__(self, config: FaceDetectionConfig):
        try:
            from facenet_pytorch import MTCNN
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Face detection evaluation requires `facenet-pytorch`."
            ) from exc

        self.model = MTCNN(
            keep_all=config.keep_all,
            device=config.device,
            min_face_size=config.min_face_size,
            thresholds=list(config.thresholds),
        )

    def detect(self, image):
        return self.model.detect(image)


class FaceDetectionEvaluator:
    def __init__(
        self,
        *,
        config: FaceDetectionConfig | None = None,
        detector: FaceDetector | None = None,
    ):
        self.config = config or FaceDetectionConfig()
        self.detector = detector or MTCNNFaceDetector(self.config)

    def evaluate_directory(self, image_root: str | Path) -> FaceDetectionSummary:
        image_paths = list_image_paths(
            image_root,
            recursive=True,
            max_images=self.config.max_images,
        )
        return self.evaluate_paths(image_paths)

    def evaluate_paths(self, image_paths: list[str | Path]) -> FaceDetectionSummary:
        selected_paths = [Path(path) for path in image_paths]
        if self.config.max_images is not None:
            selected_paths = selected_paths[: self.config.max_images]

        detections = 0
        images_with_faces = 0
        confidences: list[float] = []
        missed_images: list[str] = []

        for image_path in selected_paths:
            image = load_pil_image(image_path)
            boxes, probs = self.detector.detect(image)

            num_faces = 0
            if boxes is not None:
                num_faces = int(len(boxes))

            if num_faces > 0:
                images_with_faces += 1
                detections += num_faces
                if probs is not None:
                    for prob in np.asarray(probs).reshape(-1).tolist():
                        if prob is None:
                            continue
                        prob = float(prob)
                        if not np.isnan(prob):
                            confidences.append(prob)
            elif len(missed_images) < self.config.sample_failures:
                missed_images.append(str(image_path))

        total_images = len(selected_paths)
        detection_rate = (images_with_faces / total_images) if total_images else 0.0
        mean_faces = (detections / total_images) if total_images else 0.0
        mean_confidence = (sum(confidences) / len(confidences)) if confidences else None

        return FaceDetectionSummary(
            detector=self.config.detector,
            total_images=total_images,
            images_with_faces=images_with_faces,
            detection_rate=round(detection_rate, 6),
            total_faces_detected=detections,
            mean_faces_per_image=round(mean_faces, 6),
            mean_confidence=(round(mean_confidence, 6) if mean_confidence is not None else None),
            sampled_missed_images=missed_images,
        )
