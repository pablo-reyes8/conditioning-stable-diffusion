from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.face_detection import FaceDetectionConfig, FaceDetectionEvaluator


class DummyDetector:
    def detect(self, image):
        red_channel = int(np.asarray(image)[0, 0, 0])
        if red_channel > 0:
            return np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.95])
        return None, None


def test_face_detection_evaluator_computes_detection_rate(tmp_path: Path):
    positive = tmp_path / "positive.png"
    negative = tmp_path / "negative.png"

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(positive)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(negative)

    evaluator = FaceDetectionEvaluator(
        config=FaceDetectionConfig(sample_failures=5),
        detector=DummyDetector(),
    )
    summary = evaluator.evaluate_paths([positive, negative])

    assert summary.total_images == 2
    assert summary.images_with_faces == 1
    assert summary.detection_rate == 0.5
    assert len(summary.sampled_missed_images) == 1
