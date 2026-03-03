from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.face_detection import FaceDetectionConfig, FaceDetectionEvaluator
from src.evaluation.metrics import DistributionMetricsConfig
from src.evaluation.pipeline import evaluate_generation_run


class DummyDetector:
    def detect(self, image):
        red_channel = int(np.asarray(image)[0, 0, 0])
        if red_channel > 0:
            return np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.9])
        return None, None


class FakeDistributionEvaluator:
    def __init__(self):
        self.real_paths = []
        self.generated_paths = []

    def update_real_paths(self, image_paths):
        self.real_paths.extend(image_paths)

    def update_generated_paths(self, image_paths):
        self.generated_paths.extend(image_paths)

    def compute(self):
        return {"fid": 12.34}


def test_evaluation_pipeline_combines_distribution_and_face_detection(tmp_path: Path):
    generated_dir = tmp_path / "generated"
    real_dir = tmp_path / "real"
    generated_dir.mkdir()
    real_dir.mkdir()

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(generated_dir / "gen.png")
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(real_dir / "real.png")

    distribution = FakeDistributionEvaluator()
    face_evaluator = FaceDetectionEvaluator(
        config=FaceDetectionConfig(sample_failures=5),
        detector=DummyDetector(),
    )

    summary = evaluate_generation_run(
        generated_dir=generated_dir,
        real_dir=real_dir,
        distribution_config=DistributionMetricsConfig(enabled=True),
        face_detection_config=FaceDetectionConfig(enabled=True),
        distribution_evaluator=distribution,
        face_evaluator=face_evaluator,
    )

    assert summary.generated_image_count == 1
    assert summary.real_image_count == 1
    assert summary.distribution_metrics == {"fid": 12.34}
    assert summary.face_detection_generated["images_with_faces"] == 1
    assert len(distribution.real_paths) == 1
    assert len(distribution.generated_paths) == 1
