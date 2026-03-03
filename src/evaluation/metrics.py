from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.evaluation.io import iter_image_batches


@dataclass(slots=True)
class DistributionMetricsConfig:
    enabled: bool = True
    device: str = "cpu"
    batch_size: int = 16
    image_size: int = 299
    fid: bool = True
    kid: bool = True
    inception_score: bool = True
    kid_subset_size: int = 50


class TorchMetricsDistributionEvaluator:
    """Standard generative metrics backed by torchmetrics."""

    def __init__(self, config: DistributionMetricsConfig):
        self.config = config
        self._metrics: dict[str, Any] = {}

        if not config.enabled:
            return

        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.inception import InceptionScore
            from torchmetrics.image.kid import KernelInceptionDistance
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Evaluation metrics require `torchmetrics[image]` and `torch-fidelity`."
            ) from exc

        if config.fid:
            self._metrics["fid"] = FrechetInceptionDistance(
                feature=2048,
                normalize=True,
            ).to(config.device)
        if config.kid:
            self._metrics["kid"] = KernelInceptionDistance(
                subset_size=config.kid_subset_size,
                normalize=True,
            ).to(config.device)
        if config.inception_score:
            self._metrics["inception_score"] = InceptionScore(normalize=True).to(config.device)

    def update_real_paths(self, image_paths: list[str]) -> None:
        if not self._metrics:
            return

        for batch in iter_image_batches(
            image_paths,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            device=self.config.device,
        ):
            if "fid" in self._metrics:
                self._metrics["fid"].update(batch, real=True)
            if "kid" in self._metrics:
                self._metrics["kid"].update(batch, real=True)

    def update_generated_paths(self, image_paths: list[str]) -> None:
        if not self._metrics:
            return

        for batch in iter_image_batches(
            image_paths,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            device=self.config.device,
        ):
            if "fid" in self._metrics:
                self._metrics["fid"].update(batch, real=False)
            if "kid" in self._metrics:
                self._metrics["kid"].update(batch, real=False)
            if "inception_score" in self._metrics:
                self._metrics["inception_score"].update(batch)

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        if "fid" in self._metrics:
            results["fid"] = float(self._metrics["fid"].compute().item())
        if "kid" in self._metrics:
            kid_mean, kid_std = self._metrics["kid"].compute()
            results["kid_mean"] = float(kid_mean.item())
            results["kid_std"] = float(kid_std.item())
        if "inception_score" in self._metrics:
            is_mean, is_std = self._metrics["inception_score"].compute()
            results["inception_score_mean"] = float(is_mean.item())
            results["inception_score_std"] = float(is_std.item())
        return results
