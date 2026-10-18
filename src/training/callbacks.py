"""HF Trainer callback"""

from __future__ import annotations
from typing import Any
from transformers import TrainerCallback



_TRAIN_METRIC_KEYS = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm"
}

_DEFAULT_BEST_METRIC = "perplexity"
_DEFAULT_SECONDARY_METRICS = (
    "loss",
    "byte_level_perplexity"
)



class MLflowMetricsCallback(TrainerCallback):
    """Логгирует train/* и eval/* по шагу в MLflow"""

    def __init__(
            self, 
            best_metric: str = _DEFAULT_BEST_METRIC,
            greater_is_better: bool = True,
            secondary_metrics: tuple[str, ...] = _DEFAULT_SECONDARY_METRICS
    ) -> None:
        """Инициализирует callback и отслеживание лучшей эпохи"""


        self.enabled = mlflow_cfg.enabled
        self._greater_is_better = greater_is_better
        self._secondary = secondary_metrics
        self._best_value: float | None = None
        self._best_snapshot: dict[str, float] = {}
        self._best_epoch: float | None = None
        self._best_step: int | None = None
