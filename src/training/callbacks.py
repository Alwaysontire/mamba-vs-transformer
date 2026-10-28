"""HF Trainer callback"""

from __future__ import annotations
import math

import mlflow
from transformers import TrainerCallback


_TRAIN_METRIC_KEYS = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
}

_EVAL_METRIC_KEYS = {
    "eval_loss": "eval/loss",
}


class MLflowMetricsCallback(TrainerCallback):
    """Логгирует train/* и eval/* по шагу в активный run MLflow (его открывает kedro-mlflow)"""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs or not state.is_world_process_zero or mlflow.active_run() is None:
            return

        metrics = {
            name: float(logs[key])
            for key, name in {**_TRAIN_METRIC_KEYS, **_EVAL_METRIC_KEYS}.items()
            if isinstance(logs.get(key), (int, float))
        }
        if "eval/loss" in metrics:
            metrics["eval/perplexity"] = math.exp(metrics["eval/loss"])
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)
