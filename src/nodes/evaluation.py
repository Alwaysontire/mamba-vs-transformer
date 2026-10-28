"""Нода evaluate: прямой проход по test-сплиту и подсчёт метрик"""


from __future__ import annotations

import logging

import numpy as np

from src.config.schemas import EvaluationConfig, TrainingConfig
from src.datasets.train_artifact import TrainArtifact

logger = logging.getLogger(__name__)


def evaluate_node(
        train_artifact: TrainArtifact,
        tokens_test: np.ndarray,
        tokenization_stats: dict,
        training: dict,
        evaluation: dict,
) -> dict:
    """
    Args:
        train_artifact (TrainArtifact): Обученная модель и manifest.
        tokens_test (np.ndarray): 1D поток токенов test (memmap).
        tokenization_stats (dict): ``splits.test.n_words`` — знаменатель для word PPL.
        training (dict): Параметры обучения; используются ``device`` и ``trainer.bf16``.
        evaluation (dict): Параметры оценки для :class:`EvaluationConfig`.

    Returns:
        dict: ``final_metrics`` в формате :class:`src.reporting.inputs.EvaluationMetrics`.
    """
    import torch

    from src.evaluation.evaluate import resolve_device, sliding_window_nll
    from src.evaluation.metrics import compute_metrics

    model = train_artifact.model
    eval_cfg = EvaluationConfig(**evaluation)
    training_cfg = TrainingConfig(**training)

    max_length = getattr(model, "max_length", None)
    context_length = eval_cfg.context_length or max_length
    if context_length is None:
        raise ValueError("evaluation.context_length не задан и у модели нет max_length")
    if max_length is not None and context_length > max_length:
        raise ValueError(f"evaluation.context_length={context_length} больше max_length модели {max_length}")
    stride = eval_cfg.stride or max(context_length // 2, 1)
    if stride > context_length:
        raise ValueError(f"evaluation.stride={stride} больше context_length={context_length}")

    device = resolve_device(training_cfg.device)
    use_bf16 = (
        device.type == "cuda"
        and training_cfg.trainer.bf16 is not False
        and torch.cuda.is_bf16_supported()
    )

    nll_sum, n_scored = sliding_window_nll(
        model,
        tokens_test,
        context_length=context_length,
        stride=stride,
        batch_size=eval_cfg.batch_size,
        device=device,
        autocast_dtype=torch.bfloat16 if use_bf16 else None,
    )

    n_words = tokenization_stats["splits"]["test"]["n_words"]
    final_metrics = {
        "split": "test",
        **compute_metrics(nll_sum, n_scored, n_words),
        "n_tokens": n_scored,
        "n_words": n_words,
        "context_length": context_length,
        "stride": stride,
    }
    logger.info("test: %s", final_metrics)
    return final_metrics
