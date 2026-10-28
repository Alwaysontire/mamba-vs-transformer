"""Нода evaluate: прямой проход по test-сплиту и подсчёт метрик"""


from __future__ import annotations

import numpy as np

from src.datasets.train_artifact import TrainArtifact


def evaluate_node(
        train_artifact: TrainArtifact,
        tokens_test: np.ndarray,
        tokenization_stats: dict,
        training: dict,
) -> dict:
    """
    Args:
        train_artifact (TrainArtifact): Обученная модель и manifest.
        tokens_test (np.ndarray): 1D поток токенов test (memmap).
        tokenization_stats (dict): ``splits.test.{n_tokens, n_words, n_bytes}`` —
            знаменатели для word PPL и bits per byte.
        training (dict): Параметры обучения (batch size, max_length и т.п.).

    Returns:
        dict: ``final_metrics`` в формате :class:`src.reporting.inputs.EvaluationMetrics`
            (обязательно только ``token_level_perplexity``).
    """
    raise NotImplementedError
