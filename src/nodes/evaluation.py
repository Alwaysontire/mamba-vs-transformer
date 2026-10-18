"""Нода evaluate: прямой проход по test-сплиту и подсчёт метрик"""


from __future__ import annotations

import pandas as pd

from src.datasets.train_artifact import TrainArtifact


def evaluate_node(
        train_artifact: TrainArtifact,
        dataset_split: pd.DataFrame,
        training: dict,
        split: dict,
) -> dict:
    """
    Args:
        train_artifact (TrainArtifact): Обученная модель и manifest.
        dataset_split (pd.DataFrame): Датасет с колонкой сплита (train/val/test).
        training (dict): Параметры обучения (batch size, max_length и т.п.).
        split (dict): Параметры сплита; используется ключ ``split_col``.

    Returns:
        dict: Метрики из ``UNIFIED_METRICS`` (token/word-level perplexity).
    """
    raise NotImplementedError
