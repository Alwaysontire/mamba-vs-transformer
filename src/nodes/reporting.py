"""Нода build_report собирает HTML-отчёт по обучению из артефактов"""


from __future__ import annotations

from src.datasets.train_artifact import TrainArtifact


def build_report_node(
        train_artifact: TrainArtifact,
        metrics: dict,
) -> str:
    """
    Args:
        train_artifact (TrainArtifact): Обученная модель и manifest.
        metrics (dict): Метрики из ноды ``evaluate``.

    Returns:
        str: HTML-отчёт.
    """
    raise NotImplementedError
