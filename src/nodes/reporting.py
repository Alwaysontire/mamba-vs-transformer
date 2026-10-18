"""Ноды отчётов: по одному запуску (build_report) и сравнение запусков (compare_report)"""


from __future__ import annotations

import logging
from pathlib import Path

from src.common.io import read_json
from src.reporting import EvaluationMetrics, RunResult, TrainManifest, build_report

logger = logging.getLogger(__name__)


def build_report_node(
        manifest: dict,
        final_metrics: dict,
        reporting: dict,
) -> str:
    """
    Args:
        manifest (dict): manifest обучения (формат :class:`TrainManifest`).
        final_metrics (dict): Метрики evaluation (формат :class:`EvaluationMetrics`).
        reporting (dict): ``params:reporting``; ключи ``run_name``, ``plotly``.

    Returns:
        str: HTML-отчёт по запуску.
    """
    run = RunResult(
        name=reporting["run_name"],
        manifest=TrainManifest.model_validate(manifest),
        metrics=EvaluationMetrics.model_validate(final_metrics),
    )
    return build_report([run], plotly=reporting.get("plotly", "cdn"))


def compare_report_node(reporting: dict) -> str:
    """Сравнение всех запусков в ``runs_dir``, у которых есть manifest и final_metrics

    Args:
        reporting (dict): ``params:reporting``; ключи ``runs_dir``, ``plotly``.

    Returns:
        str: HTML-отчёт со сравнением запусков.
    """
    runs_dir = Path(reporting["runs_dir"])
    runs: list[RunResult] = []
    # Сортировка по имени: цвет закреплён за позицией, так mamba/transformer не перекрашиваются
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []:
        manifest_path = run_dir / "metrics" / "manifest.json"
        metrics_path = run_dir / "metrics" / "final_metrics.json"
        if not (manifest_path.is_file() and metrics_path.is_file()):
            logger.info("Пропускаю %s: нет manifest.json или final_metrics.json", run_dir)
            continue
        runs.append(RunResult(
            name=run_dir.name,
            manifest=TrainManifest.model_validate(read_json(manifest_path)),
            metrics=EvaluationMetrics.model_validate(read_json(metrics_path)),
        ))

    if not runs:
        raise FileNotFoundError(
            f"В {runs_dir} нет завершённых запусков; сначала "
            "`uv run kedro run --env mamba` и `uv run kedro run --env transformer`"
        )
    logger.info("Сравниваю запуски: %s", [r.name for r in runs])
    return build_report(runs, plotly=reporting.get("plotly", "cdn"))
