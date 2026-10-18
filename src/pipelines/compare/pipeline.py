"""compare: все завершённые запуски из runs_dir -> один HTML-отчёт со сравнением

Запускается после обоих env'ов, без --env: uv run kedro run --pipeline compare
"""


from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline

from src.nodes.reporting import compare_report_node


def create_pipeline(**kwargs) -> Pipeline:
    """Собирает compare-пайплайн"""

    return pipeline([
        node(
            compare_report_node,
            inputs="params:reporting",
            outputs="comparison_report",
            name="compare_report",
            tags=["report"],
        ),
    ])
