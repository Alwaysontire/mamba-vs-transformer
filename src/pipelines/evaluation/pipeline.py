"""evaluation: train_artifact + dataset_split[test] -> final_metrics -> evaluation_report

Отдельного inference нет: perplexity считается из loss прямо в прямом проходе,
промежуточные предсказания (логиты) сохранять незачем.
"""


from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline

from src.nodes.evaluation import evaluate_node
from src.nodes.reporting import build_report_node


def create_pipeline(**kwargs) -> Pipeline:
    """Собирает evaluation-пайплайн"""

    return pipeline([
        node(
            evaluate_node,
            inputs=[
                "train_artifact",
                "dataset_split",
                "params:training",
                "params:split",
            ],
            outputs="final_metrics",
            name="evaluate",
            tags=["evaluation", "gpu"],
        ),
        node(
            build_report_node,
            inputs=[
                "manifest",
                "final_metrics",
                "params:reporting",
            ],
            outputs="evaluation_report",
            name="build_report",
            tags=["evaluation", "report"],
        ),
    ])
