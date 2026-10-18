"""Training: dataset_split -> train_artifact"""


from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline

from src.nodes.training import train_node


def create_pipeline(**kwargs) -> Pipeline:
    """Собирает training-пайплайн"""

    return pipeline([
        node(
            train_node,
            inputs=[
                "dataset_split",
                "params:model",
                "params:training",
                "params:split",
                "params:run",
                "params:mlflow",
                "params:paths",
            ],
            outputs="train_artifact",
            name="train",
            tags=["train", "gpu"],
        )
    ])
