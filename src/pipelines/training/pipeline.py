"""Training: tokens_train + tokens_validation -> train_artifact"""


from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline

from src.nodes.training import train_node


def create_pipeline(**kwargs) -> Pipeline:
    """Собирает training-пайплайн"""

    return pipeline([
        node(
            train_node,
            inputs=[
                "tokens_train",
                "tokens_validation",
                "tokenization_stats",
                "params:model",
                "params:training",
                "params:tokenizer",
                "params:run",
                "params:paths",
            ],
            outputs=["train_artifact", "manifest"],
            name="train",
            tags=["train", "gpu"],
        )
    ])
