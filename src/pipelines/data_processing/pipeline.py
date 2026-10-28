"""data_processing: build -> tokenize

Изначально задумывалась еще и дедупликация и сплит по хешу.
Но Wiki-text103 уже разбит по сплитам
"""

from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline

from src.nodes.data_processing import build_dataset_node, tokenize_node

def create_pipeline(**kwargs) -> Pipeline:
    """Собирает kedro-пайплайн подготовки данных"""

    return pipeline(
        [
            node(
                build_dataset_node,
                inputs="params:data",
                outputs=["dataset", "dataset_split"],
                name="build_dataset",
                tags=["data"],
                ),
            node(
                tokenize_node,
                inputs=["dataset_split", "params:tokenizer", "params:split"],
                outputs=["tokens_train", "tokens_validation", "tokens_test", "tokenization_stats"],
                name="tokenize",
                tags=["data"],
                ),
        ]
    )
