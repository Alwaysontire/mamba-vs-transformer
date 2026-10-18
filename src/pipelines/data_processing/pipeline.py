"""data_processing: build

Изначально задумывалась еще и дедупликация и сплит по хешу. 
Но Wiki-text103 уже разбит по сплитам
"""

from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline

from src.nodes.data_processing import build_dataset_node

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
                )
        ]
    )