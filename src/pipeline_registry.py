"""Реестр  kedro-пайплайнов проекта

data_processing -> training -> evaluation (evaluate + build_report)   # на каждый env
compare: отчёт со сравнением всех запусков                           # после обоих env'ов

"""


from __future__ import annotations
from kedro.pipeline import Pipeline

from src.pipelines.data_processing import create_pipeline as data_processing_pipeline
from src.pipelines.training import create_pipeline as training_pipeline
from src.pipelines.evaluation import create_pipeline as evaluation_pipeline
from src.pipelines.compare import create_pipeline as compare_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Собирает и регистрирует все kedro-пайплайны проекта"""

    data_processing = data_processing_pipeline()
    training = training_pipeline()
    evaluation = evaluation_pipeline()

    full = data_processing + training + evaluation

    return {
        "data_processing": data_processing,
        "training": training,
        "evaluation": evaluation,
        "compare": compare_pipeline(),
        "full": full,
        "__default__": full,
    }
