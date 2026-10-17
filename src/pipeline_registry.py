"""Реестр  kedro-пайплайнов проекта

build -> train -> score -> evaluate -> build_report

"""


from __future__ import annotations
from kedro.pipeline import Pipeline

from src.pipelines.training import create_pipeline as training_pipeline
from src.pipeline.inference import create_pipeline as inference_pipeline
from src.pipelines.data_processing import create_pipeline as data_processing_pipeline
from src.pipeline.evaluation import create_pipeline as evaluation_pipeline
from src.pipeline.build_report import create_pipeline as build_report_pipeline

def register_pipeline() -> dict[str, Pipeline]:
    """Собирает и регистирирует все kedro-пайплайну проекта"""

    data_processing = data_processing_pipeline()
    training = training_pipeline()
    inference = inference_pipeline()
    evaluation = evaluation_pipeline()
    build_report = build_report_pipeline()

    full = data_processing + training + inference + evaluation + build_report

    return {
        "data_processing": data_processing,
        "training": training,
        "inference": inference,
        "evaluation": evaluation,
        "build_report": build_report,
        "full": full,
        "__default__": full,
    }
