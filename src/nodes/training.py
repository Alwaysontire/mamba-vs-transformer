"""Нода train: берет dataset_split -> обученный TrainArtifact"""


from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from src.config.schemas import MLflowConfig, ModelConfig, TrainingConfig
from src.datasets.train_artifact import TrainArtifact


logger = logging.getLogger(__name__)


def train_node(
        dataset_split: pd.DataFrame,
        model: dict,
        training: dict,
        split: dict,
        run: dict,
        mlflow: dict,
        paths: dict,
) -> TrainArtifact:
    """    
    Args:
        dataset_split (pd.DataFrame): Датасет с колонкой сплита (train/val/test).
        model (dict): Параметры модели (``params:model``) для :class:`ModelConfig`.
        training (dict): Параметры обучения для :class:`TrainingConfig`.
        label_schema (dict): Параметры схемы меток (``params:label_schema``).
        split (dict): Параметры сплита; используется ключ ``split_col``.
        run (dict): Параметры запуска; используется ``name`` для пути вывода.
        mlflow (dict): Параметры MLflow для :class:`MLflowConfig`.
        paths (dict): Пути артефактов; ключ ``train_dir`` (``"auto"`` или явный).

    Returns:
        TrainArtifact: Артефакт с обученной моделью, схемой меток и manifest'ом.
    """
    from src.training.run import build_train_manifest, train_core
    train_dir = paths.get("train_dir", "auto")
    base = f"runs/train/{run['name']}" if train_dir == "auto" else train_dir
    output_dir = Path(base) / "checkpoints"

    core = train_core(
        df=dataset_split,
        model_cfg=ModelConfig(**model),
        training_cfg=TrainingConfig(**training),
        mlflow_cfg=MLflowConfig(**mlflow),
        split_col=split["split_col"],
        output_dir=output_dir,
        log=logger,
    )

    manifest = build_train_manifest(core, model=model, training=training)
    return TrainArtifact(model=core["model"], manifest=manifest)