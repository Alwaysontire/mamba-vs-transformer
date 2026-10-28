"""Нода train: токены train/validation -> обученный TrainArtifact"""


from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from src.config.schemas import ModelConfig, TrainingConfig
from src.datasets.train_artifact import TrainArtifact


logger = logging.getLogger(__name__)


def train_node(
        tokens_train: np.ndarray,
        tokens_validation: np.ndarray,
        tokenization_stats: dict,
        model: dict,
        training: dict,
        tokenizer: dict,
        run: dict,
        paths: dict,
) -> tuple[TrainArtifact, dict]:
    """
    Args:
        tokens_train (np.ndarray): 1D поток токенов train (memmap).
        tokens_validation (np.ndarray): 1D поток токенов validation (memmap).
        tokenization_stats (dict): Статистика токенизации; ``vocab_size`` токенизатора.
        model (dict): Параметры модели (``params:model``) для :class:`ModelConfig`.
        training (dict): Параметры обучения для :class:`TrainingConfig`.
        tokenizer (dict): ``params:tokenizer``; ``name`` — сохранить вместе с моделью.
        run (dict): Параметры запуска; используется ``name`` для пути вывода.
        paths (dict): Пути артефактов; ключ ``train_dir`` (``"auto"`` или явный).

    Returns:
        tuple[TrainArtifact, dict]: Артефакт с обученной моделью и manifest'ом;
            manifest отдельно — для отчёта (формат: :class:`src.reporting.inputs.TrainManifest`).
    """
    from src.training.run import build_train_manifest, train_core

    model_cfg = TypeAdapter(ModelConfig).validate_python(model)
    if model_cfg.vocab_size < tokenization_stats["vocab_size"]:
        raise ValueError(
            f"model.vocab_size={model_cfg.vocab_size} меньше словаря токенизатора "
            f"{tokenization_stats['tokenizer']!r} ({tokenization_stats['vocab_size']})"
        )

    train_dir = paths.get("train_dir", "auto")
    base = f"runs/train/{run['name']}" if train_dir == "auto" else train_dir
    output_dir = Path(base) / "checkpoints"

    core = train_core(
        train_tokens=tokens_train,
        val_tokens=tokens_validation,
        model_cfg=model_cfg,
        training_cfg=TrainingConfig(**training),
        tokenizer_name=tokenizer["name"],
        output_dir=output_dir,
        log=logger,
    )

    manifest = build_train_manifest(core, model=model, training=training)
    return TrainArtifact(model=core["model"], manifest=manifest), manifest
