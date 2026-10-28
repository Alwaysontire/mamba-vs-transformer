"""Обучение моделей через HF Trainer"""


from __future__ import annotations
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
from pydantic import TypeAdapter

from src.common.device import resolve_device, resolve_precision
from src.common.hashing import stable_dict_hash
from src.common.io import now_iso
from src.config.schemas import ModelConfig, TrainingConfig
from src.training.datasets import LMDataset
from src.training.model import build_model


def train_core(
        *,
        train_tokens: np.ndarray,
        val_tokens: np.ndarray,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        tokenizer_name: str,
        output_dir: Path,
        log: Any,
) -> dict[str, Any]:
    """Прогоняет HF Trainer loop"""
    from transformers import EarlyStoppingCallback, TrainingArguments
    from src.training.callbacks import MLflowMetricsCallback
    from src.training.trainer import LMTrainer

    device = resolve_device(training_cfg.device)
    trainer_cfg = training_cfg.trainer

    model = build_model(model_cfg)
    train_ds = LMDataset(train_tokens, model_cfg.max_length)
    val_ds = LMDataset(val_tokens, model_cfg.max_length)
    n_params = count_params(model)
    log.info("device=%s, params=%s, train=%d окон, val=%d окон", device, n_params, len(train_ds), len(val_ds))

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=trainer_cfg.epochs,
        max_steps=trainer_cfg.max_steps,
        per_device_train_batch_size=trainer_cfg.batch_size,
        per_device_eval_batch_size=trainer_cfg.eval_batch_size or trainer_cfg.batch_size,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        max_grad_norm=trainer_cfg.max_grad_norm,
        bf16=resolve_precision(trainer_cfg.bf16, device, "bf16"),
        fp16=resolve_precision(trainer_cfg.fp16, device, "fp16"),
        use_cpu=device == "cpu",
        optim=training_cfg.optimizer.name,
        learning_rate=training_cfg.optimizer.lr,
        weight_decay=training_cfg.optimizer.weight_decay,
        adam_beta1=training_cfg.optimizer.adam_beta1,
        adam_beta2=training_cfg.optimizer.adam_beta2,
        lr_scheduler_type=training_cfg.scheduler.name,
        warmup_steps=training_cfg.scheduler.warmup_ratio,
        eval_strategy=trainer_cfg.eval_strategy,
        eval_steps=trainer_cfg.eval_steps,
        logging_strategy=trainer_cfg.logging_strategy,
        logging_steps=trainer_cfg.logging_steps,
        save_strategy=trainer_cfg.save_strategy,
        save_steps=trainer_cfg.save_steps,
        save_total_limit=trainer_cfg.save_total_limit,
        metric_for_best_model=trainer_cfg.metric_for_best_model,
        greater_is_better=trainer_cfg.greater_is_better,
        load_best_model_at_end=trainer_cfg.early_stopping_enabled,
        seed=trainer_cfg.seed,
        dataloader_num_workers=trainer_cfg.dataloader_num_workers,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to="none",
    )

    callbacks: list = [MLflowMetricsCallback()]
    if trainer_cfg.early_stopping_enabled:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=trainer_cfg.early_stopping_patience))

    trainer = LMTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
        loss_cfg=training_cfg.objective.loss,
    )
    started = time.perf_counter()
    train_result = trainer.train()
    train_time_sec = time.perf_counter() - started
    eval_metrics = trainer.evaluate()

    return {
        "model": trainer.model,
        "device": device,
        "tokenizer": tokenizer_name,
        "n_params": n_params,
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "history": _history(trainer.state.log_history),
        "train_time_sec": train_time_sec,
        "train_metrics": _scalars(train_result.metrics),
        "eval_metrics": _scalars(eval_metrics),
    }


def build_train_manifest(core: dict[str, Any], *, model: dict, training: dict) -> dict[str, Any]:
    """Манифест стейджа train в формате :class:`src.reporting.inputs.TrainManifest`"""
    model_dump = TypeAdapter(ModelConfig).validate_python(model).model_dump(mode="json")
    training_dump = TrainingConfig(**training).model_dump(mode="json")
    return {
        "stage": "train",
        "created_at": now_iso(),
        "config_hash": stable_dict_hash({"model": model_dump, "training": training_dump}),
        "model": model_dump,
        "training": training_dump,
        "tokenizer": core["tokenizer"],
        "device": core["device"],
        "n_params": core["n_params"],
        "train_tokens": core["train_tokens"],
        "val_tokens": core["val_tokens"],
        "history": core["history"],
        "train_time_sec": core["train_time_sec"],
        "train_metrics": core["train_metrics"],
        "eval_metrics": core["eval_metrics"],
    }


def count_params(model: nn.Module) -> dict[str, int]:
    """``total`` без дублей tied weights; ``non_embedding`` — без ``nn.Embedding``"""
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(
        p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()
    )
    return {"total": total, "non_embedding": total - embedding}


def _history(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``state.log_history`` -> точки ``HistoryPoint`` по шагам"""
    points: dict[int, dict[str, Any]] = {}
    for entry in log_history:
        if "loss" not in entry and "eval_loss" not in entry:
            continue
        point = points.setdefault(entry["step"], {"step": entry["step"]})
        point["epoch"] = entry.get("epoch")
        if "loss" in entry:
            point["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            point["eval_loss"] = entry["eval_loss"]
        if "learning_rate" in entry:
            point["learning_rate"] = entry["learning_rate"]
    return [points[step] for step in sorted(points)]


def _scalars(metrics: dict | None) -> dict[str, float]:
    if not metrics:
        return {}
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
