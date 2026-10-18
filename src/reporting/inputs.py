"""Контракт входа отчёта: что training (manifest) и evaluation (final_metrics) отдают reporting
Все поля, кроме ``token_level_perplexity``, опциональны 
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class PositionLoss(_Open):
    """Средний loss токенов на позиции ``position`` внутри контекстного окна"""

    position: int
    loss: float


class Throughput(_Open):
    """Скорость и память на прямом проходе evaluation"""

    tokens_per_sec: float | None = None
    peak_memory_mb: float | None = None


class EvaluationMetrics(_Open):
    """``final_metrics.json`` — выход ноды evaluate

    Пример::

        {
          "split": "test",
          "token_level_perplexity": 24.1,
          "word_level_perplexity": 31.5,
          "loss": 3.18,
          "n_tokens": 280000, "n_words": 245569,
          "context_length": 1024, "stride": 512,
          "throughput": {"tokens_per_sec": 51000.0, "peak_memory_mb": 2100.0},
          "loss_by_position": [{"position": 0, "loss": 7.1}, ...]
        }
    """

    split: str = "test"
    token_level_perplexity: float
    word_level_perplexity: float | None = None
    loss: float | None = None                      
    n_tokens: int | None = None
    n_words: int | None = None
    context_length: int | None = None
    stride: int | None = None
    throughput: Throughput | None = None
    loss_by_position: list[PositionLoss] | None = None


class HistoryPoint(_Open):
    """Точка истории обучения (как в ``Trainer.state.log_history``)"""

    step: int
    epoch: float | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None


class ParamCounts(_Open):
    total: int
    non_embedding: int | None = None


class TrainManifest(_Open):
    """``manifest.json`` — выход ноды train (``build_train_manifest``)"""

    run_name: str | None = None
    model: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    n_params: ParamCounts | None = None
    history: list[HistoryPoint] = Field(default_factory=list)
    train_time_sec: float | None = None
    created_at: str | None = None


class RunResult(BaseModel):
    """Один запуск (env) в отчёте"""

    name: str
    manifest: TrainManifest
    metrics: EvaluationMetrics

    @property
    def arch(self) -> str:
        return str(self.manifest.model.get("arch", "—"))
