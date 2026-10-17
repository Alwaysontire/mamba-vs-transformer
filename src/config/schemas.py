"""Pydantic модели конфига пайплайна"""


from __future__ import annotations
import os
from typing import Any, Literal, Annotated, get_args
from pydantic import BaseModel, Field, ConfigDict, model_validator



class _StrictModel(BaseModel):
    """Базовая модель со строгой валидацией"""
    model_config = ConfigDict(extra="forbid")


class RunConfig(_StrictModel):
    name: str


class PathsConfig(_StrictModel):
    """Локальные пути эксперимента"""

    processed_dir: str = "data/processed"
    train_dir: str = "data/train"
    predictions_dir: str = "data/predictions"
    metrics_dir: str = "data/metrics"

class MLflowConfig(_StrictModel):
    """Настройки логгирование эксперимента на MLflow"""

    enabled: bool = True
    experiment_name: str
    tracking_url: str = "sqlite:///mlflow.db"
    artifact_mode: Literal["minimal", "full"] = "minimal"

    @model_validator(mode="after")
    def __apply_env_override(self) -> MLflowConfig:
        """Переопределить trackin_url из переменной окружения"""

        env_url = os.environ.get("MLFLOW_TRACKING_URL")
        if env_url:
            object.__setattr__(self, "tracking_url", env_url)
        return self

SplitName = Literal["train", "validation", "test"]

class DataSourceConfig(_StrictModel):
    """Описание одного источника данных (parquet / csv)"""

    name: str
    kind: Literal["parquet", "csv"]
    split: SplitName
    path: str | None = None
    columns: list[str] | None = None

class AugmentConfig(_StrictModel):
    """Настройка аугментации текстовых данных"""

    enabled: bool = False
    seed: int = 42

class HeadConfig(_StrictModel):
    """Конфигурация последней классификационной головы для предсказания next_token"""

    hidden_size: list[int] = Field(default_factory=list)
    activation: Literal["gelu", "swiglu", "relu", "silu"] 
    input_dropout: float = 0.0
    dropout: float = 0.0

class ParamBudget(_StrictModel):
    """Целевое значение параметров для честного сравнения"""
    target: int = 30000000                  # 30 миллионов пока что
    tolerance: float = 0.05                 # Допустим немного отклонений
    include_emb: bool = False               # Сравниваем ли вместе с nn.Embeddings 


class _LMBase(_StrictModel):
    """Общие параметры для обеих моделей"""

    vocab_size: int
    d_model: int
    n_layers: int
    max_length: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True

class TransformerConfig(_LMBase):
    """Параметры трансформера"""

    arch: Literal["transformer"]
    n_heads: int
    d_ff: int | None = None
    pos_encoding: Literal["rope", "learned", "alibi"] = "rope"
    norm: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    activation: Literal["gelu", "swiglu", "relu"] = "swiglu"
    attn_implementation: Literal["eager", "sdpa"] = "sdpa"

    @model_validator(mode="after")
    def _check_heads(self) -> "TransformerConfig":
        if self.d_model % self.n_heads: 
            raise ValueError(f"ЮРИЙ ПРОВЕРЬ d_model={self.d_model} не делится на n_heads={self.n_heads}")
        return self

class MambaConfig(_LMBase):
    arch: Literal["mamba"]
    version: Literal[1, 2] = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    headdim: int | None = None



ModelConfig = Annotated[TransformerConfig | MambaConfig, Field(discriminator="arch")]


class TrainingConfig(_StrictModel):
    """Общие параметры для обеих архитектур"""

    batch_size: int
    lr: float
    weight_decay: float = 0.1
    warmup_steps: int = 0
    max_steps: int 
    grad_clip: float = 1.0
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    seed: int = 42

class ExperimentConfig(_StrictModel):
    """Корневой конфиг одного запуска"""

    run: RunConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    mlflow: MLflowConfig
    sources: list[DataSourceConfig]
    training: TrainingConfig
    model: ModelConfig
    param_budget: ParamBudget

    @model_validator(mode="after")
    def _check_splits_covered(self) -> ExperimentConfig:
        """Проверить, что у каждого официального сплита есть хотя бы один источник"""

        missing = set(get_args(SplitName)) -{source.split for source in self.sources}
        if missing:
            raise ValueError(f"Нет источников для сплитов: {sorted(missing)}")
        return self
