"""Pydantic модели конфига пайплайна"""


from __future__ import annotations
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

SplitName = Literal["train", "validation", "test"]

def _default_split_patterns() -> dict[str, str]:
    return {"train": "train*", "validation": "validation*", "test": "test*"}


class DataSourceConfig(_StrictModel):
    """Описание одного источника данных: директория с файлами сплитов (parquet / csv)

    ``splits`` — glob-шаблон относительно ``path`` для каждого сплита. Шаблон может
    указывать как на файлы (``train-00000-of-00002.parquet``), так и на поддиректорию
    (``train/``) — тогда берутся все файлы нужного формата внутри неё.
    """

    name: str
    kind: Literal["parquet", "csv"]
    path: str | None = None
    splits: dict[SplitName, str] = Field(default_factory=_default_split_patterns)
    column_map: dict[str, str] = Field(default_factory=lambda: {"text": "text"})

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

class LossConfig(_StrictModel):
    """Вид функции потерь"""
    type: str = "ce"
    params: dict[str, float | None] = Field(default_factory=dict)


class ParamBudget(_StrictModel):
    """Целевое значение параметров для честного сравнения"""
    target: int = 30000000                  # 30 миллионов пока что
    tolerance: float = 0.05                 # Допустим немного отклонений
    include_emb: bool = False               # Сравниваем ли вместе с nn.Embeddings 


class _LMBase(_StrictModel):
    """Общие параметры для обеих моделей"""

    vocab_size: int
    d_model: int = Field(
        description=(
            "Ширина residual-потока: размер вектора токена между блоками. "
            "Эмбеддинг: vocab_size -> d_model; каждый блок: [B, L, d_model] -> [B, L, d_model]; "
            "LM-head: d_model -> vocab_size"
        ),
    )
    n_layers: int = Field(description="Число блоков в residual-потоке")
    max_length: int = Field(1024, description="Длина контекста в токенах")
    dropout: float = 0.0
    tie_embeddings: bool = Field(True, description="LM-head использует веса эмбеддинга")

class TransformerBlockConfig(_StrictModel):
    """Параметры блока трансформера (attention + MLP); вход и выход блока — d_model модели"""

    n_heads: int = Field(description="Число голов attention; размер головы = d_model / n_heads")
    d_ff: int | None = Field(
        None, description="Скрытая ширина MLP; None -> 4 * d_model (для swiglu 8/3 * d_model)"
    )
    norm: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    activation: Literal["gelu", "swiglu", "relu"] = "swiglu"
    attn_implementation: Literal["eager", "sdpa"] = "sdpa"


class TransformerConfig(_LMBase):
    """Параметры трансформера"""

    arch: Literal["transformer"]
    pos_encoding: Literal["rope", "learned", "alibi"] = "rope"
    block: TransformerBlockConfig

    @model_validator(mode="after")
    def _check_heads(self) -> "TransformerConfig":
        if self.d_model % self.block.n_heads:
            raise ValueError(f"ЮРИЙ ПРОВЕРЬ d_model={self.d_model} не делится на n_heads={self.block.n_heads}")
        return self

class MambaBlockConfig(_StrictModel):
    """Параметры Mamba2-блока; вход и выход блока — d_model модели"""

    d_state: int = Field(64, description="Размер скрытого состояния SSM на голову")
    d_conv: int = Field(4, description="Ширина causal conv1d перед SSM")
    expand: int = Field(2, description="d_inner = expand * d_model — ширина внутри блока")
    headdim: int = Field(64, description="Размер головы SSM; n_heads = d_inner / headdim")
    ngroups: int = Field(1, description="Число групп B/C (как GQA у attention)")
    chunk_size: int = Field(256, description="Размер чанка в ядре mamba_chunk_scan_combined")
    dt_min: float = 0.001
    dt_max: float = 0.1
    conv_bias: bool = True
    bias: bool = False


class MambaConfig(_LMBase):
    """Параметры Mamba2"""

    arch: Literal["mamba"]
    block: MambaBlockConfig = Field(default_factory=MambaBlockConfig)

    @model_validator(mode="after")
    def _check_heads(self) -> "MambaConfig":
        d_inner = self.block.expand * self.d_model
        if d_inner % self.block.headdim:
            raise ValueError(f"d_inner={d_inner} (expand * d_model) не делится на headdim={self.block.headdim}")
        if (d_inner // self.block.headdim) % self.block.ngroups:
            raise ValueError(f"n_heads={d_inner // self.block.headdim} не делится на ngroups={self.block.ngroups}")
        return self

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
    sources: list[DataSourceConfig]
    training: TrainingConfig
    model: ModelConfig
    param_budget: ParamBudget

    @model_validator(mode="after")
    def _check_splits_covered(self) -> ExperimentConfig:
        """Проверить, что у каждого официального сплита есть хотя бы один источник"""

        missing = set(get_args(SplitName)) - {split for source in self.sources for split in source.splits}
        if missing:
            raise ValueError(f"Нет источников для сплитов: {sorted(missing)}")
        return self
