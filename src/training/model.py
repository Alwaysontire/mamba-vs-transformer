"""Реестр моделей + сохраненние артефактов"""



from __future__ import annotations
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch 
import torch.nn as nn

from src.config.schemas import LossConfig, ModelConfig, TransformerConfig, ObjectiveConfig, MambaConfig
from src.training.losses import compute_loss
from src.training.layers import RMSnorm, TransformerBlock, MambaBlock

ARCHITECTURES: dict[str, type[nn.Module]] = {}


def register_architecture(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Декоратор для архитектур регистирует по имени name"""

    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        if name in ARCHITECTURES:
            raise ValueError(f"Да мы уже такое реализовали: {name!r}")
        ARCHITECTURES[name] = cls
        return cls
    return deco



@register_architecture("mamba")
class Mamba(nn.Module):
    """Мамба модель для задачи LM"""

    def __init__(
            self, config: ModelConfig
            ) -> None:
        super().__init__()
        self.model_cfg = config
        self.max_length = config.max_length

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            MambaBlock(config.block, config.d_model)
         for _ in range(config.n_layers)])

        self.final_norm = RMSnorm(config.d_model)

        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.linear.weight = self.token_emb.weight

        init_lm_weights(self, config.n_layers)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb_dropout(self.token_emb(input_ids))

        for block in self.blocks:
            x = block(x)
        return self.linear(self.final_norm(x))




@register_architecture("transformer")
class Transformer(nn.Module):
    """Трансформер модель для задачи LM"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.model_cfg = config
        self.max_length = config.max_length

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        if config.pos_encoding == "learned":
            self.positional_emb = nn.Embedding(config.max_length, config.d_model)

        block_config = config.block
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    d_ff=block_config.d_ff,
                    norm=block_config.norm,
                    dropout=config.dropout,
                    n_heads=block_config.n_heads,
                    pos_encoding=config.pos_encoding,
                    activation=block_config.activation,
                    attn_implementation=block_config.attn_implementation,
                )
                for _ in range(config.n_layers)
            ]
        )

        if block_config.norm == "rmsnorm":
            self.final_norm = RMSnorm(config.d_model)
        else:
            self.final_norm = nn.LayerNorm(config.d_model)

        self.linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.linear.weight = self.token_emb.weight

        init_lm_weights(self, config.n_layers)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Получает [batch, sequence]
        Возвращает logits [batch, sequence, vocab_size]
        """
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence_length={sequence_length}, max_length={self.max_length}"
            )

        x = self.token_emb(input_ids)
        if self.model_cfg.pos_encoding == "learned":
            positions = torch.arange(sequence_length, device=input_ids.device)
            x = x + self.positional_emb(positions).unsqueeze(0)

        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.linear(x)

        return logits


def init_lm_weights(model: nn.Module, n_layers: int, std: float = 0.02) -> None:
    """GPT-2: N(0, std) для Linear/Embedding, нулевые bias, ``output_projection`` — std / sqrt(2 * n_layers)"""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, param in model.named_parameters():
        if name.endswith("output_projection.weight"):
            nn.init.normal_(param, mean=0.0, std=std / math.sqrt(2 * n_layers))


def build_model(model_cfg: ModelConfig) -> nn.Module:
    """Модель по ``model_cfg.arch`` из реестра"""
    if model_cfg.arch not in ARCHITECTURES:
        raise KeyError(f"Неизвестная архитектура {model_cfg.arch!r}, есть: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[model_cfg.arch](model_cfg)


def save_model(model: nn.Module, path: Path) -> None:
    """``config.json`` + ``model.pt`` (state_dict)"""
    from src.common.io import write_json

    write_json(Path(path) / "config.json", model.model_cfg.model_dump(mode="json"))
    torch.save(model.state_dict(), Path(path) / "model.pt")


def load_model(path: Path) -> nn.Module:
    """Обратная к :func:`save_model`: модель в режиме eval"""
    from pydantic import TypeAdapter
    from src.common.io import read_json

    model_cfg = TypeAdapter(ModelConfig).validate_python(read_json(Path(path) / "config.json"))
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=True))
    return model.eval()