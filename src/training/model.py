"""Реестр моделей + сохраненние артефактов"""



from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch 
import torch.nn as nn

from src.config.schemas import LossConfig, ModelConfig, TransformerConfig, ObjectiveConfig
from src.training.losses import compute_loss
from src.training.layers import RMSnorm, TransformerBlock

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
            self, model_cfg: ModelConfig, *, objective: ObjectiveConfig
            ) -> None:
        super().__init__()
        pass
    def forward(self):
        pass




@register_architecture("transformer")
class Transformer(nn.Module):
    """Трансформер модель для задачи LM"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
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
        if self.config.pos_encoding == "learned":
            positions = torch.arange(sequence_length, device=input_ids.device)
            x = x + self.positional_emb(positions).unsqueeze(0)

        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.linear(x)

        return logits