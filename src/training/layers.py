"""Реестр кастомных слоев для сборки модели"""


from __future__ import annotations
from collections.abc import Callable

import torch 
import torch.nn as nn

from src.config.schemas import ModelConfig, MambaBlockConfig, TransformerBlockConfig


LAYERS: dict[str, type[nn.Module]] = {}


def register_layer(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Декоратор регистрации слоев по имени name"""

    def deco(cls: type[nn.Module]) -> type[nn.Module]:
        if name in LAYERS:
            raise ValueError(f"Мы уже реализовали такой слой: {name!r}")
        LAYERS[name] = cls
        return cls
    return deco



@register_layer("mamba_block")
class MambaBlock(nn.Module):
    """Мамба блок"""
    def __init__(self, config: MambaBlockConfig):
        pass
    def forward(self):
        pass


@register_layer("transformer_block")
class MambaBlock(nn.Module):
    """Трансформер блок"""
    def __init__(self, config: TransformerBlockConfig):
        pass
    def forward(self):
        pass