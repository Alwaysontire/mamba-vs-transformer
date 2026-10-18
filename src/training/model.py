"""Реестр моделей + сохраненние артефактов"""



from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch 
import torch.nn as nn

from src.config.schemas import LossConfig, ModelConfig, ObjectiveConfig
from src.training.losses import compute_loss

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
            self, model_cfg: ModelConfig, *, obkective: ObjectiveConfig
            ) -> None:
        super().__init__()
        pass
    def forward(self):
        pass




@register_architecture("transformer")
class Transformer(nn.Module):
    """Трансформер модель для задачи LM"""

    def __init__(
            self, model_cfg: ModelConfig, *, obkective: ObjectiveConfig
            ) -> None:
        super().__init__()
        pass
    def forward(self):
        pass