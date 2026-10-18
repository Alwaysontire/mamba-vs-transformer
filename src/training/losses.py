"""Реестр функций потерь (может понадобиться например focal Loss)"""




from __future__ import annotations
from collections.abc import Callable
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F

from src.config.schemas import LossConfig

LossFn = Callable[..., torch.Tensor]
LOSSES: dict[str, LossFn] = {}




def register_loss(name: str) -> Callable[[LossFn], LossFn]:
    """Декоратор для регистрации функций потерь под именем name"""

    def deco(fn: LossFn) -> LossFn:
        if name in LOSSES:
            raise ValueError(f"Лосс уже зареган: {name!r}")
        LOSSES[name] = fn
        return fn
    
    return deco



@register_loss("ce")
def ce_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor,

) -> torch.Tensor:
    """Обычная кросс-энтропия"""
    pass

@register_loss("focal")
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    
) -> torch.Tensor:
    """Регуляризация кросс-энтропии для дисбаланса (добавил на всякий случай)"""
    pass

