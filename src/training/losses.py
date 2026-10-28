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
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Обычная кросс-энтропия по next-token
    """
    loss = F.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        ignore_index=int(ignore_index),
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    return loss.view_as(targets) if reduction == "none" else loss

@register_loss("focal")
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 1.0,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Регуляризация кросс-энтропии для дисбаланса (добавил на всякий случай)
    """
    ignore_index = int(ignore_index)
    nll = ce_loss(logits, targets, ignore_index=ignore_index, reduction="none")
    p_t = torch.exp(-nll)
    # clamp: при gamma < 1 производная (1 - p_t) ** gamma в p_t = 1 бесконечна
    loss = alpha * (1.0 - p_t).clamp_min(1e-12) ** gamma * nll

    mask = targets != ignore_index
    loss = loss * mask
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / mask.sum().clamp_min(1)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """Loss по ``cfg.type`` из реестра с параметрами ``cfg.params``"""
    if cfg.type not in LOSSES:
        raise KeyError(f"Неизвестный лосс {cfg.type!r}, есть: {sorted(LOSSES)}")
    params = {k: v for k, v in cfg.params.items() if v is not None}
    return LOSSES[cfg.type](logits, targets, **params)

