"""Evaluation: скользящее окно по 1D потоку токенов -> суммарный NLL"""


from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from itertools import islice

import numpy as np
import torch
import torch.nn as nn

from src.training.losses import ce_loss

IGNORE_INDEX = -100


def resolve_device(device: str) -> torch.device:
    """``auto`` -> cuda, mps или cpu"""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_windows(n_tokens: int, context_length: int, stride: int) -> Iterator[tuple[int, int, int]]:
    """Окна ``(begin, end, n_skip)``: вход ``tokens[begin:end]``, таргеты ``tokens[begin + 1:end + 1]``
    """
    last = n_tokens - 1
    scored_until = 0
    begin = 0
    while scored_until < last:
        end = min(begin + context_length, last)
        yield begin, end, scored_until - begin
        scored_until = end
        begin += stride


def _make_batch(tokens: np.ndarray, windows: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(end - begin for begin, end, _ in windows)
    inputs = np.zeros((len(windows), width), dtype=np.int64)
    targets = np.full((len(windows), width), IGNORE_INDEX, dtype=np.int64)
    for i, (begin, end, n_skip) in enumerate(windows):
        n = end - begin
        inputs[i, :n] = tokens[begin:end]
        targets[i, n_skip:n] = tokens[begin + 1 + n_skip : end + 1]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


@torch.inference_mode()
def sliding_window_nll(
        model: nn.Module,
        tokens: np.ndarray,
        *,
        context_length: int,
        stride: int,
        batch_size: int,
        device: torch.device,
        autocast_dtype: torch.dtype | None = None,
) -> tuple[float, int]:
    """
    Args:
        model (nn.Module): ``forward(input_ids [B, L]) -> logits [B, L, vocab]``.
        tokens (np.ndarray): 1D поток id токенов.
        context_length (int): Длина окна.
        stride (int): Шаг окна, ``<= context_length``.
        batch_size (int): Окон в батче.
        device (torch.device): Устройство прогона.
        autocast_dtype (torch.dtype | None): dtype autocast; None — без autocast.

    Returns:
        tuple: суммарный NLL  и число засчитанных токенов.
    """
    model.eval().to(device)
    autocast = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else nullcontext()

    nll_sum = 0.0
    n_scored = 0
    windows = iter_windows(len(tokens), context_length, stride)
    while batch := list(islice(windows, batch_size)):
        inputs, targets = _make_batch(tokens, batch)
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast:
            logits = model(inputs)
        nll_sum += ce_loss(logits, targets, ignore_index=IGNORE_INDEX, reduction="sum").item()
        n_scored += int((targets != IGNORE_INDEX).sum())
    return nll_sum, n_scored
