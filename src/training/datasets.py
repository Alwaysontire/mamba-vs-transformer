"""Торчовый Dataset для моделей"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset




class LMDataset(Dataset):
    """Torch Dataset для моделей под задачу LM"""

    def __init__(
            self, 
            tokens: np.array,
            max_length: int
            ) -> None:
        self.tokens = tokens
        self.max_length = max_length

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.max_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Возвращает 1 пример объекта(текст)"""
        start = idx * self.max_length
        chunk = torch.from_numpy(np.asarray(self.tokens[start: start + self.max_length + 1], dtype=np.int64))
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}
        