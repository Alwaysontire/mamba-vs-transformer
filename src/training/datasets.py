"""Торчовый Dataset для моделей"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from torch.utils.data import Dataset




class LMDataset(Dataset):
    """Torch Dataset для моделей под задачу LM"""

    def __init__(
            self, 
            df: pd.DataFrame,
            ) -> None:
        pass

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Возвращает 1 пример объекта(текст)"""
        