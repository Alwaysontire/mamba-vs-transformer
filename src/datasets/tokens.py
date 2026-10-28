"""Kedro-датасет для потока токенов сплита (1D .npy)"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from kedro.io import AbstractDataset

from src.common.paths import atomic_write


class TokensDataset(AbstractDataset[np.ndarray, np.ndarray]):
    """1D массив id токенов; load отдаёт memmap, чтобы не держать сплит в RAM"""

    def __init__(self, filepath: str, mmap: bool = True) -> None:
        self._path = Path(filepath)
        self._mmap = mmap

    def load(self) -> np.ndarray:
        return np.load(self._path, mmap_mode="r" if self._mmap else None)

    def save(self, data: np.ndarray) -> None:
        if data.ndim != 1:
            raise ValueError(f"Ожидается 1D массив токенов, shape={data.shape}")
        with atomic_write(self._path, mode="wb") as f:
            np.save(f, data)

    def _exists(self) -> bool:
        return self._path.is_file()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._path), "mmap": self._mmap}
