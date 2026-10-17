"""Parquet датасет"""

from __future__ import annotations
from typing import Any
from kedro_datasets.pandas import ParquetDataset

class TextParquetDataset(ParquetDataset):
    """pandas.ParquetDataset"""

    def save(self, data: Any) -> None:
        """Пишет parquet базовым датасетом"""
        super().save(data)