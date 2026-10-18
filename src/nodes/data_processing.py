"""Нода build_dataset: директории источников (train/validation/test) -> dataset, dataset_split"""


from __future__ import annotations

import logging

import pandas as pd

from src.config.schemas import DataSourceConfig
from src.data.contract import CONTRACT_COLUMNS, load_source

logger = logging.getLogger(__name__)


def build_dataset_node(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        data (dict): Источники из ``conf/base/source.yml`` (``params:data``),
            ключ — имя источника, значение — поля :class:`DataSourceConfig`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            ``dataset`` — объединение всех файлов (``name``, ``text``, ``source_file``);
            ``dataset_split`` — то же объединение с колонкой ``split`` (``name``, ``split``, ``text``).
    """
    sources = [DataSourceConfig(name=name, **cfg) for name, cfg in data.items()]
    merged = pd.concat([load_source(s) for s in sources], ignore_index=True)

    logger.info("Строк по сплитам: %s", merged["split"].value_counts().to_dict())

    dataset = merged[["name", "text", "source_file"]]
    dataset_split = merged[list(CONTRACT_COLUMNS)]
    return dataset, dataset_split
