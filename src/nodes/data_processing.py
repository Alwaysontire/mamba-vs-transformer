"""Ноды data_processing: источники -> dataset, dataset_split -> токены по сплитам"""


from __future__ import annotations

import logging
from typing import get_args

import numpy as np
import pandas as pd

from src.config.schemas import DataSourceConfig, SplitName
from src.data.contract import CONTRACT_COLUMNS, load_source
from src.data.tokenize import load_tokenizer, tokenize_lines

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


def tokenize_node(
        dataset_split: pd.DataFrame,
        tokenizer: dict,
        split: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Args:
        dataset_split (pd.DataFrame): Выход ``build_dataset``.
        tokenizer (dict): ``params:tokenizer``; ключи ``name``, ``batch_lines``.
        split (dict): ``params:split``; ключ ``split_col``.

    Returns:
        tuple: ``tokens_train``, ``tokens_validation``, ``tokens_test`` — 1D потоки id;
            ``tokenization_stats`` — ``{tokenizer, vocab_size, splits: {split: SplitStats}}``.
    """
    tok = load_tokenizer(tokenizer["name"])
    split_col = split["split_col"]

    tokens: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, int]] = {}
    for name in get_args(SplitName):
        lines = dataset_split.loc[dataset_split[split_col] == name, "text"].tolist()
        tokens[name], split_stats = tokenize_lines(lines, tok, batch_lines=tokenizer.get("batch_lines", 1000))
        stats[name] = split_stats.to_dict()
        logger.info("%s: %s", name, stats[name])

    tokenization_stats = {"tokenizer": tokenizer["name"], "vocab_size": len(tok), "splits": stats}
    return tokens["train"], tokens["validation"], tokens["test"], tokenization_stats
