"""Канонические колонки датасета и нормализация"""


from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy
import pandas as pd

from src.common.hashing import normalize_text_for_group_key, sha256_of_text, stable_dict_hash

from src.config.schemas import DataSourceConfig

logger = logging.getLogger(__name__)



CONTRACT_COLUMNS: tuple[str, ...] = (
    "name",
    "split"
    "text",
)
