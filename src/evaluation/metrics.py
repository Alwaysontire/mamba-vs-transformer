"""Метрики оценки"""


from __future__ import annotations

import numpy as np
import pandas as pd

UNIFIED_METRICS: tuple[str, ...] = (
    "token_level_perplexity", 
    "word_level_perplexity",
)


def compute_metrics():
    pass