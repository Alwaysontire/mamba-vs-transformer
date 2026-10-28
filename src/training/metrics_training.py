"""Compute метрики для валидационного цикла HF Trainer"""

from __future__ import annotations
from collections.abc import Callable
from typing import Any

import numpy as np



TRAINER_EVAL_METRICS: tuple[str, ...] = (
    ""
)