"""Collator: токенизирует сразу весь батч"""


from __future__ import annotations
from typing import Any

import numpy as np
import torch


class LMCollator:
    """Collate класс: токенизирует весь батч сразу"""


    def __init__(
            self, 
            tokenizer: Any,
            max_length: int,
            split_special_tokens: bool = True
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_special_tokens = split_special_tokens


    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Токенизирует тексты батча и собирает тензор для модели"""
        pass
        