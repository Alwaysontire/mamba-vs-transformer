"""Текст сплита -> 1D поток токенов + статистика для word PPL / bpb"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class SplitStats:
    n_lines: int = 0
    n_tokens: int = 0
    n_words: int = 0
    n_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _lines_have_newlines(lines: Sequence[str]) -> bool:
    """В HF-версии WikiText непустые строки уже заканчиваются на \\n"""
    non_empty = [s for s in lines[:10_000] if s]
    return bool(non_empty) and sum(s.endswith("\n") for s in non_empty) / len(non_empty) > 0.5


def iter_chunks(lines: Sequence[str], batch_lines: int) -> Iterator[str]:
    """Склеивает строки в куски по batch_lines строк, не разрывая строку"""
    sep = "" if _lines_have_newlines(lines) else "\n"
    for i in range(0, len(lines), batch_lines):
        chunk = sep.join(lines[i : i + batch_lines])
        yield chunk + sep if sep else chunk


def tokenize_lines(lines: Sequence[str], tokenizer: Any, batch_lines: int = 1000) -> tuple[np.ndarray, SplitStats]:
    """Токенизирует строки сплита в исходном порядке

    Граница между кусками всегда на конце строки, поэтому результат отличается от
    токенизации всего текста целиком только слиянием переводов строк на стыках.
    """
    dtype = np.uint16 if len(tokenizer) <= np.iinfo(np.uint16).max else np.uint32
    stats = SplitStats(n_lines=len(lines))
    parts: list[np.ndarray] = []

    chunks = list(iter_chunks(lines, batch_lines))
    for i in range(0, len(chunks), 64):
        batch = chunks[i : i + 64]
        ids = tokenizer(batch, add_special_tokens=False, verbose=False)["input_ids"]
        for chunk, chunk_ids in zip(batch, ids):
            parts.append(np.asarray(chunk_ids, dtype=dtype))
            stats.n_words += len(chunk.split())
            stats.n_bytes += len(chunk.encode("utf-8"))

    tokens = np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
    stats.n_tokens = int(tokens.size)
    return tokens, stats
