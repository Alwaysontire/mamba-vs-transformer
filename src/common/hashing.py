"""Хешировани текстов для дедупликации"""

from __future__ import annotations
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20

def sha256_of_file(path: Path) -> str:
    """Посчитать SHA256 файла потоково по чанкам"""

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()

def sha256_of_text(text: str) -> str:
    """Посчитать SHA256 строки в кодировке utf-8"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalize_text_for_group_key(text: str) -> str:
    """Канонизировать текст для group_key: NFKC + lower + схлопывание пробелов"""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    return " ".join(text.split())

def stable_dict_hash(obj: Any) -> str:
    """Посчитать SHA156 от json dict/list/scalar"""

    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_of_text(payload)
