from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.paths import atomic_write


def now_iso() -> str:
    """Возвращает текущий UTC-таймстамп в формате ISO-8601 для manifest.json.

    Returns:
        str: Таймстамп ISO-8601 в зоне UTC с точностью до секунд.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    """Читает JSON-файл в кодировке UTF-8.

    Args:
        path (Path): Путь к JSON-файлу.

    Returns:
        Any: Десериализованное содержимое файла.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = False) -> None:
    """Атомарно записывает данные в JSON-файл (UTF-8, с финальным переводом строки).

    Используется :func:`atomic_write`, поэтому частично записанный файл при сбое
    не остаётся. Кириллица сохраняется как есть (``ensure_ascii=False``), а
    несериализуемые объекты приводятся к строке (``default=str``).

    Args:
        path (Path): Путь к выходному файлу.
        payload (Any): Сериализуемый объект для записи.
        indent (int): Размер отступа для форматирования JSON.
        sort_keys (bool): Сортировать ли ключи объектов по алфавиту.
    """
    with atomic_write(path, mode="w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys, default=str)
        f.write("\n")