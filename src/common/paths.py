from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Создаёт каталог (и все родительские), если он ещё не существует"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def atomic_write(path: Path, *, mode: str = "wb", encoding: str | None = None) -> Iterator[object]:
    """Атомарно записывает файл: пишет во временный файл-сосед, затем os.replace """
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_path_str)
    try:
        if "b" in mode:
            with os.fdopen(fd, mode) as f:
                yield f
        else:
            with os.fdopen(fd, mode, encoding=encoding) as f:
                yield f
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
