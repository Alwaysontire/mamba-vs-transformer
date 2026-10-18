"""Канонические колонки датасета и нормализация"""


from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config.schemas import DataSourceConfig

logger = logging.getLogger(__name__)



CONTRACT_COLUMNS: tuple[str, ...] = (
    "name",
    "split",
    "text",
)


def resolve_split_files(source: DataSourceConfig) -> dict[str, list[Path]]:
    """Найти файлы каждого сплита по glob-шаблонам из ``source.splits``

    Raises:
        ValueError: Не задан ``path``.
        FileNotFoundError: Нет директории или для сплита не нашлось ни одного файла.
    """
    if source.path is None:
        raise ValueError(f"Не задан путь data.{source.name}.path в conf/base/source.yml")

    root = Path(source.path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Директория источника {source.name!r} не найдена: {root}")

    suffix = f".{source.kind}"
    files: dict[str, list[Path]] = {}
    for split, pattern in source.splits.items():
        matched: list[Path] = []
        for p in sorted(root.glob(pattern)):
            if p.is_dir():
                matched.extend(sorted(p.glob(f"*{suffix}")))
            elif p.suffix == suffix:
                matched.append(p)
        if not matched:
            raise FileNotFoundError(f"Для сплита {split!r} не найдено {suffix}-файлов по шаблону {root / pattern}")
        files[split] = matched
    return files


def _read_file(path: Path, kind: str) -> pd.DataFrame:
    if kind == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_source(source: DataSourceConfig) -> pd.DataFrame:
    """Прочитать все сплиты источника и привести к контракту + ``source_file``

    Порядок строк сохраняется (файлы сортируются по имени): для WikiText строки —
    это последовательные строки статей, перемешивать их нельзя.
    """
    root = Path(source.path or "").expanduser()
    frames: list[pd.DataFrame] = []
    for split, paths in resolve_split_files(source).items():
        for path in paths:
            raw = _read_file(path, source.kind)
            missing = set(source.column_map.values()) - set(raw.columns)
            if missing:
                raise KeyError(f"В {path} нет колонок {sorted(missing)}, есть: {list(raw.columns)}")

            df = pd.DataFrame({canon: raw[col] for canon, col in source.column_map.items()})
            df["text"] = df["text"].fillna("").astype(str)
            df.insert(0, "name", source.name)
            df.insert(1, "split", split)
            df["source_file"] = str(path.relative_to(root))
            frames.append(df)
            logger.info("%s/%s: %d строк из %s", source.name, split, len(df), path.name)

    return pd.concat(frames, ignore_index=True)
