"""Kedro-датасет для бандла обучения"""

from __future__ import annotations
from dataclasses import dataclass 
from pathlib import Path
from typing import Any
from kedro.io import AbstractDataset


@dataclass 
class TrainArtifact:

    model: Any
    manifest: dict


class TrainArtifactBundleDataset(AbstractDataset):
    """Kedro AbstractDataset для бандла обучения"""

    def __init__(self, filepath: Path):
        """Инициализирует датасет путем к директории артефакта"""
        self._path = Path(filepath)

    def save(self, data: TrainArtifact) -> None:
        """Сохрнаяет модель и манифест обучения в директорию"""
        from src.training.model import save_model
        from src.common.io import write_json

        self._path.mkdir(parents=True, exist_ok=True)
        save_model(data.model, self._path)
        write_json(self._path / "manifest.json", data.manifest)

    def load(self) -> TrainArtifact:
        """Загружает артефакт в готовую к инференс модель"""
        from src.training.model import load_model
        from src.common.io import read_json

        return TrainArtifact(model=load_model(self._path), manifest=read_json(self._path / "manifest.json"))

    def _exists(self) -> bool:
        """Проверяет наличие сохраненного артфеакта по файлу"""
        return (self._path / "model.pt").is_file()

    def _describe(self) -> dict[str, Any]:
        """Описание датасета для логов kedro"""
        return {"filepath": str(self._path)}
        