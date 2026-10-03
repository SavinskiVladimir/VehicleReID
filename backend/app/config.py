"""
Конфигурация приложения. Значения читаются из переменных окружения /
.env в корне проекта (тот же файл, что использует db/create_table.py и
Colab-ноутбуки) - единая точка правды для строки подключения к БД и
путей к экспортированным моделям.

Все поля, кроме database_url, имеют разумные значения по умолчанию,
совпадающие с путями, которые реально создают export_detector.py и
export_reid.py при запуске без явного --output.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str

    detector_model_path: str = "ml/detection/exported/vehicle_detector.onnx"
    reid_model_path: str = "ml/reid/exported/reid_embedder.onnx"

    detector_confidence_threshold: float = 0.4
    detector_iou_threshold: float = 0.5
    padding_fraction: float = 0.075

    # Размер входа ReID-модели раздельными int-полями (не tuple) - так
    # их можно задать обычными переменными окружения REID_IMAGE_HEIGHT/
    # REID_IMAGE_WIDTH без JSON-экранирования в .env.
    reid_image_height: int = 224
    reid_image_width: int = 224

    device: str = "cpu"
    default_top_k: int = 10

    cors_origins: List[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def reid_image_size(self) -> Tuple[int, int]:
        return (self.reid_image_height, self.reid_image_width)


@lru_cache
def get_settings() -> Settings:
    """
    Кэширует Settings() на весь процесс - .env читается один раз, а не
    на каждый вызов. В тестах, которым нужны ДРУГИЕ значения (например,
    заведомо нерабочий database_url для проверки graceful-деградации),
    используйте get_settings.cache_clear() перед переопределением
    переменных окружения - см. tests/test_main.py.
    """
    return Settings()