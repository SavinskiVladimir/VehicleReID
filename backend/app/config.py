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
from pathlib import Path
from typing import List, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

# В отличие от python-dotenv's load_dotenv() (используется, например, в
# db/create_table.py), которая по умолчанию ищет .env, поднимаясь вверх
# по дереву директорий от текущей рабочей директории, pydantic-settings
# ищет env_file СТРОГО относительно CWD, без такого подъёма - если
# uvicorn запущен не из корня проекта (например, из backend/), файл
# .env там просто не находится, и Settings() падает с "Field required"
# для всех полей без значений по умолчанию.
#
# Вычисляем путь к .env от расположения ЭТОГО файла (не от CWD), чтобы
# приложение запускалось одинаково вне зависимости от того, из какой
# директории реально вызван uvicorn. config.py лежит в backend/app/ -
# подняться на два уровня (app -> backend -> корень проекта).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"


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

    # --- Хранилище файлов (storage.py) ---
    temp_upload_dir: str = "data/storage/temp"
    permanent_storage_dir: str = "data/storage/permanent"
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 МБ
    temp_upload_ttl_seconds: int = 30 * 60  # 30 минут

    # --- Доступ к административным функциям (auth.py) ---
    # Значение по умолчанию ("change-me-in-production") намеренно
    # небезопасное и легко узнаваемое - чтобы при реальном
    # развёртывании сразу было заметно, если его забыли переопределить
    # через .env/переменные окружения.
    admin_api_key: str = "change-me-in-production"

    # --- HTTPS (main.py) ---
    # По умолчанию ВЫКЛЮЧЕНО, иначе локальная разработка без TLS-
    # сертификата была бы сломана (HTTPSRedirectMiddleware перенаправлял
    # бы http://localhost в https://localhost, где ничего не слушает).
    # В проде включается через переменную окружения ENFORCE_HTTPS=true -
    # TLS-терминация при этом всё равно происходит ПЕРЕД приложением
    # (reverse-proxy, см. этап 9 ТЗ "Развёртывание"), middleware здесь
    # лишь гарантирует редирект и HSTS-заголовок на уровне приложения.
    enforce_https: bool = False

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

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