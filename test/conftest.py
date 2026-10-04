"""
Общие фикстуры для всего набора тестов.

Изоляция от реального .env: backend/app/config.py читает .env из корня
проекта, а у каждого разработчика там свои значения (свой ADMIN_API_KEY,
порт БД, пути моделей, ENFORCE_HTTPS...). Если тесты их подхватывают,
результат зависит от машины: тест "значение по умолчанию" проходит у
того, кто ничего не менял, и падает у того, кто задал свой токен.

Поэтому на время каждого теста Settings перестаёт читать файл .env -
остаются только переменные окружения, которые тест задаёт явно
(monkeypatch.setenv), и значения по умолчанию из кода.

Что это НЕ затрагивает: tests/test_*.py по-прежнему вызывают
python-dotenv load_dotenv(), чтобы подхватить из .env TEST_DATABASE_URL -
это отдельный механизм, он работает независимо от настройки ниже.
"""

import pytest

from backend.app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_settings_from_dotenv(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()