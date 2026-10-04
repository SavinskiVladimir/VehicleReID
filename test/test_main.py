"""
Тесты для backend/app/main.py.

DetectorService/ReIDEmbedderService подменяются заглушками (реальные
веса ONNX не нужны для проверки самого приложения - lifespan, CORS,
health-check). DATABASE_URL для большинства тестов указывает на
заведомо недоступный адрес - это намеренно: health-check должен
корректно сообщать "degraded", а не ронять приложение (ТЗ п.1.3.2.4).
Отдельный тест с РЕАЛЬНЫМ Postgres (TEST_DATABASE_URL) проверяет
"здоровый" путь - пропускается, если переменная не задана, как и в
test_repository.py.

Запуск: из корня проекта  ->  pytest tests/test_main.py
"""

import os

from dotenv import load_dotenv

# ВАЖНО: main.py на уровне модуля делает `app = create_app()`, а
# create_app() требует settings.database_url (обязательное поле без
# значения по умолчанию) - значит DATABASE_URL должен быть в окружении
# уже к МОМЕНТУ импорта backend.app.main, иначе сам импорт упадёт с
# ValidationError. Значение ниже - заглушка для тестов по умолчанию
# (намеренно нерабочий адрес - большинство тестов этого файла как раз
# проверяют graceful-деградацию health-check, а не реальную БД).
#
# Порядок важен: setdefault() ДО load_dotenv(), иначе .env подставил бы
# сюда реальный DATABASE_URL разработчика (т.к. load_dotenv() по
# умолчанию не перезаписывает уже установленные переменные - значит
# нужно установить заглушку первой). TEST_DATABASE_URL при этом всё
# равно подхватится из .env ниже как обычно - setdefault выше трогает
# только DATABASE_URL, не TEST_DATABASE_URL.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
load_dotenv()

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import backend.app.main as main_module
from backend.app import startup_checks
from backend.app.config import DEFAULT_ADMIN_API_KEY, get_settings
from backend.app.db.models import Base
from backend.app.dependencies import get_db_session, get_detector_service, get_search_service


class FakeDetectorService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeReIDEmbedderService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def patch_ml_services(monkeypatch):
    # Подменяем ИМЕНА, импортированные В main.py (не оригинальные
    # классы в их собственных модулях) - именно их использует
    # create_app()/lifespan при создании сервисов.
    monkeypatch.setattr(main_module, "DetectorService", FakeDetectorService)
    monkeypatch.setattr(main_module, "ReIDEmbedderService", FakeReIDEmbedderService)
    # Реальных файлов моделей в тестах нет, а заглушки не умеют считать
    # эмбеддинги - поэтому проверки старта, которым они нужны, здесь
    # отключены. Сами проверки покрыты в test_startup_checks.py, а
    # тесты ниже (TestStartupFailFast) включают их обратно явно.
    monkeypatch.setattr(main_module, "check_model_files", lambda settings: None)
    monkeypatch.setattr(main_module, "check_embedding_dimension", lambda *args, **kwargs: 512)
    yield


_FAKE_DATABASE_URL = "postgresql+asyncpg://fake:fake@localhost:59999/fake"


@pytest.fixture(autouse=True)
def reset_settings_cache(monkeypatch):
    # monkeypatch.setenv ВСЕГДА переопределяет значение на время теста и
    # сам восстанавливает исходное после - в отличие от
    # os.environ.setdefault() (который применяется только если
    # переменной ещё нет в окружении). Это важно: если DATABASE_URL уже
    # задан как реальная переменная окружения (например, остался с
    # более ранних шагов настройки проекта - export для
    # create_table.py/create_indexes.py), setdefault() молча не
    # сработает, и тесты, ожидающие "недоступную БД", неожиданно
    # получат настоящее рабочее подключение - ровно так тест
    # test_degraded_when_database_unreachable один раз и упал.
    monkeypatch.setenv("DATABASE_URL", _FAKE_DATABASE_URL)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# create_app()
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_app_has_expected_title(self):
        app = main_module.create_app()
        assert app.title == "Vehicle Re-ID API"

    def test_cors_middleware_is_configured(self):
        app = main_module.create_app()
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes

    def test_health_route_registered(self):
        app = main_module.create_app()
        # app.routes может содержать объекты без атрибута .path (например,
        # обёртку подключённого через include_router роутера из
        # api/routes.py) - используем getattr с дефолтом вместо
        # прямого route.path, чтобы не упасть на таких элементах.
        paths = [getattr(route, "path", None) for route in app.routes]
        assert "/health" in paths


# ---------------------------------------------------------------------------
# lifespan + /health
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_degraded_when_database_unreachable(self, monkeypatch):
        # monkeypatch.setenv ПРИНУДИТЕЛЬНО переопределяет переменную на
        # время этого теста, независимо от того, что уже было в
        # окружении процесса (в отличие от os.environ.setdefault() на
        # уровне модуля выше, которая не трогает уже установленную
        # переменную - этого недостаточно, если DATABASE_URL уже
        # экспортирован в текущей терминал-сессии разработчика).
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
        get_settings.cache_clear()

        app = main_module.create_app()
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"] == "unavailable"
        # ML-сервисы при этом должны быть успешно загружены - деградация
        # БД не должна мешать загрузке моделей при старте.
        assert data["detector_loaded"] is True
        assert data["search_service_loaded"] is True
        assert data["embedding_dim"] == 512
        # Про схему при недоступной БД ничего не известно: null, а не false.
        assert data["schema"] == {
            "vector_extension": None, "table": None, "hnsw_index": None, "search_history_table": None,
        }
        assert any("недоступна" in problem for problem in data["problems"])
        # Технические детали подключения (хост/порт/пароль) наружу не отдаются.
        assert "59999" not in response.text and "fake" not in response.text

    def test_unreachable_database_is_logged_at_startup_not_fatal(self, monkeypatch, caplog):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
        get_settings.cache_clear()

        app = main_module.create_app()
        with caplog.at_level(logging.WARNING):
            with TestClient(app):  # приложение стартует, а не падает
                pass

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "БД недоступна при старте" in messages
        assert "docker compose up" in messages  # с подсказкой, что делать

    def test_default_admin_key_is_warned_about_at_startup(self, monkeypatch, caplog):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
        # Значение по умолчанию задаём ЯВНО: ни реальный .env, ни экспортированная
        # в оболочке ADMIN_API_KEY не должны влиять на этот тест.
        monkeypatch.setenv("ADMIN_API_KEY", DEFAULT_ADMIN_API_KEY)
        get_settings.cache_clear()

        app = main_module.create_app()
        with caplog.at_level(logging.WARNING):
            with TestClient(app):
                pass

        assert any("ADMIN_API_KEY" in r.getMessage() for r in caplog.records)

    def test_custom_admin_key_gives_no_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
        monkeypatch.setenv("ADMIN_API_KEY", "x7Qp-2mVd_aLk93Zr")
        get_settings.cache_clear()

        app = main_module.create_app()
        with caplog.at_level(logging.WARNING):
            with TestClient(app):
                pass

        assert not any("ADMIN_API_KEY" in r.getMessage() for r in caplog.records)

    def test_services_loaded_into_app_state(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
        get_settings.cache_clear()

        app = main_module.create_app()
        with TestClient(app):
            assert isinstance(app.state.detector_service, FakeDetectorService)
            assert app.state.search_service is not None
            assert app.state.session_factory is not None

    def test_engine_disposed_after_shutdown(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
        get_settings.cache_clear()

        app = main_module.create_app()
        with TestClient(app):
            engine = app.state.engine
            assert engine is not None
        # После выхода из контекста TestClient lifespan должен был
        # вызвать shutdown-часть (await engine.dispose()) - сложно
        # напрямую проверить "disposed"-флаг у async-движка синхронно,
        # но важно, что выход из контекста не падает с ошибкой.


_TEST_DB = os.environ.get("TEST_DATABASE_URL")
_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS observations_embedding_hnsw_idx ON observations "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
)


def _prepare_schema(url: str, *, table: bool, index: bool) -> None:
    """Приводит тестовую БД к нужному состоянию схемы (синхронная обёртка)."""

    async def _run():
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            if table:
                await conn.run_sync(Base.metadata.create_all)
            if table and index:
                await conn.execute(text(_INDEX_SQL))
        await engine.dispose()

    asyncio.run(_run())


@pytest.mark.skipif(
    not _TEST_DB,
    reason="TEST_DATABASE_URL не задан - пропускаю проверки с реальной БД",
)
class TestHealthCheckWithRealDatabase:
    """Пропускается, если TEST_DATABASE_URL не задан (как в test_repository.py)."""

    def _health(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", _TEST_DB)
        get_settings.cache_clear()
        app = main_module.create_app()
        with TestClient(app) as client:
            return client.get("/health").json()

    def test_ok_when_schema_and_index_ready(self, monkeypatch):
        _prepare_schema(_TEST_DB, table=True, index=True)
        data = self._health(monkeypatch)

        assert data["status"] == "ok"
        assert data["database"] == "ok"
        assert data["schema"] == {
            "vector_extension": True, "table": True, "hnsw_index": True, "search_history_table": True,
        }
        assert data["problems"] == []
        assert data["warnings"] == []

    def test_still_ok_but_warns_when_hnsw_index_missing(self, monkeypatch):
        # Поиск без индекса работает (полным перебором) - сервис не
        # "degraded", но предупреждение о несоответствии ТЗ по скорости есть.
        _prepare_schema(_TEST_DB, table=True, index=False)
        data = self._health(monkeypatch)

        assert data["status"] == "ok"
        assert data["schema"]["hnsw_index"] is False
        assert any("create_indexes" in w for w in data["warnings"])

    def test_startup_log_warns_about_missing_hnsw_index(self, monkeypatch, caplog):
        _prepare_schema(_TEST_DB, table=True, index=False)
        monkeypatch.setenv("DATABASE_URL", _TEST_DB)
        get_settings.cache_clear()

        app = main_module.create_app()
        with caplog.at_level(logging.WARNING):
            with TestClient(app):
                pass

        assert any("create_indexes" in r.getMessage() for r in caplog.records)

    def test_degraded_when_table_missing(self, monkeypatch):
        _prepare_schema(_TEST_DB, table=False, index=False)
        data = self._health(monkeypatch)

        assert data["status"] == "degraded"
        assert data["database"] == "ok"  # сама БД доступна, не готова схема
        assert data["schema"]["table"] is False
        assert any("create_table" in p for p in data["problems"])


# ---------------------------------------------------------------------------
# Отказ при старте: неисправимая без человека конфигурация
# ---------------------------------------------------------------------------

class TestStartupFailFast:
    def test_missing_model_files_prevent_startup_with_clear_message(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main_module, "check_model_files", startup_checks.check_model_files)
        monkeypatch.setenv("DATABASE_URL", _FAKE_DATABASE_URL)
        monkeypatch.setenv("DETECTOR_MODEL_PATH", str(tmp_path / "нет_такого_детектора.onnx"))
        monkeypatch.setenv("REID_MODEL_PATH", str(tmp_path / "нет_такого_reid.onnx"))
        get_settings.cache_clear()

        app = main_module.create_app()
        with pytest.raises(startup_checks.StartupCheckError) as exc_info:
            with TestClient(app):
                pass

        message = str(exc_info.value)
        assert "нет_такого_детектора.onnx" in message
        assert "нет_такого_reid.onnx" in message
        assert "export_detector.py" in message and "export_reid.py" in message

    def test_embedding_dimension_mismatch_prevents_startup(self, monkeypatch):
        class EmbedderWithWrongDim:
            def __init__(self, **kwargs):
                pass

            def embed_as_list(self, preprocessed):
                return [0.0] * 64  # в БД колонка VECTOR(512)

        monkeypatch.setattr(main_module, "ReIDEmbedderService", EmbedderWithWrongDim)
        monkeypatch.setattr(
            main_module, "check_embedding_dimension", startup_checks.check_embedding_dimension
        )
        monkeypatch.setenv("DATABASE_URL", _FAKE_DATABASE_URL)
        get_settings.cache_clear()

        app = main_module.create_app()
        with pytest.raises(startup_checks.StartupCheckError, match="64.*512|512.*64"):
            with TestClient(app):
                pass


# ---------------------------------------------------------------------------
# dependencies.py
# ---------------------------------------------------------------------------

class _FakeState:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeApp:
    def __init__(self, state):
        self.state = state


class _FakeRequest:
    def __init__(self, app):
        self.app = app


class TestDependencies:
    def test_get_detector_service_reads_from_app_state(self):
        fake_detector = object()
        request = _FakeRequest(_FakeApp(_FakeState(detector_service=fake_detector)))
        assert get_detector_service(request) is fake_detector

    def test_get_search_service_reads_from_app_state(self):
        fake_search = object()
        request = _FakeRequest(_FakeApp(_FakeState(search_service=fake_search)))
        assert get_search_service(request) is fake_search

    @pytest.mark.asyncio
    async def test_get_db_session_yields_session_from_factory(self):
        class FakeSession:
            closed = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                self.closed = True

        fake_session = FakeSession()

        def fake_session_factory():
            return fake_session

        request = _FakeRequest(_FakeApp(_FakeState(session_factory=fake_session_factory)))

        agen = get_db_session(request)
        yielded = await agen.__anext__()
        assert yielded is fake_session

        # Завершаем генератор - должен закрыть сессию через __aexit__.
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()