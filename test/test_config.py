"""
Тесты для backend/app/config.py: относительные пути из настроек
(модели, хранилище) должны считаться от корня проекта, а не от текущей
директории процесса.

Settings создаётся с _env_file=None - тесты не должны зависеть от того,
что лежит в реальном .env разработчика.

Запуск: из корня проекта  ->  pytest tests/test_config.py
"""

from pathlib import Path

import pytest

from backend.app import config
from backend.app.config import DEFAULT_ADMIN_API_KEY, Settings

_PATH_ENV_VARS = (
    "DETECTOR_MODEL_PATH",
    "REID_MODEL_PATH",
    "TEMP_UPLOAD_DIR",
    "PERMANENT_STORAGE_DIR",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in _PATH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")


def _settings() -> Settings:
    return Settings(_env_file=None)


class TestPathsResolvedAgainstProjectRoot:
    def test_default_paths_are_absolute_and_inside_project(self):
        settings = _settings()
        root = config._PROJECT_ROOT

        for value in (
            settings.detector_model_path,
            settings.reid_model_path,
            settings.temp_upload_dir,
            settings.permanent_storage_dir,
        ):
            path = Path(value)
            assert path.is_absolute()
            assert root in path.parents

    def test_default_model_locations_match_what_export_scripts_produce(self):
        settings = _settings()
        root = config._PROJECT_ROOT

        assert Path(settings.detector_model_path) == root / "ml/detection/exported/vehicle_detector.onnx"
        assert Path(settings.reid_model_path) == root / "ml/reid/exported/reid_embedder.onnx"

    def test_result_does_not_depend_on_current_working_directory(self, tmp_path, monkeypatch):
        before = _settings()
        monkeypatch.chdir(tmp_path)  # запуск "из другой папки"
        after = _settings()

        assert after.detector_model_path == before.detector_model_path
        assert after.reid_model_path == before.reid_model_path
        assert after.temp_upload_dir == before.temp_upload_dir
        assert after.permanent_storage_dir == before.permanent_storage_dir

    def test_relative_override_is_resolved_against_project_root(self, monkeypatch):
        monkeypatch.setenv("REID_MODEL_PATH", "custom/my_model.onnx")
        assert Path(_settings().reid_model_path) == config._PROJECT_ROOT / "custom/my_model.onnx"

    def test_absolute_override_is_kept_as_is(self, tmp_path, monkeypatch):
        absolute = tmp_path / "volume" / "detector.onnx"
        monkeypatch.setenv("DETECTOR_MODEL_PATH", str(absolute))
        assert Path(_settings().detector_model_path) == absolute

    def test_storage_dirs_follow_the_same_rule(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEMP_UPLOAD_DIR", "var/tmp_uploads")
        monkeypatch.setenv("PERMANENT_STORAGE_DIR", str(tmp_path / "perm"))
        settings = _settings()

        assert Path(settings.temp_upload_dir) == config._PROJECT_ROOT / "var/tmp_uploads"
        assert Path(settings.permanent_storage_dir) == tmp_path / "perm"


class TestIsolationFromDeveloperDotenv:
    def test_project_dotenv_does_not_leak_into_tests(self, monkeypatch):
        # Settings() БЕЗ _env_file=None: изоляцию обеспечивает фикстура из
        # conftest.py. Если у разработчика в .env задан свой ADMIN_API_KEY,
        # без изоляции здесь был бы он, а не значение по умолчанию.
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        assert Settings().admin_api_key == DEFAULT_ADMIN_API_KEY

    def test_explicit_environment_variables_still_work(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", "задан-тестом")
        assert Settings().admin_api_key == "задан-тестом"