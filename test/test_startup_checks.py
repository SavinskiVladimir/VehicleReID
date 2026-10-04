"""
Тесты для backend/app/startup_checks.py.

Размерность эмбеддинга проверяется на НАСТОЯЩЕЙ ONNX-модели (крошечный
граф, собранный через onnx.helper прямо в тесте) и настоящем
ReIDEmbedderService/onnxruntime - без торча и обучения. Так проверяется
именно то, что произойдёт при запуске с реальным reid_embedder.onnx, а не
поведение заглушки.

Часть тестов требует настоящий PostgreSQL (TEST_DATABASE_URL) - как и в
test_repository.py, они пропускаются, если переменная не задана.

Запуск: из корня проекта  ->  pytest tests/test_startup_checks.py
"""

import os
import types

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.app.db.models import Base
from backend.app.services.image_service import ImageService
from backend.app.config import DEFAULT_ADMIN_API_KEY
from backend.app.startup_checks import (
    DatabaseStatus,
    StartupCheckError,
    admin_key_warnings,
    check_embedding_dimension,
    check_model_files,
    inspect_database,
)

load_dotenv()
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


# ---------------------------------------------------------------------------
# Файлы моделей
# ---------------------------------------------------------------------------

def _settings(detector, reid):
    return types.SimpleNamespace(detector_model_path=str(detector), reid_model_path=str(reid))


class TestCheckModelFiles:
    def test_passes_when_both_files_exist(self, tmp_path):
        detector = tmp_path / "det.onnx"
        reid = tmp_path / "reid.onnx"
        detector.write_bytes(b"x")
        reid.write_bytes(b"x")
        check_model_files(_settings(detector, reid))  # не должно падать

    def test_reports_both_missing_files_at_once(self, tmp_path):
        with pytest.raises(StartupCheckError) as exc_info:
            check_model_files(_settings(tmp_path / "det.onnx", tmp_path / "reid.onnx"))
        message = str(exc_info.value)
        # Оба файла в ОДНОМ сообщении - чтобы не чинить по одному за запуск.
        assert "det.onnx" in message and "reid.onnx" in message

    def test_message_tells_how_to_obtain_each_model(self, tmp_path):
        with pytest.raises(StartupCheckError) as exc_info:
            check_model_files(_settings(tmp_path / "det.onnx", tmp_path / "reid.onnx"))
        message = str(exc_info.value)
        assert "export_detector.py" in message
        assert "export_reid.py" in message
        assert "DETECTOR_MODEL_PATH" in message and "REID_MODEL_PATH" in message

    def test_reports_only_the_missing_one(self, tmp_path):
        detector = tmp_path / "det.onnx"
        detector.write_bytes(b"x")
        with pytest.raises(StartupCheckError) as exc_info:
            check_model_files(_settings(detector, tmp_path / "reid.onnx"))
        message = str(exc_info.value)
        assert "reid.onnx" in message
        assert "export_detector.py" not in message


# ---------------------------------------------------------------------------
# Размерность эмбеддинга
# ---------------------------------------------------------------------------

def _build_toy_embedder_onnx(path, out_dim, input_hw):
    """
    Минимальный настоящий ONNX-граф: input[N,3,H,W] -> GlobalAveragePool
    -> Flatten -> MatMul(3 x out_dim) -> embedding[N,out_dim]. Имена входа
    и выхода ("input"/"embedding") совпадают с теми, что даёт
    ml/reid/export_reid.py.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper, save

    height, width = input_hw
    graph = helper.make_graph(
        nodes=[
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "W"], ["embedding"]),
        ],
        name="toy_embedder",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, height, width])],
        outputs=[helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, out_dim])],
        initializer=[numpy_helper.from_array(np.random.randn(3, out_dim).astype(np.float32), "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # понятно любому достаточно свежему onnxruntime
    save(model, str(path))


@pytest.fixture
def onnx_support():
    pytest.importorskip("onnx")


class TestCheckEmbeddingDimensionRealOnnx:
    def test_correct_dimension_passes_and_is_returned(self, tmp_path, onnx_support):
        from backend.app.services.reid_embedder import ReIDEmbedderService

        model_path = tmp_path / "reid_512.onnx"
        _build_toy_embedder_onnx(model_path, out_dim=512, input_hw=(64, 64))

        embedder = ReIDEmbedderService(str(model_path))
        actual = check_embedding_dimension(embedder, ImageService(image_size=(64, 64)), expected_dim=512)

        assert actual == 512

    def test_wrong_dimension_is_rejected_with_both_numbers_in_message(self, tmp_path, onnx_support):
        from backend.app.services.reid_embedder import ReIDEmbedderService

        model_path = tmp_path / "reid_64.onnx"
        _build_toy_embedder_onnx(model_path, out_dim=64, input_hw=(64, 64))

        embedder = ReIDEmbedderService(str(model_path))
        with pytest.raises(StartupCheckError) as exc_info:
            check_embedding_dimension(embedder, ImageService(image_size=(64, 64)), expected_dim=512)

        message = str(exc_info.value)
        assert "64" in message and "512" in message
        assert "embedding-dim" in message  # подсказка, как исправить

    def test_input_size_mismatch_points_to_image_size_settings(self, tmp_path, onnx_support):
        from backend.app.services.reid_embedder import ReIDEmbedderService

        # Модель экспортирована под 64x64, а в настройках (по умолчанию) 224x224.
        model_path = tmp_path / "reid_fixed_64.onnx"
        _build_toy_embedder_onnx(model_path, out_dim=512, input_hw=(64, 64))

        embedder = ReIDEmbedderService(str(model_path))
        with pytest.raises(StartupCheckError) as exc_info:
            check_embedding_dimension(embedder, ImageService(image_size=(224, 224)), expected_dim=512)

        message = str(exc_info.value)
        assert "REID_IMAGE_HEIGHT" in message
        assert "--imgsz" in message


class TestCheckEmbeddingDimensionWithFakeEmbedder:
    class _Embedder:
        def __init__(self, dim=None, error=None):
            self._dim = dim
            self._error = error

        def embed_as_list(self, preprocessed):
            if self._error:
                raise self._error
            return [0.0] * self._dim

    def test_matching_dimension(self):
        assert check_embedding_dimension(self._Embedder(dim=512), ImageService((32, 32)), 512) == 512

    def test_mismatching_dimension(self):
        with pytest.raises(StartupCheckError):
            check_embedding_dimension(self._Embedder(dim=128), ImageService((32, 32)), 512)

    def test_embedder_failure_is_wrapped_with_original_error(self):
        embedder = self._Embedder(error=RuntimeError("странная ошибка onnx"))
        with pytest.raises(StartupCheckError, match="странная ошибка onnx"):
            check_embedding_dimension(embedder, ImageService((32, 32)), 512)


# ---------------------------------------------------------------------------
# DatabaseStatus (чистая логика, без БД)
# ---------------------------------------------------------------------------

class TestDatabaseStatus:
    def test_ready_requires_reachable_extension_and_table(self):
        assert DatabaseStatus(True, True, True, True).ready is True
        assert DatabaseStatus(True, True, True, False).ready is True  # индекс - не обязателен
        assert DatabaseStatus(True, False, True, True).ready is False
        assert DatabaseStatus(True, True, False, False).ready is False
        assert DatabaseStatus(False).ready is False

    def test_unreachable_reports_single_actionable_problem(self):
        problems = DatabaseStatus(reachable=False, error="secret-host:5432 refused").problems()
        assert len(problems) == 1
        assert "docker compose up" in problems[0]
        # Технические детали подключения не попадают в текст для пользователя.
        assert "secret-host" not in problems[0]

    def test_missing_table_and_extension_each_have_a_remedy(self):
        problems = DatabaseStatus(True, vector_extension=False, table_exists=False).problems()
        assert any("pgvector" in p for p in problems)
        assert any("create_table" in p for p in problems)

    def test_missing_index_is_warning_not_problem(self):
        status = DatabaseStatus(True, True, True, hnsw_index=False)
        assert status.problems() == []
        assert any("create_indexes" in w for w in status.warnings())

    def test_no_index_warning_when_database_not_ready(self):
        # Нет таблицы - говорить про индекс бессмысленно.
        assert DatabaseStatus(True, True, False, False).warnings() == []

    def test_clean_state_has_no_problems_or_warnings(self):
        status = DatabaseStatus(True, True, True, True)
        assert status.problems() == [] and status.warnings() == []


# ---------------------------------------------------------------------------
# inspect_database
# ---------------------------------------------------------------------------

class TestInspectDatabaseUnreachable:
    @pytest.mark.asyncio
    async def test_never_raises_for_unreachable_database(self):
        engine = create_async_engine("postgresql+asyncpg://fake:fake@localhost:59999/fake")
        try:
            status = await inspect_database(engine, timeout_seconds=2)
        finally:
            await engine.dispose()

        assert status.reachable is False
        assert status.ready is False
        assert status.vector_extension is None  # про схему ничего неизвестно
        assert status.error  # причина сохранена - для лога


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL не задан")
class TestInspectDatabaseReal:
    _INDEX_SQL = (
        "CREATE INDEX IF NOT EXISTS observations_embedding_hnsw_idx ON observations "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    @pytest_asyncio.fixture
    async def engine(self):
        engine = create_async_engine(TEST_DATABASE_URL)
        yield engine
        await engine.dispose()

    async def _reset(self, engine, *, table, index):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            if table:
                await conn.run_sync(Base.metadata.create_all)
            if table and index:
                await conn.execute(text(self._INDEX_SQL))

    @pytest.mark.asyncio
    async def test_fully_ready(self, engine):
        await self._reset(engine, table=True, index=True)
        status = await inspect_database(engine)

        assert status.reachable and status.vector_extension
        assert status.table_exists and status.hnsw_index
        assert status.ready and status.problems() == [] and status.warnings() == []

    @pytest.mark.asyncio
    async def test_table_without_index(self, engine):
        await self._reset(engine, table=True, index=False)
        status = await inspect_database(engine)

        assert status.ready
        assert status.hnsw_index is False
        assert status.warnings()

    @pytest.mark.asyncio
    async def test_no_table(self, engine):
        await self._reset(engine, table=False, index=False)
        status = await inspect_database(engine)

        assert status.reachable
        assert status.table_exists is False
        assert status.ready is False
        assert any("create_table" in p for p in status.problems())


# ---------------------------------------------------------------------------
# Токен администратора
# ---------------------------------------------------------------------------

class TestAdminKeyWarnings:
    def test_default_key_triggers_warning(self):
        warnings = admin_key_warnings(DEFAULT_ADMIN_API_KEY)
        assert len(warnings) == 1
        assert "ADMIN_API_KEY" in warnings[0] and ".env" in warnings[0]

    def test_custom_ascii_key_is_fine(self):
        assert admin_key_warnings("x7Qp-2mVd_aLk93Zr") == []

    def test_non_ascii_key_triggers_warning(self):
        warnings = admin_key_warnings("мой-секретный-токен")
        assert len(warnings) == 1
        assert "ASCII" in warnings[0]