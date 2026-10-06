"""
Общие фикстуры для всего набора тестов.

Изоляция от окружения разработчика. backend/app/config.py читает настройки
из .env в корне проекта и из переменных окружения, а у каждого разработчика
там свои значения (свой ADMIN_API_KEY, порт БД, пути моделей и хранилища,
ENFORCE_HTTPS...). Если тесты их подхватывают, результат зависит от машины:
тест "значение по умолчанию" проходит у того, кто ничего не менял, и падает у
того, кто задал свой токен; а тесты, запускающие приложение, создавали бы
папки хранилища там, куда указывает ваш .env (или прямо в data/storage проекта).

Утечка идёт двумя путями, закрываются оба:
  1. Файл .env, который Settings читает сам -> отключается (env_file=None).
  2. Переменные окружения: тестовые модули вызывают python-dotenv
     load_dotenv() (нужно для TEST_DATABASE_URL), и он выгружает ВЕСЬ .env в
     os.environ. Поэтому перед каждым тестом из окружения удаляются все
     переменные, которые читает Settings, и ставятся безопасные значения по
     умолчанию: заведомо недоступная БД и временная папка вместо хранилища.
     Тесты, которым нужны другие значения, задают их сами через
     monkeypatch.setenv - это выполняется позже и имеет приоритет.

TEST_DATABASE_URL при этом не затрагивается (это не настройка приложения).
"""

import pytest

from backend.app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_settings_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    # Всё, что читает Settings, - из окружения прочь (утечка из load_dotenv() или из оболочки).
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)

    # Безопасные значения по умолчанию. Тест, которому нужно иное, перекрывает их
    # своим monkeypatch.setenv.
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
    monkeypatch.setenv("TEMP_UPLOAD_DIR", str(tmp_path / "isolated_temp"))
    monkeypatch.setenv("PERMANENT_STORAGE_DIR", str(tmp_path / "isolated_permanent"))

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def build_toy_embedder():
    """
    Фабрика НАСТОЯЩЕЙ крошечной ONNX-модели эмбеддинга (без торча и обучения):
    input[N,3,H,W] -> GlobalAveragePool -> Flatten -> MatMul(3 x out_dim) ->
    embedding[N,out_dim]. Имена входа и выхода те же, что у
    ml/reid/export_reid.py. Эмбеддинг зависит от среднего цвета картинки,
    поэтому разноцветные изображения получают разные направления векторов -
    этого хватает, чтобы проверять ранжирование поиска на настоящем
    onnxruntime. Пропускает тест, если пакет onnx не установлен.
    """
    pytest.importorskip("onnx")

    def _build(path, out_dim: int = 512, size: int = 64):
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper, save

        np.random.seed(0)
        graph = helper.make_graph(
            [helper.make_node("GlobalAveragePool", ["input"], ["p"]),
             helper.make_node("Flatten", ["p"], ["f"], axis=1),
             helper.make_node("MatMul", ["f", "W"], ["embedding"])],
            "toy",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, size, size])],
            [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, out_dim])],
            initializer=[numpy_helper.from_array(np.random.randn(3, out_dim).astype(np.float32), "W")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        save(model, str(path))
        return path

    return _build