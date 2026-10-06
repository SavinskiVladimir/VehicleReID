"""
Тесты CLI scripts/ingest_images.py.

Два уровня:
  * run() в процессе теста - быстро проверяет коды возврата, сообщения об
    ошибках окружения (нет модели, БД, таблицы), прогресс и итоговую сводку.
    Детектор подменяется заглушкой (в тестах нет весов YOLO), остальное - настоящее:
    PostgreSQL, onnxruntime с крошечной ONNX-моделью, файловое хранилище.
  * НАСТОЯЩИЙ подпроцесс `python scripts/ingest_images.py ...`, запущенный из
    другой директории - проверяет то, что тестом в процессе не поймать:
    sys.path, чтение настроек, кодировку вывода, код возврата ОС.

Требуется TEST_DATABASE_URL (и onnx) - иначе пропускаются.

Запуск: из корня проекта  ->  pytest tests/test_ingest_cli.py
"""

import asyncio
import os
import re
import subprocess
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://fake:fake@localhost:59999/fake")
load_dotenv()

import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import ingest_images  # noqa: E402  (scripts/ - не пакет, путь добавлен выше)

from backend.app.config import get_settings  # noqa: E402
from backend.app.db.models import Base  # noqa: E402

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
needs_database = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL не задан")


def make_image(path: Path, color=(230, 20, 20), vehicles: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64 + vehicles, 64), color).save(path)
    return path


class CountingDetector:
    """Число ТС = ширина кадра - 64 (как в test_ingest_service.py)."""

    def __init__(self, cancel_on_call=None):
        self.calls = 0
        self.cancel_on_call = cancel_on_call

    def detect(self, image):
        self.calls += 1
        if self.cancel_on_call == self.calls:
            raise asyncio.CancelledError  # имитация Ctrl+C посреди загрузки
        return [types.SimpleNamespace(bbox=(i * 10, 0, i * 10 + 20, 30), confidence=0.9 - 0.1 * i)
                for i in range(image.width - 64)]

    def crop_with_padding(self, image, bbox):
        return image.crop(bbox)


def parse(*argv):
    return ingest_images.build_parser().parse_args([str(a) for a in argv])


# ===========================================================================
# Разбор аргументов (без БД)
# ===========================================================================

class TestParser:
    def test_defaults(self):
        args = parse("some/folder")
        assert args.paths == ["some/folder"]
        assert args.mode == "all"
        assert args.confidence is None and args.limit is None
        assert not (args.dry_run or args.replace or args.no_recursive or args.verbose)
        assert args.progress_every == 50

    def test_all_options(self):
        args = parse("a", "b", "--mode", "whole", "--confidence", "0.6", "--limit", "10",
                     "--dry-run", "--replace", "--no-recursive", "-v", "--progress-every", "5")
        assert args.paths == ["a", "b"]
        assert (args.mode, args.confidence, args.limit) == ("whole", 0.6, 10)
        assert args.dry_run and args.replace and args.no_recursive and args.verbose
        assert args.progress_every == 5

    def test_unknown_mode_is_rejected(self, capsys):
        with pytest.raises(SystemExit):
            parse("folder", "--mode", "everything")

    def test_path_is_required(self):
        with pytest.raises(SystemExit):
            parse()

    def test_duration_formatting(self):
        assert ingest_images._format_duration(45) == "45 с"
        assert ingest_images._format_duration(125) == "2 мин 05 с"
        assert ingest_images._format_duration(3 * 3600 + 7 * 60) == "3 ч 07 мин"


# ===========================================================================
# run() в процессе теста
# ===========================================================================

class CliEnv:
    def __init__(self, tmp_path, engine, toy_path):
        self.tmp_path = tmp_path
        self.engine = engine
        self.toy_path = toy_path
        self.frames = tmp_path / "frames"

    async def count(self, table):
        async with self.engine.connect() as conn:
            return (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar()

    def files(self, subdir):
        directory = self.tmp_path / "permanent" / subdir
        return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


@pytest_asyncio.fixture
async def cli_env(tmp_path, monkeypatch, build_toy_embedder):
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL не задан")

    toy = build_toy_embedder(tmp_path / "reid.onnx", out_dim=512, size=64)
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("REID_MODEL_PATH", str(toy))
    monkeypatch.setenv("REID_IMAGE_HEIGHT", "64")
    monkeypatch.setenv("REID_IMAGE_WIDTH", "64")
    monkeypatch.setenv("TEMP_UPLOAD_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("PERMANENT_STORAGE_DIR", str(tmp_path / "permanent"))
    monkeypatch.setenv("DETECTOR_MODEL_PATH", str(tmp_path / "нет_детектора.onnx"))
    get_settings.cache_clear()

    yield CliEnv(tmp_path, engine, toy)
    await engine.dispose()
    get_settings.cache_clear()


def _three_frames(folder: Path):
    make_image(folder / "two.png", (230, 20, 20), vehicles=2)
    make_image(folder / "one.png", (20, 230, 20), vehicles=1)
    make_image(folder / "none.png", (20, 20, 230), vehicles=0)


@needs_database
class TestRunOutcomes:
    @pytest.mark.asyncio
    async def test_full_run_loads_everything_and_reports_a_summary(self, cli_env, capsys):
        _three_frames(cli_env.frames)

        code = await ingest_images.run(parse(cli_env.frames), detector=CountingDetector())

        out = capsys.readouterr().out
        assert code == 0
        assert await cli_env.count("observations") == 3        # 2 + 1 + 0
        assert "Найдено изображений: 3" in out
        assert re.search(r"С новыми наблюдениями:\s+2\b", out)
        assert "добавлено наблюдений: 3" in out
        assert re.search(r"Без найденных ТС:\s+1\b", out)
        assert re.search(r"Ошибок:\s+0\b", out)

    @pytest.mark.asyncio
    async def test_second_run_skips_everything(self, cli_env, capsys):
        _three_frames(cli_env.frames)
        await ingest_images.run(parse(cli_env.frames), detector=CountingDetector())
        capsys.readouterr()

        detector = CountingDetector()
        code = await ingest_images.run(parse(cli_env.frames), detector=detector)

        out = capsys.readouterr().out
        assert code == 0
        assert re.search(r"Пропущено \(уже были\):\s+3\b", out)
        assert detector.calls == 0                              # ни кадра не обработано заново
        assert await cli_env.count("observations") == 3

    @pytest.mark.asyncio
    async def test_dry_run_reports_but_writes_nothing(self, cli_env, capsys):
        _three_frames(cli_env.frames)

        code = await ingest_images.run(parse(cli_env.frames, "--dry-run"), detector=CountingDetector())

        out = capsys.readouterr().out
        assert code == 0
        assert "ничего не записано" in out and "было бы добавлено наблюдений: 3" in out
        assert await cli_env.count("observations") == 0
        assert await cli_env.count("ingest_sources") == 0
        assert cli_env.files("images") == [] and cli_env.files("crops") == []

    @pytest.mark.asyncio
    async def test_bad_file_gives_exit_code_2_but_the_rest_is_loaded(self, cli_env, capsys):
        _three_frames(cli_env.frames)
        broken = cli_env.frames / "broken.jpg"
        broken.write_bytes("не картинка".encode("utf-8"))

        code = await ingest_images.run(parse(cli_env.frames), detector=CountingDetector())

        captured = capsys.readouterr()
        assert code == 2
        assert await cli_env.count("observations") == 3          # остальные загружены
        assert "broken.jpg" in captured.err                      # ошибка показана сразу
        assert "broken.jpg" in captured.out                      # и повторена в итоговом списке
        assert re.search(r"Ошибок:\s+1\b", captured.out)

    @pytest.mark.asyncio
    async def test_copy_in_the_same_folder_is_counted_once(self, cli_env, capsys):
        import shutil
        original = make_image(cli_env.frames / "bus.png", (230, 20, 20), vehicles=2)
        shutil.copy(original, cli_env.frames / "копия.png")

        # проверка: копия считается один раз, как и в настоящей загрузке
        await ingest_images.run(parse(cli_env.frames, "--dry-run"), detector=CountingDetector())
        dry = capsys.readouterr().out
        assert "было бы добавлено наблюдений: 2" in dry
        assert re.search(r"Пропущено \(уже были\):\s+1\b", dry)

        # настоящая загрузка и затем replace: статистика замен не завышена
        await ingest_images.run(parse(cli_env.frames), detector=CountingDetector())
        capsys.readouterr()
        await ingest_images.run(parse(cli_env.frames, "--replace"), detector=CountingDetector())
        out = capsys.readouterr().out
        assert re.search(r"Заменено наблюдений:\s+2\b", out)       # а не 4
        assert await cli_env.count("observations") == 2

    @pytest.mark.asyncio
    async def test_limit(self, cli_env, capsys):
        _three_frames(cli_env.frames)

        await ingest_images.run(parse(cli_env.frames, "--limit", "2"), detector=CountingDetector())

        out = capsys.readouterr().out
        assert "Найдено изображений: 3, обрабатывается: 2 (--limit)" in out
        assert await cli_env.count("ingest_sources") == 2

    @pytest.mark.asyncio
    async def test_replace_reports_replaced_observations(self, cli_env, capsys):
        _three_frames(cli_env.frames)
        await ingest_images.run(parse(cli_env.frames), detector=CountingDetector())
        capsys.readouterr()

        code = await ingest_images.run(
            parse(cli_env.frames, "--mode", "whole", "--replace"), detector=CountingDetector()
        )

        out = capsys.readouterr().out
        assert code == 0
        assert re.search(r"Заменено наблюдений:\s+3\b", out)     # было 2 + 1
        assert await cli_env.count("observations") == 3          # теперь по одному на кадр целиком

    @pytest.mark.asyncio
    async def test_whole_mode_needs_no_detector_at_all(self, cli_env, capsys):
        # DETECTOR_MODEL_PATH указывает на несуществующий файл, детектор не передан.
        _three_frames(cli_env.frames)

        code = await ingest_images.run(parse(cli_env.frames, "--mode", "whole"))

        assert code == 0
        assert await cli_env.count("observations") == 3

    @pytest.mark.asyncio
    async def test_verbose_prints_each_file(self, cli_env, capsys):
        _three_frames(cli_env.frames)

        await ingest_images.run(parse(cli_env.frames, "-v"), detector=CountingDetector())

        out = capsys.readouterr().out
        assert out.count("добавлено") >= 2 and "нет ТС" in out
        assert "two.png" in out and "(2 шт.)" in out

    @pytest.mark.asyncio
    async def test_dry_run_verbose_does_not_claim_files_were_added(self, cli_env, capsys):
        _three_frames(cli_env.frames)

        await ingest_images.run(parse(cli_env.frames, "--dry-run", "-v"), detector=CountingDetector())

        out = capsys.readouterr().out
        assert "добавилось бы" in out
        assert not re.search(r"^добавлено\s", out, flags=re.M)

    @pytest.mark.asyncio
    async def test_progress_line_is_printed(self, cli_env, capsys):
        for i in range(5):
            make_image(cli_env.frames / f"{i}.png", (40 * i, 100, 100), vehicles=1)

        await ingest_images.run(parse(cli_env.frames, "--progress-every", "2"), detector=CountingDetector())

        out = capsys.readouterr().out
        assert "[2/5]" in out and "[4/5]" in out
        assert "изобр./с" in out

    @pytest.mark.asyncio
    async def test_interrupt_prints_summary_returns_130_and_keeps_loaded_files(self, cli_env, capsys):
        for i in range(3):
            make_image(cli_env.frames / f"{i}.png", (60 * i, 100, 100), vehicles=1)

        # Ctrl+C на втором файле
        code = await ingest_images.run(parse(cli_env.frames), detector=CountingDetector(cancel_on_call=2))

        out = capsys.readouterr().out
        assert code == 130
        assert "Прервано пользователем" in out and "можно запустить снова" in out
        assert await cli_env.count("observations") == 1          # первый успел, и он целиком

        # продолжение: простой повторный запуск догружает остальное, без дублей
        code = await ingest_images.run(parse(cli_env.frames), detector=CountingDetector())
        assert code == 0
        assert await cli_env.count("observations") == 3


@needs_database
class TestEnvironmentChecks:
    """Всё, что можно проверить ДО долгой работы, проверяется до неё - с понятным сообщением."""

    @pytest.mark.asyncio
    async def test_missing_path(self, cli_env, capsys):
        code = await ingest_images.run(parse(cli_env.tmp_path / "опечатка"), detector=CountingDetector())
        assert code == 1
        assert "не найден" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_folder_without_images(self, cli_env, capsys):
        (cli_env.frames).mkdir()
        (cli_env.frames / "readme.txt").write_text("нет картинок")
        code = await ingest_images.run(parse(cli_env.frames), detector=CountingDetector())
        assert code == 1
        assert "не найдено изображений" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_invalid_limit(self, cli_env, capsys):
        _three_frames(cli_env.frames)
        assert await ingest_images.run(parse(cli_env.frames, "--limit", "0"), detector=CountingDetector()) == 1

    @pytest.mark.asyncio
    async def test_missing_reid_model(self, cli_env, monkeypatch, capsys):
        _three_frames(cli_env.frames)
        monkeypatch.setenv("REID_MODEL_PATH", str(cli_env.tmp_path / "нет_reid.onnx"))
        get_settings.cache_clear()

        code = await ingest_images.run(parse(cli_env.frames, "--mode", "whole"))

        assert code == 1
        assert "ReID-модель" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_missing_detector_model_suggests_whole_mode(self, cli_env, capsys):
        _three_frames(cli_env.frames)
        code = await ingest_images.run(parse(cli_env.frames, "--mode", "all"))   # детектор не передан
        assert code == 1
        err = capsys.readouterr().err
        assert "детектора" in err and "--mode whole" in err

    @pytest.mark.asyncio
    async def test_wrong_embedding_dimension(self, cli_env, build_toy_embedder, monkeypatch, capsys):
        _three_frames(cli_env.frames)
        wrong = build_toy_embedder(cli_env.tmp_path / "reid64.onnx", out_dim=64, size=64)
        monkeypatch.setenv("REID_MODEL_PATH", str(wrong))
        get_settings.cache_clear()

        code = await ingest_images.run(parse(cli_env.frames, "--mode", "whole"))

        assert code == 1
        assert "64" in capsys.readouterr().err
        assert await cli_env.count("observations") == 0

    @pytest.mark.asyncio
    async def test_missing_ingest_table_points_to_create_table(self, cli_env, capsys):
        # БД, созданная до появления пакетной загрузки: основные таблицы есть, ingest_sources нет.
        async with cli_env.engine.begin() as conn:
            await conn.execute(text("DROP TABLE ingest_sources"))
        _three_frames(cli_env.frames)

        code = await ingest_images.run(parse(cli_env.frames, "--mode", "whole"))

        assert code == 1
        err = capsys.readouterr().err
        assert "ingest_sources" in err and "create_table" in err
        assert await cli_env.count("observations") == 0

    @pytest.mark.asyncio
    async def test_database_without_schema(self, cli_env, capsys):
        async with cli_env.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        _three_frames(cli_env.frames)

        code = await ingest_images.run(parse(cli_env.frames, "--mode", "whole"))

        assert code == 1
        assert "не готова" in capsys.readouterr().err


# ===========================================================================
# Настоящий подпроцесс
# ===========================================================================

@needs_database
class TestRealSubprocess:
    def _env(self, tmp_path, toy):
        env = os.environ.copy()
        env.update({
            "DATABASE_URL": TEST_DATABASE_URL,
            "REID_MODEL_PATH": str(toy),
            "REID_IMAGE_HEIGHT": "64",
            "REID_IMAGE_WIDTH": "64",
            "TEMP_UPLOAD_DIR": str(tmp_path / "temp"),
            "PERMANENT_STORAGE_DIR": str(tmp_path / "permanent"),
            "PYTHONIOENCODING": "utf-8",
        })
        return env

    def _run(self, tmp_path, toy, *argv):
        # cwd - НЕ корень проекта: скрипт обязан работать из любой директории.
        return subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest_images.py"), *map(str, argv)],
            env=self._env(tmp_path, toy), cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8",
            timeout=120,
        )

    def _count(self, table):
        async def _q():
            engine = create_async_engine(TEST_DATABASE_URL)
            async with engine.connect() as conn:
                value = (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar()
            await engine.dispose()
            return value
        return asyncio.run(_q())

    @pytest.fixture
    def prepared(self, tmp_path, build_toy_embedder):
        toy = build_toy_embedder(tmp_path / "reid.onnx", out_dim=512, size=64)

        async def _reset():
            engine = create_async_engine(TEST_DATABASE_URL)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
            await engine.dispose()

        asyncio.run(_reset())
        crops = tmp_path / "crops"
        for i, color in enumerate(((230, 20, 20), (20, 230, 20), (20, 20, 230))):
            make_image(crops / f"car_{i}.png", color, vehicles=0)
        return toy, crops

    def test_whole_mode_end_to_end_from_another_directory(self, tmp_path, prepared):
        toy, crops = prepared

        result = self._run(tmp_path, toy, crops, "--mode", "whole")

        assert result.returncode == 0, result.stderr
        assert "добавлено наблюдений: 3" in result.stdout
        assert self._count("observations") == 3
        assert self._count("ingest_sources") == 3

    def test_rerun_is_idempotent_across_processes(self, tmp_path, prepared):
        toy, crops = prepared
        self._run(tmp_path, toy, crops, "--mode", "whole")

        result = self._run(tmp_path, toy, crops, "--mode", "whole")

        assert result.returncode == 0, result.stderr
        assert re.search(r"Пропущено \(уже были\):\s+3\b", result.stdout)
        assert self._count("observations") == 3

    def test_dry_run_in_a_real_process_writes_nothing(self, tmp_path, prepared):
        toy, crops = prepared

        result = self._run(tmp_path, toy, crops, "--mode", "whole", "--dry-run")

        assert result.returncode == 0, result.stderr
        assert self._count("observations") == 0

    def test_exit_code_1_and_message_on_wrong_path(self, tmp_path, prepared):
        toy, _ = prepared

        result = self._run(tmp_path, toy, tmp_path / "нет_такой_папки", "--mode", "whole")

        assert result.returncode == 1
        assert "не найден" in result.stderr

    def test_exit_code_2_when_some_file_fails(self, tmp_path, prepared):
        toy, crops = prepared
        (crops / "broken.png").write_bytes("битый".encode("utf-8"))

        result = self._run(tmp_path, toy, crops, "--mode", "whole")

        assert result.returncode == 2
        assert "broken.png" in result.stderr
        assert self._count("observations") == 3