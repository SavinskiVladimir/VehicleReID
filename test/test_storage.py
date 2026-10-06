"""
Тесты для backend/app/storage.py.

Запуск: из корня проекта  ->  pytest tests/test_storage.py
"""

import io
import uuid
import shutil
import time

import pytest
from PIL import Image

from backend.app.storage import FileStorage, UploadNotFoundError, UploadValidationError


def _make_fake_upload_file(content_type="image/jpeg", filename="car.jpg", data=None):
    """Минимальная имитация fastapi.UploadFile - только то, что использует storage.py."""
    import types

    if data is None:
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (100, 100, 100)).save(buf, format="JPEG")
        data = buf.getvalue()

    async def read():
        return data

    fake = types.SimpleNamespace(content_type=content_type, filename=filename, read=read)
    return fake


@pytest.fixture
def storage(tmp_path):
    return FileStorage(
        temp_dir=str(tmp_path / "temp"),
        permanent_dir=str(tmp_path / "permanent"),
        max_upload_size_bytes=1024 * 1024,
        temp_ttl_seconds=60,
    )


class TestValidateAndRead:
    @pytest.mark.asyncio
    async def test_accepts_valid_jpeg(self, storage):
        upload = _make_fake_upload_file()
        data = await storage.validate_and_read(upload)
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_rejects_wrong_content_type(self, storage):
        upload = _make_fake_upload_file(content_type="text/plain", filename="not_image.txt", data=b"hello")
        with pytest.raises(UploadValidationError, match="формат"):
            await storage.validate_and_read(upload)

    @pytest.mark.asyncio
    async def test_rejects_wrong_extension_even_with_correct_content_type(self, storage):
        # Content-Type подделать легко - проверяем, что расширение файла
        # ТОЖЕ валидируется, а не только (подделываемый) заголовок.
        upload = _make_fake_upload_file(content_type="image/jpeg", filename="payload.exe")
        with pytest.raises(UploadValidationError, match="расширение"):
            await storage.validate_and_read(upload)

    @pytest.mark.asyncio
    async def test_rejects_oversized_file(self, storage):
        big_data = b"x" * (2 * 1024 * 1024)  # 2 МБ > лимита в 1 МБ
        upload = _make_fake_upload_file(data=big_data)
        with pytest.raises(UploadValidationError, match="большой"):
            await storage.validate_and_read(upload)

    @pytest.mark.asyncio
    async def test_rejects_empty_file(self, storage):
        upload = _make_fake_upload_file(data=b"")
        with pytest.raises(UploadValidationError, match="пуст"):
            await storage.validate_and_read(upload)


class TestTempStorage:
    def test_save_and_load_roundtrip(self, storage):
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), (200, 50, 50)).save(buf, format="JPEG")

        upload_id = storage.save_temp(buf.getvalue())
        loaded = storage.load_temp(upload_id)

        assert loaded.size == (40, 30)

    def test_load_nonexistent_upload_id_raises(self, storage):
        import uuid

        with pytest.raises(UploadNotFoundError):
            storage.load_temp(str(uuid.uuid4()))

    def test_load_malformed_upload_id_raises(self, storage):
        with pytest.raises((UploadNotFoundError, ValueError)):
            storage.load_temp("../../etc/passwd")

    def test_expired_upload_raises_and_is_cleaned_up(self, tmp_path):
        short_ttl_storage = FileStorage(
            temp_dir=str(tmp_path / "temp"),
            permanent_dir=str(tmp_path / "permanent"),
            temp_ttl_seconds=0,  # мгновенно просрочен
        )
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        upload_id = short_ttl_storage.save_temp(buf.getvalue())

        time.sleep(0.05)  # гарантируем, что age_seconds > 0

        with pytest.raises(UploadNotFoundError, match="просрочен"):
            short_ttl_storage.load_temp(upload_id)

        # Повторная попытка - файл должен быть уже удалён (не просто
        # "считается просроченным", а реально вычищен с диска).
        path = short_ttl_storage._temp_path(upload_id)
        assert not path.exists()

    def test_delete_temp_removes_file(self, storage):
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        upload_id = storage.save_temp(buf.getvalue())

        storage.delete_temp(upload_id)

        with pytest.raises(UploadNotFoundError):
            storage.load_temp(upload_id)

    def test_delete_temp_with_invalid_id_does_not_raise(self, storage):
        storage.delete_temp("not-a-valid-uuid")  # не должно падать


class TestPermanentStorage:
    def test_save_permanent_creates_file_and_returns_path(self, storage):
        image = Image.new("RGB", (50, 40), (10, 20, 30))
        file_id, path = storage.save_permanent(image, subdir="images")

        assert path.endswith(".jpg")
        from pathlib import Path
        assert Path(path).is_file()

        reloaded = Image.open(path)
        assert reloaded.size == (50, 40)

    def test_images_and_crops_go_to_separate_subdirs(self, storage):
        image = Image.new("RGB", (20, 20))
        _, image_path = storage.save_permanent(image, subdir="images")
        _, crop_path = storage.save_permanent(image, subdir="crops")

        assert "images" in image_path
        assert "crops" in crop_path
        assert image_path != crop_path


class TestDeletePermanent:
    def test_deletes_file_inside_storage(self, storage):
        _, path = storage.save_permanent(Image.new("RGB", (10, 10)), subdir="queries")
        assert storage.delete_permanent(path) is True
        from pathlib import Path
        assert not Path(path).exists()

    def test_missing_file_returns_false(self, storage, tmp_path):
        assert storage.delete_permanent(str(tmp_path / "permanent" / "queries" / "nope.jpg")) is False

    def test_file_outside_storage_is_never_deleted(self, storage, tmp_path):
        # Путь приходит из БД: запись, указывающая на файл исходного датасета
        # (как у наблюдений из demo_populate_db.py), не должна удалять его.
        foreign = tmp_path / "dataset" / "image.jpg"
        foreign.parent.mkdir()
        foreign.write_bytes(b"data")

        assert storage.delete_permanent(str(foreign)) is False
        assert foreign.exists()

    def test_path_traversal_out_of_storage_is_rejected(self, storage, tmp_path):
        foreign = tmp_path / "secret.jpg"
        foreign.write_bytes(b"data")
        sneaky = str(tmp_path / "permanent" / "queries" / ".." / ".." / "secret.jpg")

        assert storage.delete_permanent(sneaky) is False
        assert foreign.exists()

    def test_directory_is_not_deleted(self, storage, tmp_path):
        directory = tmp_path / "permanent" / "queries"
        directory.mkdir(parents=True, exist_ok=True)
        assert storage.delete_permanent(str(directory)) is False
        assert directory.exists()


def _aged_upload(storage, age_seconds, name=None):
    import os
    path = storage.temp_dir / (name or f"{uuid.uuid4()}.jpg")
    path.write_bytes(b"x")
    old = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


class TestCleanupExpiredTemp:
    def test_removes_only_expired_uploads_and_counts_them(self, storage):
        expired1, expired2 = _aged_upload(storage, 3600), _aged_upload(storage, 120)
        fresh = _aged_upload(storage, 5)

        assert storage.cleanup_expired_temp() == 2
        assert not expired1.exists() and not expired2.exists() and fresh.exists()

    def test_foreign_files_in_the_folder_are_left_alone(self, storage):
        not_uuid = _aged_upload(storage, 3600, name="notes.jpg")
        other_type = _aged_upload(storage, 3600, name=f"{uuid.uuid4()}.txt")

        assert storage.cleanup_expired_temp() == 0
        assert not_uuid.exists() and other_type.exists()

    def test_empty_or_missing_folder_is_not_an_error(self, storage):
        assert storage.cleanup_expired_temp() == 0
        shutil.rmtree(storage.temp_dir)
        assert storage.cleanup_expired_temp() == 0

    def test_save_temp_cleans_up_old_uploads(self, storage):
        expired = _aged_upload(storage, 3600)
        storage.save_temp(_png_bytes())
        assert not expired.exists()

    def test_cleanup_on_save_is_throttled(self, storage):
        storage.save_temp(_png_bytes())                       # первая загрузка запускает очистку
        expired_later = _aged_upload(storage, 3600)

        storage.save_temp(_png_bytes())                       # в пределах интервала - не сканируем папку заново
        assert expired_later.exists()

        storage.cleanup_interval_seconds = 0                  # интервал истёк
        storage.save_temp(_png_bytes())
        assert not expired_later.exists()

    def test_fresh_uploads_survive_cleanup_on_save(self, storage):
        first = storage.save_temp(_png_bytes())
        storage.cleanup_interval_seconds = 0
        storage.save_temp(_png_bytes())
        assert storage.load_temp(first) is not None


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()