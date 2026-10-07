# """
# Хранение файлов: временная загрузка (функция 1 ТЗ - "изображение
# сохранено во временном хранилище") и постоянное хранение изображений
# наблюдений (функция 8 ТЗ).
#
# Разделение на ДВА хранилища намеренное:
#     - temp/ - изображение между вызовом /api/detect и /api/search
#       (пользователь ещё не решил, искать ли вообще и по какому именно
#       ТС на кадре) - живёт ограниченное время, затем считается
#       просроченным.
#     - permanent/ - изображения, которые РЕАЛЬНО сохранены как
#       наблюдения в БД (после вызова /api/observations) - живут, пока
#       существует соответствующая запись в таблице observations.
#
# Хранение на локальной ФС (не S3/объектное хранилище) - сознательное
# упрощение, согласованное с требованием ТЗ п.1.1.2.3 "изображение
# наблюдения (ИЛИ ссылку на него)" - в БД хранится путь, не байты.
# """
#
# from __future__ import annotations
#
# import time
# import uuid
# from pathlib import Path
# from typing import Tuple
#
# from fastapi import UploadFile
# from PIL import Image
#
# _ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
# _ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
#
#
# class UploadValidationError(Exception):
#     """Загруженный файл не прошёл проверку формата/размера (ТЗ п.1.1.2.4, безопасность)."""
#
#
# class UploadNotFoundError(Exception):
#     """upload_id не найден или временный файл уже просрочен."""
#
#
# class FileStorage:
#     """
#     Args:
#         temp_dir: директория для временных загрузок (между /detect и /search).
#         permanent_dir: директория для постоянно хранимых изображений наблюдений.
#         max_upload_size_bytes: максимальный размер загружаемого файла.
#         temp_ttl_seconds: через сколько секунд временная загрузка считается
#             просроченной (защита от бесконтрольного роста temp/ без
#             полноценной фоновой очистки - достаточно для объёма этой
#             работы, см. docstring модуля).
#     """
#
#     def __init__(
#         self,
#         temp_dir: str,
#         permanent_dir: str,
#         max_upload_size_bytes: int = 10 * 1024 * 1024,  # 10 МБ
#         temp_ttl_seconds: int = 30 * 60,  # 30 минут
#     ):
#         self.temp_dir = Path(temp_dir)
#         self.permanent_dir = Path(permanent_dir)
#         self.max_upload_size_bytes = max_upload_size_bytes
#         self.temp_ttl_seconds = temp_ttl_seconds
#         self.cleanup_interval_seconds = 60  # как часто save_temp запускает очистку
#         self._last_cleanup = 0.0
#
#         self.temp_dir.mkdir(parents=True, exist_ok=True)
#         self.permanent_dir.mkdir(parents=True, exist_ok=True)
#
#     async def validate_and_read(self, file: UploadFile) -> bytes:
#         """
#         Проверяет формат (по заявленному Content-Type И по расширению
#         имени файла - полагаться только на Content-Type небезопасно,
#         его может подделать клиент) и размер файла ДО того, как он
#         будет передан на обработку (ТЗ п.1.1.2.4: "загружаемые
#         пользователем файлы должны проходить проверку формата и
#         размера перед передачей на обработку").
#
#         Returns:
#             Байты файла, если проверка пройдена.
#
#         Raises:
#             UploadValidationError: формат не поддерживается или файл
#                 превышает допустимый размер.
#         """
#         if file.content_type not in _ALLOWED_CONTENT_TYPES:
#             raise UploadValidationError(
#                 f"Неподдерживаемый формат файла: {file.content_type}. "
#                 f"Допустимые форматы: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}"
#             )
#
#         suffix = Path(file.filename or "").suffix.lower()
#         if suffix not in _ALLOWED_EXTENSIONS:
#             raise UploadValidationError(
#                 f"Неподдерживаемое расширение файла: {suffix!r}. "
#                 f"Допустимые расширения: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
#             )
#
#         data = await file.read()
#         if len(data) > self.max_upload_size_bytes:
#             raise UploadValidationError(
#                 f"Файл слишком большой: {len(data)} байт "
#                 f"(максимум {self.max_upload_size_bytes} байт)"
#             )
#         if len(data) == 0:
#             raise UploadValidationError("Файл пуст")
#
#         return data
#
#     def cleanup_expired_temp(self) -> int:
#         """
#         Удаляет просроченные временные загрузки (старше temp_ttl_seconds) и возвращает их число.
#         Раньше просроченный файл удалялся только при обращении к нему по upload_id, а загрузки,
#         к которым пользователь так и не вернулся (закрыл вкладку после /detect), копились вечно.
#         Трогает только файлы вида <uuid>.jpg: чужие файлы в папке остаются нетронутыми.
#         """
#         now, removed = time.time(), 0
#         try:
#             candidates = list(self.temp_dir.glob("*.jpg"))
#         except OSError:
#             return 0
#         for path in candidates:
#             try:
#                 uuid.UUID(path.stem)
#                 if now - path.stat().st_mtime > self.temp_ttl_seconds:
#                     path.unlink()
#                     removed += 1
#             except (ValueError, OSError):  # не наш файл либо удалён параллельно
#                 continue
#         return removed
#
#     def _cleanup_if_due(self) -> None:
#         now = time.time()
#         if now - self._last_cleanup >= self.cleanup_interval_seconds:
#             self._last_cleanup = now
#             self.cleanup_expired_temp()
#
#     def save_temp(self, image_bytes: bytes) -> str:
#         """
#         Сохраняет изображение во временное хранилище, возвращает
#         upload_id для последующего обращения (между /api/detect и
#         /api/search - без повторной передачи файла целиком).
#         """
#         self._cleanup_if_due()  # не чаще раза в cleanup_interval_seconds
#         upload_id = str(uuid.uuid4())
#         path = self.temp_dir / f"{upload_id}.jpg"
#
#         # Перекодируем в JPEG независимо от исходного формата - единый
#         # формат хранения упрощает дальнейшую обработку (PIL.Image.open
#         # всё равно понимает любой исходный формат на вход, но хранить
#         # удобнее единообразно).
#         from io import BytesIO
#         try:
#             image = Image.open(BytesIO(image_bytes)).convert("RGB")
#         except Exception as exc:  # битый или обрезанный файл, не изображение, "бомба декомпрессии"
#             raise UploadValidationError("Файл повреждён или не является изображением.") from exc
#         image.save(path, format="JPEG", quality=95)
#
#         return upload_id
#
#     def load_temp(self, upload_id: str) -> Image.Image:
#         """
#         Загружает изображение из временного хранилища по upload_id.
#
#         Raises:
#             UploadNotFoundError: если upload_id не существует, либо
#                 файл старше temp_ttl_seconds (считается просроченным -
#                 клиенту нужно загрузить изображение заново).
#         """
#         path = self._temp_path(upload_id)
#         if not path.is_file():
#             raise UploadNotFoundError(
#                 f"upload_id={upload_id} не найден. Возможно, уже удалён "
#                 f"или никогда не существовал."
#             )
#
#         age_seconds = time.time() - path.stat().st_mtime
#         if age_seconds > self.temp_ttl_seconds:
#             path.unlink(missing_ok=True)
#             raise UploadNotFoundError(
#                 f"upload_id={upload_id} просрочен (старше "
#                 f"{self.temp_ttl_seconds} секунд) - загрузите изображение заново."
#             )
#
#         return Image.open(path).convert("RGB")
#
#     def _temp_path(self, upload_id: str) -> Path:
#         # uuid.UUID(...) - валидация формата upload_id ДО построения
#         # пути: без неё upload_id вида "../../etc/passwd" мог бы
#         # вывести путь за пределы temp_dir (path traversal).
#         try:
#             validated = uuid.UUID(upload_id)
#         except (ValueError, AttributeError, TypeError):
#             raise UploadNotFoundError(f"upload_id={upload_id!r} имеет недопустимый формат.")
#         return self.temp_dir / f"{validated}.jpg"
#
#     def save_permanent(self, image: Image.Image, subdir: str) -> Tuple[str, str]:
#         """
#         Сохраняет изображение в постоянное хранилище (вызывается при
#         успешном добавлении наблюдения в БД).
#
#         Args:
#             subdir: "images" (полный кадр) или "crops" (кроп ТС) -
#                 разделяет два типа файлов по разным подпапкам.
#
#         Returns:
#             (file_id, абсолютный путь к сохранённому файлу) - путь
#             сохраняется в БД как image_path/crop_path.
#         """
#         file_id = str(uuid.uuid4())
#         target_dir = self.permanent_dir / subdir
#         target_dir.mkdir(parents=True, exist_ok=True)
#
#         path = target_dir / f"{file_id}.jpg"
#         image.save(path, format="JPEG", quality=95)
#
#         return file_id, str(path.resolve())
#
#     def delete_temp(self, upload_id: str) -> None:
#         """Удаляет временный файл (вызывается после успешного /api/search или /api/observations - он больше не нужен)."""
#         try:
#             path = self._temp_path(upload_id)
#             path.unlink(missing_ok=True)
#         except UploadNotFoundError:
#             pass  # некорректный upload_id - нечего удалять
#
#     def delete_permanent(self, path_str: str) -> bool:
#         """
#         Удаляет файл из постоянного хранилища. Путь приходит из БД, поэтому
#         удаляются ТОЛЬКО файлы внутри permanent_dir: запись с путём вроде
#         D:\\data\\VeRi\\... (как у наблюдений из demo_populate_db.py - файлы
#         исходного датасета) или с "../" не должна приводить к удалению
#         чего-либо за пределами хранилища.
#
#         Returns:
#             True, если файл был удалён; False, если его нет или путь вне хранилища.
#         """
#         try:
#             path = Path(path_str).resolve()
#             path.relative_to(self.permanent_dir.resolve())
#         except (ValueError, OSError):
#             return False
#         if not path.is_file():
#             return False
#         path.unlink()
#         return True

"""
Хранение файлов: временная загрузка (функция 1 ТЗ - "изображение
сохранено во временном хранилище") и постоянное хранение изображений
наблюдений (функция 8 ТЗ).

Разделение на ДВА хранилища намеренное:
    - temp/ - изображение между вызовом /api/detect и /api/search
      (пользователь ещё не решил, искать ли вообще и по какому именно
      ТС на кадре) - живёт ограниченное время, затем считается
      просроченным.
    - permanent/ - изображения, которые РЕАЛЬНО сохранены как
      наблюдения в БД (после вызова /api/observations) - живут, пока
      существует соответствующая запись в таблице observations.

Хранение на локальной ФС (не S3/объектное хранилище) - сознательное
упрощение, согласованное с требованием ТЗ п.1.1.2.3 "изображение
наблюдения (ИЛИ ссылку на него)" - в БД хранится путь, не байты.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile
from PIL import Image

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class UploadValidationError(Exception):
    """Загруженный файл не прошёл проверку формата/размера (ТЗ п.1.1.2.4, безопасность)."""


class UploadNotFoundError(Exception):
    """upload_id не найден или временный файл уже просрочен."""


class FileStorage:
    """
    Args:
        temp_dir: директория для временных загрузок (между /detect и /search).
        permanent_dir: директория для постоянно хранимых изображений наблюдений.
        max_upload_size_bytes: максимальный размер загружаемого файла.
        temp_ttl_seconds: через сколько секунд временная загрузка считается
            просроченной (защита от бесконтрольного роста temp/ без
            полноценной фоновой очистки - достаточно для объёма этой
            работы, см. docstring модуля).
    """

    def __init__(
        self,
        temp_dir: str,
        permanent_dir: str,
        max_upload_size_bytes: int = 10 * 1024 * 1024,  # 10 МБ
        temp_ttl_seconds: int = 30 * 60,  # 30 минут
    ):
        self.temp_dir = Path(temp_dir)
        self.permanent_dir = Path(permanent_dir)
        self.max_upload_size_bytes = max_upload_size_bytes
        self.temp_ttl_seconds = temp_ttl_seconds
        self.cleanup_interval_seconds = 60  # как часто save_temp запускает очистку
        self._last_cleanup = 0.0

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.permanent_dir.mkdir(parents=True, exist_ok=True)

    async def validate_and_read(self, file: UploadFile) -> bytes:
        """
        Проверяет формат (по заявленному Content-Type И по расширению
        имени файла - полагаться только на Content-Type небезопасно,
        его может подделать клиент) и размер файла ДО того, как он
        будет передан на обработку (ТЗ п.1.1.2.4: "загружаемые
        пользователем файлы должны проходить проверку формата и
        размера перед передачей на обработку").

        Returns:
            Байты файла, если проверка пройдена.

        Raises:
            UploadValidationError: формат не поддерживается или файл
                превышает допустимый размер.
        """
        if file.content_type not in _ALLOWED_CONTENT_TYPES:
            raise UploadValidationError(
                f"Неподдерживаемый формат файла: {file.content_type}. "
                f"Допустимые форматы: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}"
            )

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise UploadValidationError(
                f"Неподдерживаемое расширение файла: {suffix!r}. "
                f"Допустимые расширения: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            )

        data = await file.read()
        if len(data) > self.max_upload_size_bytes:
            raise UploadValidationError(
                f"Файл слишком большой: {len(data)} байт "
                f"(максимум {self.max_upload_size_bytes} байт)"
            )
        if len(data) == 0:
            raise UploadValidationError("Файл пуст")

        return data

    def cleanup_expired_temp(self) -> int:
        """
        Удаляет просроченные временные загрузки (старше temp_ttl_seconds) и возвращает их число.
        Раньше просроченный файл удалялся только при обращении к нему по upload_id, а загрузки,
        к которым пользователь так и не вернулся (закрыл вкладку после /detect), копились вечно.
        Трогает только файлы вида <uuid>.jpg: чужие файлы в папке остаются нетронутыми.
        """
        now, removed = time.time(), 0
        try:
            candidates = list(self.temp_dir.glob("*.jpg"))
        except OSError:
            return 0
        for path in candidates:
            try:
                uuid.UUID(path.stem)
                if now - path.stat().st_mtime > self.temp_ttl_seconds:
                    path.unlink()
                    removed += 1
            except (ValueError, OSError):  # не наш файл либо удалён параллельно
                continue
        return removed

    def _cleanup_if_due(self) -> None:
        now = time.time()
        if now - self._last_cleanup >= self.cleanup_interval_seconds:
            self._last_cleanup = now
            self.cleanup_expired_temp()

    def save_temp(self, image_bytes: bytes) -> str:
        """
        Сохраняет изображение во временное хранилище, возвращает
        upload_id для последующего обращения (между /api/detect и
        /api/search - без повторной передачи файла целиком).
        """
        self._cleanup_if_due()  # не чаще раза в cleanup_interval_seconds
        upload_id = str(uuid.uuid4())
        path = self.temp_dir / f"{upload_id}.jpg"

        # Перекодируем в JPEG независимо от исходного формата - единый
        # формат хранения упрощает дальнейшую обработку (PIL.Image.open
        # всё равно понимает любой исходный формат на вход, но хранить
        # удобнее единообразно).
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.save(path, format="JPEG", quality=95)

        return upload_id

    def load_temp(self, upload_id: str) -> Image.Image:
        """
        Загружает изображение из временного хранилища по upload_id.

        Raises:
            UploadNotFoundError: если upload_id не существует, либо
                файл старше temp_ttl_seconds (считается просроченным -
                клиенту нужно загрузить изображение заново).
        """
        path = self._temp_path(upload_id)
        if not path.is_file():
            raise UploadNotFoundError(
                f"upload_id={upload_id} не найден. Возможно, уже удалён "
                f"или никогда не существовал."
            )

        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > self.temp_ttl_seconds:
            path.unlink(missing_ok=True)
            raise UploadNotFoundError(
                f"upload_id={upload_id} просрочен (старше "
                f"{self.temp_ttl_seconds} секунд) - загрузите изображение заново."
            )

        return Image.open(path).convert("RGB")

    def _temp_path(self, upload_id: str) -> Path:
        # uuid.UUID(...) - валидация формата upload_id ДО построения
        # пути: без неё upload_id вида "../../etc/passwd" мог бы
        # вывести путь за пределы temp_dir (path traversal).
        validated = uuid.UUID(upload_id)
        return self.temp_dir / f"{validated}.jpg"

    def save_permanent(self, image: Image.Image, subdir: str) -> Tuple[str, str]:
        """
        Сохраняет изображение в постоянное хранилище (вызывается при
        успешном добавлении наблюдения в БД).

        Args:
            subdir: "images" (полный кадр) или "crops" (кроп ТС) -
                разделяет два типа файлов по разным подпапкам.

        Returns:
            (file_id, абсолютный путь к сохранённому файлу) - путь
            сохраняется в БД как image_path/crop_path.
        """
        file_id = str(uuid.uuid4())
        target_dir = self.permanent_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        path = target_dir / f"{file_id}.jpg"
        image.save(path, format="JPEG", quality=95)

        return file_id, str(path.resolve())

    def delete_temp(self, upload_id: str) -> None:
        """Удаляет временный файл (вызывается после успешного /api/search или /api/observations - он больше не нужен)."""
        try:
            path = self._temp_path(upload_id)
            path.unlink(missing_ok=True)
        except ValueError:
            pass  # некорректный upload_id - нечего удалять

    def delete_permanent(self, path_str: str) -> bool:
        """
        Удаляет файл из постоянного хранилища. Путь приходит из БД, поэтому
        удаляются ТОЛЬКО файлы внутри permanent_dir: запись с путём вроде
        D:\\data\\VeRi\\... (как у наблюдений из demo_populate_db.py - файлы
        исходного датасета) или с "../" не должна приводить к удалению
        чего-либо за пределами хранилища.

        Returns:
            True, если файл был удалён; False, если его нет или путь вне хранилища.
        """
        try:
            path = Path(path_str).resolve()
            path.relative_to(self.permanent_dir.resolve())
        except (ValueError, OSError):
            return False
        if not path.is_file():
            return False
        path.unlink()
        return True