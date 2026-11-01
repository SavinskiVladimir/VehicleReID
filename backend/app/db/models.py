"""
ORM-модель наблюдения ТС (SQLAlchemy 2.0, async).

Согласно ТЗ (п.1.3.2.3, требования к информационному обеспечению),
состав хранимых данных должен включать: изображение наблюдения (или
ссылку на него), координаты ограничивающей области, векторное
представление ТС, временную метку добавления записи и уникальный
идентификатор наблюдения - все эти поля отражены в модели ниже.

Векторное поле реализовано через pgvector.sqlalchemy.Vector,
единообразно для всех компонентов системы (размерность 512,
совпадающая с embedding_dim ReID-модели, см. ml/reid/models/
efficientnetv2_reid.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Размерность эмбеддинга ДОЛЖНА совпадать с embedding_dim, с которым
# обучалась и экспортировалась ReID-модель (см. checkpoint["embedding_dim"]
# в ml/reid/export_reid.py). Если эта модель когда-либо будет
# переобучена с другой размерностью - здесь и в вызовах ReIDEmbedderService
# нужно будет синхронно поменять значение.
EMBEDDING_DIM = 512


class Base(DeclarativeBase):
    pass


class Observation(Base):
    """
    Одно наблюдение ТС: исходное изображение + кроп + bbox + эмбеддинг.

    Соответствует функции 8 ТЗ ("Добавление нового наблюдения в базу
    данных") на этапе создания записи и функциям 6/9 (поиск, история)
    на этапе чтения.
    """

    __tablename__ = "observations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Пути/ссылки на файлы, а не сами байты изображения - согласно ТЗ
    # "изображение наблюдения (ИЛИ ссылку на него)". Хранение самих
    # файлов (локальная ФС / S3-совместимое хранилище) вынесено за
    # пределы этой модели - здесь фиксируется только путь.
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    crop_path: Mapped[str] = mapped_column(String, nullable=False)

    # Координаты ограничивающего прямоугольника на ИСХОДНОМ изображении
    # (не на кропе) - чтобы можно было восстановить, откуда именно
    # кадрирована область, если потребуется повторная обработка.
    bbox_x1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x2: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y2: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[list] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    @property
    def bbox(self) -> dict:
        """
        Мост между плоским хранением координат (bbox_x1..bbox_y2 - так
        удобнее для SQL-фильтрации/индексов, если понадобится) и
        вложенной Pydantic-схемой BoundingBox, которую ожидает
        ObservationOut.model_validate(obs, from_attributes=True) в
        search_service.py.
        """
        return {"x1": self.bbox_x1, "y1": self.bbox_y1, "x2": self.bbox_x2, "y2": self.bbox_y2}

    def __repr__(self) -> str:
        return f"<Observation id={self.id} created_at={self.created_at}>"


class SearchQuery(Base):
    """
    Один выполненный поиск - запись ЖУРНАЛА ПОИСКА (функция 9 ТЗ).

    НЕ путать с Observation: Observation - запись БАЗЫ НАБЛЮДЕНИЙ, то
    есть то, В ЧЁМ ищем (функция 8 ТЗ); SearchQuery - то, ЧТО искали и когда.
    Это разные сущности с разным жизненным циклом: база наблюдений растёт
    осознанно (кнопка "Сохранить как наблюдение", пакетное пополнение), а
    журнал пишется автоматически при каждом поиске и ограничен по размеру
    (settings.search_history_max_items).

    Вектор запроса хранится затем, чтобы повторный поиск из истории не
    требовал ни исходного изображения, ни запуска моделей: сравнение идёт
    по сохранённому вектору с ТЕКУЩИМ содержимым базы наблюдений
    (поэтому результат повторного поиска может отличаться от исходного -
    в базе могли появиться или исчезнуть записи).
    """

    __tablename__ = "search_queries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Путь к кропу, по которому искали (для превью в истории). Полный кадр
    # не хранится - он не нужен для повторного поиска и занимал бы место.
    crop_path: Mapped[str] = mapped_column(String, nullable=False)

    # Рамка выбранного ТС на ИСХОДНОМ кадре (как и у Observation).
    bbox_x1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x2: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y2: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[list] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    # Итог поиска на момент выполнения - для списка истории без перезапуска поиска.
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    top_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,  # список истории всегда сортируется по времени
    )

    @property
    def bbox(self) -> dict:
        return {"x1": self.bbox_x1, "y1": self.bbox_y1, "x2": self.bbox_x2, "y2": self.bbox_y2}

    def __repr__(self) -> str:
        return f"<SearchQuery id={self.id} created_at={self.created_at}>"


class IngestSource(Base):
    """
    Учёт файлов, загруженных пакетным пополнением базы (scripts/ingest_images.py).

    Ключ - SHA-256 СОДЕРЖИМОГО файла, а не имя: переименованная или
    перенесённая копия того же изображения не даст дубликатов в базе, а
    повторный запуск на той же папке (в том числе после прерывания) обработает
    только новое. В observation_ids хранятся id наблюдений, созданных из этого
    файла, - это позволяет режиму --replace заменить их, а не наплодить дубли
    при повторной загрузке с другими настройками (режим, порог детекции).

    Запись создаётся и для файла, на котором ТС не найдено (observation_ids
    пустой): повторный запуск не будет заново гонять по нему детектор.
    """

    __tablename__ = "ingest_sources"

    source_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Последний встреченный путь - только для информации (файл могли перенести).
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    observation_ids: Mapped[list] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def __repr__(self) -> str:
        return f"<IngestSource {self.source_hash[:12]} observations={len(self.observation_ids)}>"

class Feedback(Base):
    """
    Отметка оператора о результате поиска (функция 10 ТЗ): «совпадение» / «не совпадение»
    для пары (запрос из журнала поиска, кандидат-наблюдение).

    Вектор и кроп запроса хранятся В САМОЙ отметке (снимок): журнал поиска ограничен по размеру
    и чистится, а отметки - это данные для дообучения модели, они должны его пережить.
    """

    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("query_id", "candidate_id", name="uq_feedback_pair"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("search_queries.id", ondelete="SET NULL"), nullable=True
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("observations.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(16), nullable=False)  # "match" | "no_match"
    operator_id: Mapped[str] = mapped_column(String, nullable=False)
    query_embedding: Mapped[list] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    query_crop_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )