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

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
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
    crop_path: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)

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