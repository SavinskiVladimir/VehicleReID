"""
Слой доступа к данным (async SQLAlchemy 2.0 + pgvector).

Согласно ТЗ (п.1.3.2.1): "взаимодействие с подсистемой хранения и
поиска должно быть реализовано через отдельный слой доступа к данным,
инкапсулирующий детали конкретной реализации векторного поиска" - этот
модуль и есть такой слой. search_service.py и routes.py зависят только
от публичных функций этого файла, а не от деталей SQLAlchemy/pgvector
напрямую.

Реализует функции 6, 8, 9 из Таблицы 1.3 ТЗ:
    6. Поиск ближайших соседей по признаковому сходству.
    8. Добавление нового наблюдения в базу данных.
    9. Хранение и просмотр истории наблюдений.
"""

from __future__ import annotations

import uuid
from typing import List, Sequence, Tuple

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Observation


async def add_observation(
    session: AsyncSession,
    image_path: str,
    crop_path: str,
    bbox: Tuple[int, int, int, int],
    embedding: Sequence[float],
) -> Observation:
    """
    Добавляет новое наблюдение в базу (функция 8 ТЗ).

    Args:
        bbox: (x1, y1, x2, y2) в пикселях исходного изображения.
        embedding: нормализованный вектор фиксированной размерности
            (см. EMBEDDING_DIM в models.py).

    Returns:
        Созданный объект Observation (уже с присвоенным id и created_at
        после commit).
    """
    x1, y1, x2, y2 = bbox
    observation = Observation(
        image_path=image_path,
        crop_path=crop_path,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        embedding=list(embedding),
    )
    session.add(observation)
    await session.commit()
    await session.refresh(observation)
    return observation


async def search_similar(
    session: AsyncSession,
    query_embedding: Sequence[float],
    top_k: int = 10,
    ef_search: int = 40,
) -> List[Tuple[Observation, float]]:
    """
    Поиск ближайших соседей по признаковому сходству (функция 6 ТЗ).

    Использует оператор косинусного расстояния pgvector (`<=>`),
    соответствующий выбранному в аналитическом разделе работы методу
    индексации HNSW (см. db/create_indexes.py - должен быть выполнен
    заранее, иначе pgvector просто сделает полный перебор по всем
    строкам без ошибки, но и без выигрыша в скорости на больших
    объёмах).

    Args:
        ef_search: размер списка кандидатов на этапе ПОИСКА (не путать
            с ef_construction при построении индекса в
            create_indexes.py) - управляет компромиссом между
            точностью (recall) и скоростью конкретно этого запроса.
            Чем больше значение, тем точнее, но медленнее поиск. 40 -
            значение по умолчанию в самом pgvector. Параметр
            устанавливается через SET LOCAL, то есть действует только
            в пределах текущей транзакции/сессии и не требует
            пересоздания индекса для подбора.

    Returns:
        Список (Observation, similarity), отсортированный по убыванию
        similarity. similarity = 1 - cosine_distance, то есть 1.0 -
        полностью идентичные векторы, 0.0 - ортогональные.
    """
    # SET LOCAL hnsw.ef_search - это GUC-параметр самого расширения
    # pgvector (регистрируется при CREATE EXTENSION vector), доступен
    # даже если HNSW-индекс на таблице ещё не создан - в этом случае он
    # просто ни на что не влияет (postgres делает точный перебор), без
    # ошибки. Безопасно выполнять независимо от того, создан ли индекс.
    #
    # ВАЖНО: команда SET в PostgreSQL НЕ поддерживает bind-параметры
    # ($1-стиль) для своего значения - это ограничение самого протокола
    # PostgreSQL, а не SQLAlchemy/asyncpg (SET - команда конфигурации
    # сессии, а не DML/SELECT). Поэтому значение подставляется в SQL
    # напрямую, а не через параметризованный запрос, как везде в этом
    # файле. Это безопасно ИМЕННО ЗДЕСЬ: ef_search типизирован как int
    # на уровне сигнатуры функции, а не как произвольная строка от
    # пользователя - int(...) ниже дополнительно гарантирует это даже
    # при неожиданном входном значении (упадёт с ValueError, а не
    # пропустит произвольный SQL).
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))

    # pgvector.sqlalchemy.Vector предоставляет метод cosine_distance()
    # прямо на уровне колонки ORM-модели - собственно то самое
    # "инкапсулирование деталей реализации векторного поиска" из ТЗ,
    # вызывающий код (search_service.py) не видит SQL с оператором <=>.
    distance = Observation.embedding.cosine_distance(list(query_embedding))

    stmt = (
        select(Observation, distance.label("distance"))
        .order_by(distance)
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [(obs, 1.0 - dist) for obs, dist in rows]


async def get_observation(session: AsyncSession, observation_id: uuid.UUID) -> Observation | None:
    """Возвращает одно наблюдение по id, либо None, если не найдено."""
    return await session.get(Observation, observation_id)


async def list_observations(
    session: AsyncSession, limit: int = 50, offset: int = 0
) -> Tuple[List[Observation], int]:
    """
    История ранее загруженных наблюдений (функция 9 ТЗ), с пагинацией,
    отсортированная по убыванию даты добавления (сначала новые).

    Returns:
        (список наблюдений на текущей странице, общее число записей).
    """
    count_stmt = select(func.count()).select_from(Observation)
    total = (await session.execute(count_stmt)).scalar_one()

    stmt = (
        select(Observation)
        .order_by(Observation.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = (await session.execute(stmt)).scalars().all()

    return list(items), total


async def delete_observation(session: AsyncSession, observation_id: uuid.UUID) -> bool:
    """
    Удаляет наблюдение по id (требование ТЗ п.1.3.2.3: "система хранения
    должна обеспечивать возможность добавления, поиска по векторному
    сходству и удаления записей").

    Returns:
        True, если запись была найдена и удалена; False, если запись с
        таким id не существовала.
    """
    observation = await session.get(Observation, observation_id)
    if observation is None:
        return False

    await session.delete(observation)
    await session.commit()
    return True