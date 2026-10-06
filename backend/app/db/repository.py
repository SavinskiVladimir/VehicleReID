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

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import IngestSource, Observation, SearchQuery


async def add_observation(
    session: AsyncSession,
    image_path: str,
    crop_path: str,
    bbox: Tuple[int, int, int, int],
    embedding: Sequence[float],
    commit: bool = True,
) -> Observation:
    """
    Добавляет новое наблюдение в базу (функция 8 ТЗ).

    Args:
        bbox: (x1, y1, x2, y2) в пикселях исходного изображения.
        embedding: нормализованный вектор фиксированной размерности
            (см. EMBEDDING_DIM в models.py).
        commit: False - только flush, транзакцию закрывает вызывающий код.
            Нужно пакетной загрузке: все наблюдения одного файла и отметка о
            его загрузке должны попасть в БД одной транзакцией (всё или ничего).

    Returns:
        Созданный объект Observation (уже с присвоенным id и created_at).
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
    if commit:
        await session.commit()
        await session.refresh(observation)
    else:
        await session.flush()  # id и created_at проставляются при flush
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


# ---------------------------------------------------------------------------
# Журнал поиска (таблица search_queries) - ОТДЕЛЬНО от базы наблюдений.
# ---------------------------------------------------------------------------

async def add_search_query(
    session: AsyncSession,
    crop_path: str,
    bbox: Tuple[int, int, int, int],
    embedding: Sequence[float],
    top_k: int,
    result_count: int,
    top_similarity: float | None,
) -> SearchQuery:
    """Записывает выполненный поиск в журнал."""
    x1, y1, x2, y2 = bbox
    query = SearchQuery(
        crop_path=crop_path,
        bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
        embedding=list(embedding),
        top_k=top_k,
        result_count=result_count,
        top_similarity=top_similarity,
    )
    session.add(query)
    await session.commit()
    await session.refresh(query)
    return query


async def list_search_queries(
    session: AsyncSession, limit: int = 20, offset: int = 0
) -> Tuple[List[SearchQuery], int]:
    """Журнал поиска, сначала новые. Возвращает (страница, всего записей)."""
    total = (await session.execute(select(func.count()).select_from(SearchQuery))).scalar_one()
    stmt = (
        select(SearchQuery)
        .order_by(SearchQuery.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list((await session.execute(stmt)).scalars().all()), total


async def get_search_query(session: AsyncSession, query_id: uuid.UUID) -> SearchQuery | None:
    return await session.get(SearchQuery, query_id)


async def delete_search_query(session: AsyncSession, query_id: uuid.UUID) -> str | None:
    """
    Удаляет запись журнала. Возвращает путь к её кропу (чтобы вызывающий код
    удалил и файл с диска) либо None, если записи с таким id нет.
    """
    query = await session.get(SearchQuery, query_id)
    if query is None:
        return None
    crop_path = query.crop_path
    await session.delete(query)
    await session.commit()
    return crop_path


async def clear_search_queries(session: AsyncSession) -> List[str]:
    """Полностью очищает журнал. Возвращает пути удалённых кропов."""
    crop_paths = list((await session.execute(select(SearchQuery.crop_path))).scalars().all())
    await session.execute(delete(SearchQuery))
    await session.commit()
    return crop_paths


async def prune_search_queries(session: AsyncSession, keep: int) -> List[str]:
    """
    Оставляет в журнале только `keep` самых новых записей, остальные удаляет.
    Возвращает пути кропов удалённых записей (файлы удаляет вызывающий код).
    Без ограничения журнал рос бы бесконечно: он пишется автоматически при
    каждом поиске, в отличие от базы наблюдений.
    """
    stmt = (
        select(SearchQuery.id, SearchQuery.crop_path)
        .order_by(SearchQuery.created_at.desc())
        .offset(keep)
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        return []
    await session.execute(delete(SearchQuery).where(SearchQuery.id.in_([r[0] for r in rows])))
    await session.commit()
    return [r[1] for r in rows]


async def observation_exists(
    session: AsyncSession,
    image_path: str,
) -> bool:
    """Проверка существования записи для избежания дубликатов."""
    stmt = select(Observation.id).where(
        Observation.image_path == image_path
    ).limit(1)

    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None

# ---------------------------------------------------------------------------
# Пакетная загрузка (таблица ingest_sources). Функции НЕ коммитят: транзакцию
# закрывает вызывающий код (IngestService) - вместе с самими наблюдениями.
# ---------------------------------------------------------------------------

async def get_ingest_source(session: AsyncSession, source_hash: str) -> IngestSource | None:
    return await session.get(IngestSource, source_hash)


async def add_ingest_source(
    session: AsyncSession,
    source_hash: str,
    source_path: str,
    mode: str,
    observation_ids: Sequence[uuid.UUID],
) -> IngestSource:
    source = IngestSource(
        source_hash=source_hash,
        source_path=source_path,
        mode=mode,
        observation_ids=list(observation_ids),
    )
    session.add(source)
    await session.flush()
    return source


async def delete_ingest_source(session: AsyncSession, source_hash: str) -> None:
    await session.execute(delete(IngestSource).where(IngestSource.source_hash == source_hash))


async def delete_observations_by_ids(
    session: AsyncSession, observation_ids: Sequence[uuid.UUID]
) -> List[Tuple[str, str]]:
    """
    Удаляет наблюдения по списку id. Возвращает (image_path, crop_path) удалённых -
    чтобы вызывающий код удалил файлы ПОСЛЕ успешного коммита (иначе при откате
    транзакции записи остались бы без файлов). Несуществующие id игнорируются.
    """
    if not observation_ids:
        return []
    ids = list(observation_ids)
    rows = (await session.execute(
        select(Observation.image_path, Observation.crop_path).where(Observation.id.in_(ids))
    )).all()
    await session.execute(delete(Observation).where(Observation.id.in_(ids)))
    return [(r[0], r[1]) for r in rows]