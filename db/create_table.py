"""
Разовый скрипт создания таблиц (заготовка перед появлением полноценных
Alembic-миграций). Читает DATABASE_URL из .env в корне проекта, а не
хранит строку подключения захардкоженной здесь - при любом изменении
.env (порт, пароль, хост) скрипт всегда использует актуальные значения
без ручной правки.

Запуск (из корня проекта):
    python db/create_table.py
"""

import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine

# Ищет .env в корне проекта независимо от того, откуда запущен скрипт.
load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


async def main() -> None:
    # echo=True - печатает реально выполняемый SQL, полезно при отладке
    # подобных проблем с подключением/аутентификацией.
    engine = create_async_engine(DATABASE_URL, echo=True)

    from backend.app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("Таблица создана.")


if __name__ == "__main__":
    asyncio.run(main())