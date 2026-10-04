# Vehicle Re-ID

Система поиска транспортных средств по изображению (дипломная работа).
Пользователь загружает кадр, система находит на нём ТС, пользователь
выбирает нужное, и по векторному признаковому описанию находятся
наиболее похожие наблюдения в базе.

```
 браузер ──► frontend (React, :3000) ──► backend (FastAPI, :8000) ──► PostgreSQL + pgvector
                  Vite-прокси /api              │
                                                ├─ детектор YOLO11s   (ONNX)
                                                └─ ReID EfficientNetV2 (ONNX, эмбеддинг 512)
```

## Структура

| Папка | Назначение |
|---|---|
| `backend/` | API на FastAPI: сервисы (детекция, предобработка, эмбеддинг, поиск), слой БД |
| `frontend/` | веб-интерфейс (React + Vite), подробности - `frontend/README.md` |
| `ml/detection/`, `ml/reid/` | обучение, оценка и экспорт моделей |
| `db/` | инициализация БД: расширение, таблица, HNSW-индекс |
| `scripts/` | демонстрационные скрипты (`demo_search.py`, `demo_populate_db.py`) |
| `tests/` | тесты (pytest), запускаются из корня проекта |
| `docker-compose.yml` | база данных (PostgreSQL с pgvector) |

## Требования

- Python 3.12, Node.js `^20.19` или `>=22.12`
- Docker (на Windows - Docker Engine внутри WSL2) для PostgreSQL
- Обученные и экспортированные модели (шаг 5)

## Запуск с нуля

Все команды - **из корня проекта**, если не сказано иное.

**1. Настройки**
```powershell
Copy-Item .env.example .env
```
Откройте `.env`: как минимум смените `ADMIN_API_KEY`, при занятом порту -
`POSTGRES_PORT` (и тот же порт в `DATABASE_URL`). Описание всех
параметров - в самом `.env.example`.

**2. База данных** (в терминале, где доступен Docker; на Windows - Ubuntu в WSL2)
```bash
docker compose up -d db
docker compose ps        # дождаться статуса healthy
```

**3. Зависимости backend**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

**4. Схема БД** (один раз)
```powershell
python db\create_table.py
python db\create_indexes.py
```
Индекс обязателен для требования ТЗ "поиск не более 30 мс на 100 000 записей":
без него поиск работает полным перебором (сервер при этом работает, но
предупредит об этом).

**5. Модели** - положить экспортированные ONNX-файлы туда, где их ждёт backend:
```powershell
# детектор -> ml/detection/exported/vehicle_detector.onnx
python ml\detection\export_detector.py --checkpoint ml\detection\runs\vehicle_detector\weights\best.pt --output ml\detection\exported --imgsz 640

# ReID -> ml/reid/exported/reid_embedder.onnx (запускать ИЗ папки ml\reid)
cd ml\reid
python export_reid.py --checkpoint checkpoints\best_checkpoint.pt --output exported --imgsz 224 224
cd ..\..
```
Размер `--imgsz` ReID-модели должен совпадать с `REID_IMAGE_HEIGHT/WIDTH` в `.env`.

**6. Backend**
```powershell
uvicorn backend.app.main:app --reload
```
При старте сервер проверяет согласованность конфигурации: файлы моделей на
месте, размерность эмбеддинга модели равна 512 (как в колонке БД), состояние
БД. Если что-то не так - он либо не запустится с понятным сообщением
(модели), либо запустится и сообщит в логе и в `/health` (БД).

**7. Frontend** (в другом терминале)
```powershell
cd frontend
npm install
npm run dev
```
Открыть `http://localhost:3000`.

**8. Проверка**
```powershell
curl http://127.0.0.1:8000/health
```

| Поле ответа | Что значит |
|---|---|
| `status: "ok"` | БД доступна, расширение `vector` и таблица `observations` на месте |
| `status: "degraded"` | чего-то не хватает - причины и что делать в `problems` |
| `warnings` | не мешает работе, но влияет на скорость (например, нет HNSW-индекса) |
| `schema.*: null` | БД недоступна, про схему ничего не известно |
| `embedding_dim` | размерность эмбеддинга загруженной ReID-модели (должна быть 512) |

Документация API с возможностью пробовать запросы - `http://127.0.0.1:8000/docs`.

## Конфигурация

- Все настройки - в `.env` в корне проекта; backend находит его независимо от
  того, из какой папки запущен.
- Относительные пути (модели, хранилище файлов) считаются **от корня проекта**.
- Сохранение и удаление наблюдений требуют заголовок `X-Admin-Token` со значением
  `ADMIN_API_KEY` (интерфейс запросит токен сам при первом таком действии).
- `CORS_ORIGINS` принимает только JSON-массив. В штатной схеме он не нужен.

## Тесты

```powershell
pip install -r requirements-test.txt
pytest
```
Тесты, которым нужен настоящий PostgreSQL (`test_repository`, `test_create_indexes`,
`test_startup_checks`, часть `test_main`), используют **отдельную** БД из
`TEST_DATABASE_URL` - они удаляют и пересоздают в ней таблицу. Без этой переменной
такие тесты пропускаются. Создание тестовой БД (в терминале с Docker):
```bash
docker compose exec db psql -U vehiclereid -d vehiclereid -c "CREATE DATABASE vehiclereid_test;"
docker compose exec db psql -U vehiclereid -d vehiclereid_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
Тесты запускаются **из корня проекта**. При push в GitHub их выполняет CI
(`.github/workflows/tests.yml`, с настоящим PostgreSQL+pgvector).

## Частые проблемы

| Симптом | Причина и что делать |
|---|---|
| Сервер не стартует: «Не найдены файлы моделей» | Нет ONNX-файла. В сообщении указано, какой именно и какой командой его получить (шаг 5) |
| Сервер не стартует: «Размерность эмбеддинга (N) не совпадает (512)» | Модель экспортирована с другим `--embedding-dim`. Переэкспортируйте или меняйте схему БД |
| Сервер не стартует: «не смогла обработать тестовое изображение» | `REID_IMAGE_HEIGHT/WIDTH` не совпадают с `--imgsz` при экспорте |
| `/health`: `degraded` | Смотрите `problems`: чаще всего не запущена БД или не выполнен `db\create_table.py` |
| `ModuleNotFoundError: No module named 'backend'` | Команда запущена не из корня проекта |
| Поиск отвечает 503 | Причина - в консоли `uvicorn` (полный traceback). Чаще всего БД или размерность эмбеддинга |
| `password authentication failed` | Том БД создан со старым паролем: `docker compose down -v` и заново (удалит данные) |
| `Connect call failed ... 5433` | Контейнер не поднят или порт не совпадает с `DATABASE_URL`: `docker compose ps`; на Windows помогает `wsl --shutdown` и повторный `up` |
| Запросы к API «перехватывает» посторонний сервер (редирект на чужие страницы, 307) | Порт 8000 занят другим процессом: `netstat -ano \| findstr :8000`; в WSL - `docker ps`, `sudo ss -tulpn \| grep :8000` |
| В браузере остался старый интерфейс | `Ctrl+Shift+R`; при необходимости DevTools - Application - Clear site data |
| `npm` не найден в PowerShell | Установить Node.js LTS системно (`winget install OpenJS.NodeJS.LTS`), открыть новое окно терминала |

## Что пока не сделано

- Контейнеризация backend и frontend, reverse-proxy с HTTPS - этап 9 плана разработки
  (сейчас в Docker работает только БД).
- GPU-профиль (`DEVICE=cuda`) требует `onnxruntime-gpu`; настраивается при развёртывании.
- Роли реализованы упрощённо - единый токен администратора (см. `backend/app/auth.py`).