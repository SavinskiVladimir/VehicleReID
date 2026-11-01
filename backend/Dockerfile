# Образ серверной части (этап 9). Контекст сборки - КОРЕНЬ проекта (нужны backend/, db/, constraints.txt):
#   docker build -f backend/Dockerfile -t vehicle-reid-backend .
# Модели (.onnx) и данные в образ НЕ входят: модели монтируются томами только для чтения в
# /app/ml/detection/exported и /app/ml/reid/exported, хранилище файлов - том /app/data/storage.
# Настройки читаются из переменных окружения (файл .env в образ не копируется, см. .dockerignore).
FROM python:3.12-slim

# CPU-сборка PyTorch (он нужен ultralytics): намного легче CUDA-сборки. Для GPU-профиля индекс
# переопределяется при сборке: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG ORT_PACKAGE=onnxruntime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_AUTOINSTALL=False \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# YOLO_AUTOINSTALL=False: ultralytics не пытается доустанавливать пакеты через pip во время работы (в контейнере
# нет ни прав, ни гарантии сети): недостающая зависимость даёт сразу понятную ошибку, а не зависание.

# libgl1 и libglib2.0-0: нужны opencv (его тянет ultralytics); libgomp1 - OpenMP для onnxruntime и torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости ставятся ДО копирования кода: пока constraints.txt и requirements.txt не менялись,
# этот тяжёлый слой берётся из кэша, и пересборка после правки кода занимает секунды.
# Версии torch/torchvision берутся из constraints.txt (те же, на которых проходят тесты).
COPY constraints.txt ./constraints.txt
COPY backend/requirements.txt ./requirements.txt
#RUN pip install $(grep -E '^(torch|torchvision)==' constraints.txt | tr '\n' ' ') --index-url ${TORCH_INDEX_URL} \
#    && pip install -r requirements.txt -c constraints.txt
COPY wheels /wheels
RUN pip install $(grep -E '^(torch|torchvision)==' constraints.txt | tr '\n' ' ') --find-links /wheels \
    && pip install -r requirements.txt -c constraints.txt

# GPU-профиль: onnxruntime-gpu заменяет обычный onnxruntime (это один и тот же модуль, вместе их держать нельзя).
RUN if [ "$ORT_PACKAGE" = "onnxruntime-gpu" ]; then \
      pip uninstall -y onnxruntime && \
      pip install "$(grep '^onnxruntime==' constraints.txt | sed 's/^onnxruntime/onnxruntime-gpu/')" ; \
    fi

# Раскладка как в проекте: backend/app/config.py считает корень проекта (/app) от своего расположения,
# от него же - пути моделей и хранилища по умолчанию.
COPY backend ./backend
COPY db ./db
COPY scripts ./scripts

# Не root. Папка хранилища создаётся заранее с нужным владельцем: именованный том при первом
# запуске наследует права папки из образа, иначе приложение не смогло бы писать файлы.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/storage/temp /app/data/storage/permanent \
    && chown -R app:app /app/data
USER app

EXPOSE 8000

# /health отвечает 200 и когда БД недоступна (статус degraded), поэтому проверяется живость процесса,
# а не готовность БД. start-period покрывает загрузку моделей.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Схема БД и HNSW-индекс создаются при каждом старте: скрипты идемпотентны (CREATE ... IF NOT EXISTS),
# существующие данные не затрагиваются. Команда записана в CMD, а не в отдельном .sh: у такого файла,
# выгруженного на Windows с переводами строк CRLF, контейнер не стартовал бы.
# --proxy-headers: приложение стоит за nginx и должно видеть исходную схему и адрес клиента.
CMD ["sh", "-c", "python db/create_table.py && python db/create_indexes.py && exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]