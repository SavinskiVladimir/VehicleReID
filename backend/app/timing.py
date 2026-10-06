"""
Поэтапные замеры времени запроса (C3 плана этапа 7) - основа для проверки бюджетов ТЗ п.1.1.2.2:

    detect + crop   получение кадрированного изображения ТС   <= 100 мс на CPU, 3 мс на GPU
    embed           признаковое описание (предобработка + ONNX-модель)   <= 300 мс на CPU, 100 мс на GPU
    search          поиск ближайших соседей в БД (с формированием ответа)   <= 30 мс при 100 000 записей
    total           весь запрос на сервере   <= 700 мс на CPU, 300 мс на GPU (в ТЗ - вместе с сетью;
                    сетевую часть видно в браузере: DevTools -> Network -> Timing)

Результат пишется в заголовок ответа Server-Timing (его показывает DevTools) и одной строкой в лог.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


class StageTimer:
    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.stages: Dict[str, float] = {}  # имя этапа -> миллисекунды

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Замеряет блок; повторное использование имени суммируется. Время пишется и при исключении."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000

    def server_timing(self) -> str:
        """Значение заголовка Server-Timing: 'detect;dur=41.2, embed;dur=88.0, total;dur=140.5'."""
        parts = [f"{name};dur={ms:.1f}" for name, ms in self.stages.items()]
        parts.append(f"total;dur={self.total_ms:.1f}")
        return ", ".join(parts)

    def summary(self) -> str:
        """Строка для лога: 'detect=41мс embed=88мс total=140мс'."""
        parts = [f"{name}={ms:.0f}мс" for name, ms in self.stages.items()]
        parts.append(f"total={self.total_ms:.0f}мс")
        return " ".join(parts)