"""
Сервис формирования признакового описания ТС (функция 5 из Таблицы 1.3
ТЗ) через инференс экспортированной ONNX-модели (см. ml/reid/export_reid.py).

Согласно ТЗ (п.1.3.2.1), backend взаимодействует с ML-модулем через
отдельный сервисный интерфейс, обеспечивающий независимую замену
ML-модуля без изменения остальной части системы - этот класс и есть
такая точка входа для формирования эмбеддинга, симметрично
DetectorService для детекции.

Инференс через onnxruntime, а не через полный torch/ultralytics -
осознанный выбор ради лёгкого serving-окружения (см. docstring
image_service.py). Единственная зависимость здесь - onnxruntime,
никакого PyTorch на сервере не требуется вообще для этого сервиса.
"""

from __future__ import annotations

from typing import List

import numpy as np
import onnxruntime as ort


class ReIDEmbedderService:
    """
    Сервис извлечения нормализованного векторного представления ТС.

    Пример использования (в search_service.py):

        image_service = ImageService(image_size=(224, 224))
        embedder = ReIDEmbedderService(model_path="ml/reid/exported/reid_embedder.onnx")

        preprocessed = image_service.preprocess(cropped_image)
        embedding = embedder.embed(preprocessed)  # np.ndarray [512]

    Требования к быстродействию (ТЗ п.1.3.2.2): время формирования
    признакового описания не должно превышать 300 мс на CPU / 100 мс на
    GPU - при развёртывании стоит замерить реальную задержку на целевом
    оборудовании (см. вывод _verify_and_benchmark в export_reid.py,
    который уже это проверяет на этапе экспорта).
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Args:
            model_path: путь к reid_embedder.onnx (см. ml/reid/export_reid.py).
            device: 'cpu' или 'cuda'. При 'cuda' используется
                CUDAExecutionProvider (требует onnxruntime-gpu вместо
                обычного onnxruntime), с автоматическим откатом на CPU,
                если провайдер недоступен в окружении - соответствует
                требованию ТЗ о работе системы как на GPU, так и на CPU.
        """
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def embed(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Args:
            preprocessed_image: np.ndarray формы [1, 3, H, W] или
                [N, 3, H, W] (батч), результат ImageService.preprocess()
                или .preprocess_batch().

        Returns:
            np.ndarray формы [D] (для батча из 1 изображения) или
            [N, D] (для батча из N изображений) - L2-нормализованный
            эмбеддинг(и), готовый(е) для сохранения в pgvector и
            косинусного поиска.
        """
        output = self.session.run(
            [self._output_name], {self._input_name: preprocessed_image}
        )[0]

        if output.shape[0] == 1:
            return output[0]
        return output

    def embed_as_list(self, preprocessed_image: np.ndarray) -> List[float]:
        """
        Удобный метод для случая единственного изображения: возвращает
        обычный список float, готовый для передачи в
        ObservationCreate.embedding (Pydantic-схема ожидает List[float],
        а не np.ndarray).
        """
        embedding = self.embed(preprocessed_image)
        if embedding.ndim != 1:
            raise ValueError(
                "embed_as_list ожидает эмбеддинг ОДНОГО изображения "
                f"(1D массив), получен массив формы {embedding.shape}. "
                "Для батча используйте embed() напрямую."
            )
        return embedding.tolist()