"""
Предобработка кадрированного изображения ТС для подачи на вход
ReID-модели (функция 4 из Таблицы 1.3 ТЗ).

Намеренно реализовано БЕЗ зависимости от torch/torchvision: сама идея
экспорта ReID-модели в ONNX (см. ml/reid/export_reid.py) в том, чтобы
инференс на сервере не тащил за собой полный вес обучающего фреймворка.
Если бы предобработка тут использовала torchvision.transforms, backend
всё равно требовал бы установленный torch только ради Resize+Normalize -
бессмысленно тяжёлая зависимость для serving-пути. Вместо этого - чистый
PIL + numpy, результат - обычный numpy-массив, напрямую скармливаемый в
onnxruntime.InferenceSession (см. reid_embedder.py).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

# Те же статистики нормализации ImageNet, что использовались при
# обучении (ml/reid/train.py, build_train_transform) - ДОЛЖНЫ совпадать,
# иначе распределение входа на инференсе разойдётся с тем, что видела
# модель при обучении.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageService:
    """
    Сервис предобработки изображений под вход ReID-модели.

    Args:
        image_size: (H, W), ДОЛЖЕН совпадать с --imgsz, использованным
            при экспорте ONNX-модели (ml/reid/export_reid.py), иначе
            граф ONNX с фиксированным входом либо откажется принимать
            тензор другого размера, либо (при dynamic input) выдаст
            эмбеддинг худшего качества из-за несоответствия masштаба
            объектов тому, что модель видела при обучении.
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Приводит изображение к формату, требуемому ReID-моделью:
        RGB -> resize -> [0,1] -> нормализация ImageNet -> CHW -> batch
        размера 1.

        Returns:
            np.ndarray формы [1, 3, H, W], dtype float32 - готов к
            передаче как есть в onnxruntime.InferenceSession.run(...).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        resized = image.resize(
            (self.image_size[1], self.image_size[0]),  # PIL.resize принимает (W, H)
            resample=Image.BILINEAR,
        )

        array = np.asarray(resized, dtype=np.float32) / 255.0  # [H, W, 3], диапазон [0,1]
        array = (array - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC -> CHW -> добавляем batch-размерность
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)

        return np.ascontiguousarray(array, dtype=np.float32)

    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Предобрабатывает несколько кропов одним вызовом (например,
        несколько ТС, найденных на одном кадре) - удобно для
        батч-инференса через reid_embedder.py.

        Returns:
            np.ndarray формы [N, 3, H, W]. Для пустого списка возвращает
            массив формы [0, 3, H, W] вместо падения на пустой
            np.concatenate.
        """
        if not images:
            return np.empty((0, 3, self.image_size[0], self.image_size[1]), dtype=np.float32)

        batches = [self.preprocess(img) for img in images]
        return np.concatenate(batches, axis=0)