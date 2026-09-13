"""
Загрузчик датасета VeRi-776 для задачи Vehicle Re-Identification.

Ожидаемая структура каталогов после скачивания VeRi-776:

    VeRi/
        image_train/
        image_query/
        image_test/

Формат имени файла (стандартный релиз VeRi-776):

    {vehicle_id:04d}_c{camera_id:03d}_{frame_id:08d}_{track_idx}.jpg

    пример:  0002_c002_00030600_0.jpg
             -> vehicle_id = 2, camera_id = 2

Модуль строит три сплита (train / query / gallery) в виде списков
VeRiSample(path, pid, camid) и предоставляет обёртку
torch.utils.data.Dataset для каждого из них.

Важно:
    - Для train идентичности перенумеровываются в непрерывный диапазон
      [0, num_train_pids) — это требование classification head
      (CrossEntropyLoss ожидает индексы классов 0..N-1).
    - Для query/gallery исходные vehicle_id сохраняются без изменений,
      т.к. по ним происходит сопоставление query <-> gallery на этапе
      оценки (mAP / CMC).
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

# VeRi-776: 0001_c001_00016450_0.jpg
_FILENAME_PATTERN = re.compile(r"^(\d+)_c(\d+)_")


@dataclass
class VeRiSample:
    path: str
    pid: int      # идентичность транспортного средства
    camid: int    # id камеры


def _parse_filename(filename: str) -> Tuple[int, int]:
    """Извлекает (vehicle_id, camera_id) из имени файла VeRi-776."""
    match = _FILENAME_PATTERN.match(os.path.basename(filename))
    if match is None:
        raise ValueError(
            f"Имя файла не соответствует формату VeRi-776: {filename}"
        )
    vehicle_id, camera_id = match.groups()
    return int(vehicle_id), int(camera_id)


def _scan_folder(folder: str, relabel: bool = False) -> List[VeRiSample]:
    """
    Сканирует папку с изображениями и строит список VeRiSample.

    Изображения с pid в (-1, 0) пропускаются (junk/distractor images) —
    в стандартном VeRi-776 их нет, но проверка оставлена для
    совместимости с другими релизами/датасетами похожего формата.
    """
    img_paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.png"))
    )
    if not img_paths:
        raise FileNotFoundError(f"Изображения не найдены в {folder}")

    raw_samples = []
    pid_set = set()
    for path in img_paths:
        pid, camid = _parse_filename(path)
        if pid in (-1, 0):
            continue
        raw_samples.append((path, pid, camid))
        pid_set.add(pid)

    if relabel:
        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_set))}
        samples = [
            VeRiSample(p, pid2label[pid], camid) for p, pid, camid in raw_samples
        ]
    else:
        samples = [VeRiSample(p, pid, camid) for p, pid, camid in raw_samples]

    return samples


class VeRi776:
    """
    Строит train / query / gallery сплиты из корневой директории VeRi-776.
    """

    def __init__(self, root: str):
        self.root = root
        self.train_dir = os.path.join(root, "image_train")
        self.query_dir = os.path.join(root, "image_query")
        self.gallery_dir = os.path.join(root, "image_test")

        for d in (self.train_dir, self.query_dir, self.gallery_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Ожидаемая подпапка VeRi-776 не найдена: {d}"
                )

        self.train = _scan_folder(self.train_dir, relabel=True)
        self.query = _scan_folder(self.query_dir, relabel=False)
        self.gallery = _scan_folder(self.gallery_dir, relabel=False)

        self.num_train_pids = len({s.pid for s in self.train})
        self.num_train_cams = len({s.camid for s in self.train})

    def summary(self) -> str:
        return (
            f"VeRi-776 датасет @ {self.root}\n"
            f"  train:   {len(self.train):>6} изобр. | "
            f"{self.num_train_pids:>4} идентичностей | {self.num_train_cams} камер\n"
            f"  query:   {len(self.query):>6} изобр. | "
            f"{len({s.pid for s in self.query}):>4} идентичностей\n"
            f"  gallery: {len(self.gallery):>6} изобр. | "
            f"{len({s.pid for s in self.gallery}):>4} идентичностей"
        )


class VeRiImageDataset(Dataset):
    """
    torch Dataset, оборачивающий список VeRiSample.

    Возвращает (image_tensor, pid, camid, path). Путь `path` сохраняется,
    чтобы можно было отследить, какому исходному файлу соответствует
    результат — пригодится для отладки и для отображения найденных
    кандидатов в веб-интерфейсе (top-K similar).
    """

    def __init__(
        self,
        samples: List[VeRiSample],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.pid, sample.camid, sample.path


if __name__ == "__main__":
    # Быстрая ручная проверка:
    #   python veri_dataset.py /path/to/VeRi
    import sys

    if len(sys.argv) != 2:
        print("Использование: python veri_dataset.py <путь_к_корню_VeRi>")
        sys.exit(1)

    ds = VeRi776(sys.argv[1])
    print(ds.summary())