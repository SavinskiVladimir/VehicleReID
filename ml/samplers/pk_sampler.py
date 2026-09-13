"""
PK-sampler для метрического обучения (Batch-Hard Triplet Loss).

Каждый мини-батч формируется из P идентичностей по K изображений
каждая (размер батча = P * K). Это гарантирует, что в каждом батче
для любого якоря найдётся достаточно позитивных примеров для
online hard-mining, что требуется для batch-hard triplet loss
(см. ml/losses/triplet.py).
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Sequence

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    Формирует индексы так, что каждый батч содержит `num_instances`
    изображений для каждой из `num_pids_per_batch` случайно выбранных
    идентичностей.

    Args:
        pids: последовательность идентичностей, поэлементно
            соответствующая датасету, с которым будет использоваться
            сэмплер (например, [s.pid for s in dataset.samples]).
        num_pids_per_batch: P — количество различных идентичностей
            на батч.
        num_instances: K — количество изображений на идентичность.
            Идентичности, у которых меньше K изображений, дополняются
            выборкой с повторением до K.

    Итоговый batch_size = num_pids_per_batch * num_instances.
    Сэмплер передаётся в DataLoader через `sampler=`, при этом
    `shuffle=True` в DataLoader указывать не нужно — перемешивание
    делается самим сэмплером.

    Примечание: `len(sampler)` — это оценка ёмкости сверху (сумма по
    идентичностям), а не точное число индексов за один вызов
    `__iter__`. Реальное количество обычно чуть меньше, так как в
    цикле выбора батчей идентичность выбывает из пула, как только
    у неё заканчиваются накопленные "пачки" по K изображений — это
    может произойти раньше, чем у остальных. Не используйте
    len(sampler) для точного расчёта steps_per_epoch, только как
    приблизительный ориентир.
    """

    def __init__(
        self,
        pids: Sequence[int],
        num_pids_per_batch: int,
        num_instances: int,
    ):
        self.pids = list(pids)
        self.num_pids_per_batch = num_pids_per_batch
        self.num_instances = num_instances
        self.batch_size = num_pids_per_batch * num_instances

        self.index_dict = defaultdict(list)
        for index, pid in enumerate(self.pids):
            self.index_dict[pid].append(index)
        self.pid_list = list(self.index_dict.keys())

        if len(self.pid_list) < self.num_pids_per_batch:
            raise ValueError(
                f"num_pids_per_batch ({self.num_pids_per_batch}) больше, чем "
                f"количество доступных идентичностей ({len(self.pid_list)})."
            )

        # Оценка длины эпохи: для каждой идентичности считаем, сколько
        # "полных" наборов размера num_instances из неё можно набрать.
        self.length = 0
        for pid in self.pid_list:
            num_imgs = len(self.index_dict[pid])
            num_imgs = max(num_imgs, self.num_instances)
            self.length += num_imgs - num_imgs % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pid_list:
            idxs = copy.deepcopy(self.index_dict[pid])
            if len(idxs) < self.num_instances:
                idxs = random.choices(idxs, k=self.num_instances)
            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avail_pids = copy.deepcopy(self.pid_list)
        final_idxs = []

        while len(avail_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avail_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avail_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


if __name__ == "__main__":
    # Мини-проверка на синтетических данных:
    # 5 идентичностей, у каждой разное число изображений.
    fake_pids = (
        [0] * 3 + [1] * 7 + [2] * 4 + [3] * 2 + [4] * 5
    )
    sampler = PKSampler(fake_pids, num_pids_per_batch=2, num_instances=4)
    batches = list(sampler)
    print(f"Всего индексов: {len(batches)}, batch_size = {sampler.batch_size}")
    print(f"len(sampler) = {len(sampler)}")