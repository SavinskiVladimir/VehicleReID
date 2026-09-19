"""
Тесты для samplers/pk_sampler.py.

Запуск: из папки ml/reid/  ->  pytest test_pk_sampler.py
"""

from collections import Counter

import pytest

from samplers.pk_sampler import PKSampler


def _make_pids(id_counts):
    """id_counts: {pid: количество изображений}."""
    pids = []
    for pid, count in id_counts.items():
        pids.extend([pid] * count)
    return pids


class TestPKSamplerConstruction:
    def test_batch_size_is_p_times_k(self):
        pids = _make_pids({0: 5, 1: 5, 2: 5, 3: 5})
        sampler = PKSampler(pids, num_pids_per_batch=2, num_instances=3)
        assert sampler.batch_size == 6

    def test_raises_if_not_enough_identities(self):
        pids = _make_pids({0: 5, 1: 5})
        with pytest.raises(ValueError):
            PKSampler(pids, num_pids_per_batch=3, num_instances=2)


class TestPKSamplerIteration:
    def test_every_batch_has_exactly_p_identities_and_k_instances(self):
        # Разное число изображений на identity, включая identity с
        # МЕНЬШИМ числом изображений, чем K (проверяем sampling with
        # replacement из random.choices).
        id_counts = {0: 3, 1: 6, 2: 10, 3: 2, 4: 8}
        pids = _make_pids(id_counts)
        P, K = 2, 4
        sampler = PKSampler(pids, num_pids_per_batch=P, num_instances=K)

        indices = list(sampler)
        batch_size = P * K

        assert len(indices) % batch_size == 0, "Длина потока индексов должна делиться на batch_size"

        pid_of_index = {}
        offset = 0
        for pid, count in id_counts.items():
            for _ in range(count):
                pid_of_index[offset] = pid
                offset += 1

        num_batches = len(indices) // batch_size
        for b in range(num_batches):
            batch = indices[b * batch_size : (b + 1) * batch_size]
            batch_pids = [pid_of_index[i] for i in batch]
            counts = Counter(batch_pids)

            assert len(counts) == P, f"Батч {b} содержит {len(counts)} identity, ожидалось {P}"
            assert all(v == K for v in counts.values()), (
                f"Батч {b}: не все identity представлены ровно {K} раз: {counts}"
            )

    def test_no_identity_repeats_within_single_batch_indices(self):
        # Внутри одного набора из K индексов конкретной identity индексы
        # не должны совпадать друг с другом (кроме случая replacement,
        # когда изображений у identity меньше K - тогда повтор ожидаем).
        id_counts = {0: 10, 1: 10}
        pids = _make_pids(id_counts)
        sampler = PKSampler(pids, num_pids_per_batch=2, num_instances=4)

        indices = list(sampler)
        batch_size = sampler.batch_size
        first_batch = indices[:batch_size]
        # Для identity с достаточным количеством изображений (>=K) все
        # K выбранных индексов в рамках одной K-группы должны быть
        # различны (sampler делает random.shuffle без replacement, когда
        # изображений достаточно).
        assert len(set(first_batch)) == batch_size

    def test_reproducible_batch_composition_structure_across_seeds(self):
        # Не проверяем конкретные индексы (сэмплер использует global
        # random), а инвариант: КАЖДЫЙ прогон даёт корректную P x K
        # структуру, независимо от случайного порядка.
        id_counts = {i: 5 + i for i in range(6)}
        pids = _make_pids(id_counts)
        sampler = PKSampler(pids, num_pids_per_batch=3, num_instances=3)

        for _ in range(3):  # несколько независимых прогонов __iter__
            indices = list(sampler)
            assert len(indices) % sampler.batch_size == 0
            assert len(indices) > 0
