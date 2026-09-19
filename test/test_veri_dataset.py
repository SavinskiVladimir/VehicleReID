"""
Тесты для datasets/veri_dataset.py.

Запуск: из папки ml/reid/  ->  pytest test_veri_dataset.py
"""

import os
import random

import pytest
from PIL import Image

from ml.reid.datasets.veri_dataset import (
    VeRi776,
    VeRiImageDataset,
    VeRiSample,
    _parse_filename,
    _scan_folder,
)


# ---------------------------------------------------------------------------
# _parse_filename
# ---------------------------------------------------------------------------

class TestParseFilename:
    def test_extracts_vehicle_and_camera_id(self):
        vehicle_id, camera_id = _parse_filename("0002_c002_00030600_0.jpg")
        assert vehicle_id == 2
        assert camera_id == 2

    def test_ignores_directory_prefix(self):
        vehicle_id, camera_id = _parse_filename("/some/path/0123_c045_00000001_3.jpg")
        assert vehicle_id == 123
        assert camera_id == 45

    def test_raises_on_unrecognized_format(self):
        with pytest.raises(ValueError):
            _parse_filename("not_a_valid_name.jpg")


# ---------------------------------------------------------------------------
# Вспомогательная фабрика синтетического VeRi-776-подобного датасета
# ---------------------------------------------------------------------------

def _make_fake_veri_dataset(root, seed=0):
    """
    Создаёт минимальный, но структурно корректный VeRi-776 датасет:
    3 идентичности x 4 изображения в train (с "дырой" в нумерации id,
    чтобы проверить релейблинг), 2 идентичности в query/gallery.
    """
    random.seed(seed)
    raw_train_ids = [5, 12, 40]  # намеренно НЕ 0,1,2 - проверяем релейблинг

    for split, ids, per_id in [
        ("image_train", raw_train_ids, 4),
        ("image_query", [1, 2], 1),
        ("image_test", [1, 2], 3),
    ]:
        split_dir = os.path.join(root, split)
        os.makedirs(split_dir, exist_ok=True)
        for vid in ids:
            for k in range(per_id):
                cam = random.randint(1, 5)
                frame = random.randint(0, 99999999)
                fname = f"{vid:04d}_c{cam:03d}_{frame:08d}_{k}.jpg"
                Image.new("RGB", (16, 16), (100, 100, 100)).save(
                    os.path.join(split_dir, fname)
                )
    return raw_train_ids


# ---------------------------------------------------------------------------
# _scan_folder
# ---------------------------------------------------------------------------

class TestScanFolder:
    def test_relabels_to_contiguous_range(self, tmp_path):
        raw_ids = _make_fake_veri_dataset(tmp_path, seed=1)
        samples = _scan_folder(str(tmp_path / "image_train"), relabel=True)

        labels = sorted({s.pid for s in samples})
        assert labels == list(range(len(raw_ids)))  # [0, 1, 2] - непрерывно

    def test_keeps_original_ids_without_relabel(self, tmp_path):
        raw_ids = _make_fake_veri_dataset(tmp_path, seed=2)
        samples = _scan_folder(str(tmp_path / "image_query"), relabel=False)

        pids = sorted({s.pid for s in samples})
        assert pids == [1, 2]  # исходные id, не 0-based релейблинг

    def test_raises_if_folder_empty(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _scan_folder(str(empty_dir), relabel=False)


# ---------------------------------------------------------------------------
# VeRi776
# ---------------------------------------------------------------------------

class TestVeRi776:
    def test_loads_all_three_splits(self, tmp_path):
        raw_ids = _make_fake_veri_dataset(tmp_path, seed=3)
        dataset = VeRi776(str(tmp_path))

        assert len(dataset.train) == len(raw_ids) * 4
        assert len(dataset.query) == 2
        assert len(dataset.gallery) == 2 * 3
        assert dataset.num_train_pids == len(raw_ids)

    def test_raises_on_missing_subfolder(self, tmp_path):
        os.makedirs(tmp_path / "image_train")
        # image_query / image_test отсутствуют
        with pytest.raises(FileNotFoundError):
            VeRi776(str(tmp_path))

    def test_summary_contains_counts(self, tmp_path):
        _make_fake_veri_dataset(tmp_path, seed=4)
        dataset = VeRi776(str(tmp_path))
        summary = dataset.summary()
        assert "train" in summary
        assert "query" in summary
        assert "gallery" in summary


# ---------------------------------------------------------------------------
# VeRiImageDataset
# ---------------------------------------------------------------------------

class TestVeRiImageDataset:
    def test_len_matches_samples(self, tmp_path):
        _make_fake_veri_dataset(tmp_path, seed=5)
        dataset = VeRi776(str(tmp_path))
        image_dataset = VeRiImageDataset(dataset.train)
        assert len(image_dataset) == len(dataset.train)

    def test_getitem_without_transform_returns_pil_image(self, tmp_path):
        _make_fake_veri_dataset(tmp_path, seed=6)
        dataset = VeRi776(str(tmp_path))
        image_dataset = VeRiImageDataset(dataset.train)

        img, pid, camid, path = image_dataset[0]
        assert isinstance(img, Image.Image)
        assert isinstance(pid, int)
        assert isinstance(camid, int)
        assert os.path.isfile(path)

    def test_getitem_applies_transform(self, tmp_path):
        _make_fake_veri_dataset(tmp_path, seed=7)
        dataset = VeRi776(str(tmp_path))

        calls = []

        def fake_transform(img):
            calls.append(img)
            return "TRANSFORMED"

        image_dataset = VeRiImageDataset(dataset.train, transform=fake_transform)
        img, *_ = image_dataset[0]

        assert img == "TRANSFORMED"
        assert len(calls) == 1
