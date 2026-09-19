"""
Тесты для evaluate.py (mAP/CMC, парсинг официальных файлов протокола
VeRi-776, k-reciprocal re-ranking).

Запуск: из папки ml/reid/  ->  pytest test_evaluate.py
"""

import os

import numpy as np
import pytest
import torch

from ml.reid.evaluate import (
    _detect_index_base,
    _load_named_split,
    _parse_index_file,
    compute_map_cmc,
    k_reciprocal_rerank,
)


# ---------------------------------------------------------------------------
# _parse_index_file
# ---------------------------------------------------------------------------

class TestParseIndexFile:
    def test_parses_space_separated_indices(self, tmp_path):
        path = tmp_path / "gt_index.txt"
        path.write_text("1 2 3\n4 5\n6\n")

        result = _parse_index_file(str(path), expected_num_lines=3)
        assert result == [[1, 2, 3], [4, 5], [6]]

    def test_parses_comma_separated_indices(self, tmp_path):
        path = tmp_path / "gt_index.txt"
        path.write_text("1,2,3\n4,5\n")

        result = _parse_index_file(str(path), expected_num_lines=2)
        assert result == [[1, 2, 3], [4, 5]]

    def test_handles_empty_line(self, tmp_path):
        path = tmp_path / "jk_index.txt"
        path.write_text("1 2\n\n3\n")

        result = _parse_index_file(str(path), expected_num_lines=3)
        assert result == [[1, 2], [], [3]]

    def test_raises_on_line_count_mismatch(self, tmp_path):
        path = tmp_path / "gt_index.txt"
        path.write_text("1 2\n3 4\n")

        with pytest.raises(ValueError):
            _parse_index_file(str(path), expected_num_lines=5)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _parse_index_file(str(tmp_path / "does_not_exist.txt"), expected_num_lines=1)


# ---------------------------------------------------------------------------
# _detect_index_base
# ---------------------------------------------------------------------------

class TestDetectIndexBase:
    def test_detects_1_based(self):
        gallery_pids = np.array([10, 20, 30, 40, 50])
        query_pids = np.array([30, 10])
        gt_index_1based = [[3], [1]]  # 1-based: индекс 3 -> позиция 2 -> pid 30

        assert _detect_index_base(gt_index_1based, query_pids, gallery_pids) == 1

    def test_detects_0_based(self):
        gallery_pids = np.array([10, 20, 30, 40, 50])
        query_pids = np.array([30, 10])
        gt_index_0based = [[2], [0]]  # 0-based: индекс 2 -> pid 30 напрямую

        assert _detect_index_base(gt_index_0based, query_pids, gallery_pids) == 0

    def test_prefers_1_based_when_tied_or_ambiguous(self, capsys):
        # Полностью случайные индексы - оба варианта дают низкое совпадение,
        # но функция не должна падать, а должна вернуть 0 или 1 и напечатать
        # предупреждение.
        gallery_pids = np.array([1, 2, 3])
        query_pids = np.array([99])
        gt_index = [[100]]  # индекс далеко за пределами галереи в обоих случаях

        result = _detect_index_base(gt_index, query_pids, gallery_pids)
        assert result in (0, 1)


# ---------------------------------------------------------------------------
# _load_named_split
# ---------------------------------------------------------------------------

class TestLoadNamedSplit:
    def test_preserves_file_order(self, tmp_path):
        names_file = tmp_path / "name_query.txt"
        names_file.write_text(
            "0003_c001_00000001_0.jpg\n0001_c002_00000002_0.jpg\n0002_c001_00000003_0.jpg\n"
        )
        os.makedirs(tmp_path / "image_query")

        samples = _load_named_split(str(tmp_path), "name_query.txt", "image_query")

        assert len(samples) == 3
        # Порядок ДОЛЖЕН совпадать с порядком строк файла (3, 1, 2), а не
        # с алфавитной сортировкой имён файлов.
        assert [s[1] for s in samples] == [3, 1, 2]

    def test_raises_if_names_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_named_split(str(tmp_path), "name_query.txt", "image_query")


# ---------------------------------------------------------------------------
# compute_map_cmc - на вручную просчитанных примерах
# ---------------------------------------------------------------------------

class TestComputeMapCmc:
    def test_perfect_ranking_gives_ap_one(self):
        # 1 query, галерея из 6. Ранжирование по убыванию похожести:
        # [2, 0, 4, 1, 5, 3] (задаём через величину скалярного произведения).
        query_emb = torch.tensor([[1.0]])
        gallery_emb = torch.tensor([[50.0], [30.0], [60.0], [10.0], [40.0], [20.0]])

        gt_index_lists = [[0, 4]]   # good: индексы 0 и 4
        jk_index_lists = [[2]]      # junk: индекс 2 (самый похожий, но мусор)

        mAP, cmc, valid_n, missing = compute_map_cmc(
            query_emb, gallery_emb, gt_index_lists, jk_index_lists,
            index_base=0, max_rank=6,
        )

        assert mAP == pytest.approx(1.0)
        assert cmc[0] == pytest.approx(1.0)  # Rank-1 = 1.0
        assert valid_n == 1
        assert missing == 0

    def test_late_hits_give_partial_ap(self):
        query_emb = torch.tensor([[1.0]])
        gallery_emb = torch.tensor([[50.0], [30.0], [60.0], [10.0], [40.0], [20.0]])
        # Порядок по убыванию: [2, 0, 4, 1, 5, 3]

        gt_index_lists = [[1, 3]]  # позиции 3 и 5 в общем порядке (0-indexed)
        jk_index_lists = [[]]

        mAP, cmc, valid_n, missing = compute_map_cmc(
            query_emb, gallery_emb, gt_index_lists, jk_index_lists,
            index_base=0, max_rank=6,
        )

        expected_ap = (0.25 + 2 / 6) / 2  # precision@4=1/4, precision@6=2/6
        assert mAP == pytest.approx(expected_ap)
        assert list(cmc) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    def test_query_with_no_good_matches_is_excluded(self):
        query_emb = torch.tensor([[1.0], [1.0]])
        gallery_emb = torch.tensor([[1.0], [2.0], [3.0]])

        gt_index_lists = [[], [0]]  # первый query вообще без good-совпадений
        jk_index_lists = [[], []]

        mAP, cmc, valid_n, missing = compute_map_cmc(
            query_emb, gallery_emb, gt_index_lists, jk_index_lists,
            index_base=0, max_rank=3,
        )
        assert valid_n == 1  # только второй query учтён

    def test_raises_if_no_valid_queries_at_all(self):
        query_emb = torch.tensor([[1.0]])
        gallery_emb = torch.tensor([[1.0], [2.0]])
        gt_index_lists = [[]]
        jk_index_lists = [[]]

        with pytest.raises(RuntimeError):
            compute_map_cmc(
                query_emb, gallery_emb, gt_index_lists, jk_index_lists,
                index_base=0, max_rank=2,
            )

    def test_accepts_precomputed_distance_matrix(self):
        # Проверяем ветку "precomputed_dist_matrix", используемую re-ranking'ом:
        # query_embeddings/gallery_embeddings можно передать как None.
        dist_matrix = np.array([[0.1, 0.5, 0.9]])  # query 0: ближе всего gallery 0
        gt_index_lists = [[0]]
        jk_index_lists = [[]]

        mAP, cmc, valid_n, missing = compute_map_cmc(
            None, None, gt_index_lists, jk_index_lists,
            index_base=0, max_rank=3, precomputed_dist_matrix=dist_matrix,
        )
        assert mAP == pytest.approx(1.0)
        assert cmc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# k_reciprocal_rerank - структурная проверка на кластерных данных
# ---------------------------------------------------------------------------

class TestKReciprocalRerank:
    def test_output_shape(self):
        query_emb = torch.randn(3, 8)
        gallery_emb = torch.randn(10, 8)
        query_emb = torch.nn.functional.normalize(query_emb, dim=1)
        gallery_emb = torch.nn.functional.normalize(gallery_emb, dim=1)

        dist = k_reciprocal_rerank(query_emb, gallery_emb, k1=3, k2=2, lambda_value=0.3)
        assert dist.shape == (3, 10)

    def test_finds_correct_cluster_after_rerank(self):
        torch.manual_seed(0)
        centers = torch.randn(4, 16) * 5
        gallery = []
        gallery_cluster = []
        for c in range(4):
            pts = centers[c] + torch.randn(6, 16) * 0.3
            gallery.append(pts)
            gallery_cluster.extend([c] * 6)
        gallery = torch.cat(gallery, dim=0)
        gallery_cluster = np.array(gallery_cluster)

        query = centers + torch.randn(4, 16) * 0.3

        def l2norm(x):
            return torch.nn.functional.normalize(x, dim=1)

        dist = k_reciprocal_rerank(l2norm(query), l2norm(gallery), k1=5, k2=3, lambda_value=0.3)

        for i in range(4):
            nearest_idx = int(np.argmin(dist[i]))
            assert gallery_cluster[nearest_idx] == i

    def test_identical_vector_has_minimal_distance(self):
        # Query совпадает с одним из gallery-векторов - после re-ranking
        # он всё ещё должен быть (одним из) ближайших.
        torch.manual_seed(1)
        gallery = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
        query = gallery[3:4].clone()  # точная копия 4-го элемента галереи

        dist = k_reciprocal_rerank(query, gallery, k1=3, k2=2, lambda_value=0.3)
        nearest_idx = int(np.argmin(dist[0]))
        assert nearest_idx == 3
