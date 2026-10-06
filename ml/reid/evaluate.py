"""
Оценка обученной VehicleReIDNet на VeRi-776: mAP и CMC@k.

Что происходит в этом файле, по шагам:

    1. Загружаем чекпоинт модели.
    2. Загружаем query и gallery СТРОГО в том порядке, в котором имена
       файлов перечислены в name_query.txt / name_test.txt. Это критично:
       официальные файлы gt_index.txt и jk_index.txt ссылаются на НОМЕРА
       СТРОК именно в этих текстовых файлах, а не на алфавитный порядок
       файлов в папке (в общем случае они могут совпадать, но полагаться
       на это нельзя).
    3. Читаем gt_index.txt (какие индексы галереи - настоящие совпадения
       для каждого query) и jk_index.txt (какие индексы галереи считаются
       "мусором" для этого query и должны быть исключены из ранжирования
       перед подсчётом метрик - как правило, это снимки того же
       транспортного средства той же камерой, что и query).
    4. Извлекаем L2-нормализованные эмбеддинги для всех query и gallery
       изображений.
    5. Строим полную матрицу косинусных близостей query x gallery,
       ранжируем галерею для каждого query, вычитаем "мусорные" индексы
       из ранжирования (не разрывая порядок оставшихся), считаем AP и
       CMC по формулам (22)-(24) из обзорной статьи.

Пример запуска:

    python ml/evaluate.py \
        --data-root data/VeRi \
        --checkpoint ml/checkpoints/epoch_009.pt
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml.reid.datasets.veri_dataset import _parse_filename
from ml.reid.models.efficientnetv2_reid import VehicleReIDNet


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Аргументы командной строки
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка VehicleReIDNet на VeRi-776 (mAP/CMC)")

    parser.add_argument("--data-root", type=str, required=True,
                         help="Путь к корню VeRi-776 (там же, где name_query.txt, gt_index.txt и т.д.)")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Путь к чекпоинту (.pt), сохранённому train.py")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=50,
                         help="До какого ранга считать CMC (обычно достаточно 50)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--flip-tta", action="store_true",
                         help="Test-time augmentation: усреднять эмбеддинг оригинала "
                              "и горизонтально отражённой версии изображения")

    parser.add_argument("--rerank", action="store_true",
                         help="Включить k-reciprocal re-ranking (Zhong et al., 2017) "
                              "поверх обычного косинусного ранжирования")
    parser.add_argument("--rerank-k1", type=int, default=20,
                         help="k1 - размер окрестности для k-reciprocal множества")
    parser.add_argument("--rerank-k2", type=int, default=6,
                         help="k2 - размер окна для локального query expansion")
    parser.add_argument("--rerank-lambda", type=float, default=0.3,
                         help="Вес исходного расстояния в финальной смеси "
                              "(1-lambda)*Jaccard + lambda*original")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Загрузка query/gallery в порядке официальных файлов name_*.txt
# ---------------------------------------------------------------------------

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_named_split(
    root: str, names_filename: str, image_subdir: str
) -> List[Tuple[str, int, int]]:
    """
    Возвращает список (путь_к_файлу, pid, camid) СТРОГО в порядке строк
    names_filename (например, name_query.txt). Порядок обязателен, т.к.
    gt_index.txt/jk_index.txt ссылаются на номера строк этого файла.
    """
    names_path = os.path.join(root, names_filename)
    if not os.path.isfile(names_path):
        raise FileNotFoundError(
            f"Не найден {names_path}. Ожидается, что этот файл лежит "
            f"в корне VeRi-776 рядом с папками image_query/image_test/."
        )

    names = _read_lines(names_path)
    samples = []
    for name in names:
        pid, camid = _parse_filename(name)
        path = os.path.join(root, image_subdir, name)
        samples.append((path, pid, camid))
    return samples


# ---------------------------------------------------------------------------
# Парсинг gt_index.txt / jk_index.txt
# ---------------------------------------------------------------------------

def _parse_index_file(path: str, expected_num_lines: int) -> List[List[int]]:
    """
    Парсит файл вида gt_index.txt/jk_index.txt: одна строка на query,
    в строке - индексы галереи (разделённые пробелами и/или запятыми).

    Индексы в исходном файле оставляются КАК ЕСТЬ (без вычитания смещения
    1-based/0-based) - выбор смещения делает _detect_index_base ниже,
    т.к. заранее нельзя быть на 100% уверенным, какой конвенции
    придерживается конкретный релиз датасета.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Не найден {path}.")

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    result = []
    for line in raw_lines:
        tokens = re.split(r"[\s,]+", line.strip())
        idxs = [int(t) for t in tokens if t]
        result.append(idxs)

    if len(result) != expected_num_lines:
        raise ValueError(
            f"{path}: ожидалось {expected_num_lines} строк (по числу query "
            f"в name_query.txt), но найдено {len(result)}. Формат файла, "
            f"похоже, отличается от ожидаемого — откройте его вручную "
            f"(например, командой `head -3 {path}`) и проверьте разделители."
        )

    return result


def _detect_index_base(
    index_lists: List[List[int]],
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    sample_size: int = 50,
) -> int:
    """
    Автоматически определяет, 1-based или 0-based индексация используется
    в gt_index.txt, сверяя pid галерейного изображения по указанному
    индексу с pid соответствующего query (для gt_index.txt эти pid ДОЛЖНЫ
    совпадать, если индексация угадана верно).

    Это подстраховка на случай, если конкретный релиз VeRi-776 использует
    не ту конвенцию индексации, которую мы предполагаем по умолчанию
    (1-based, как принято в оригинальном MATLAB-тулките датасета).
    """

    def agreement(offset: int) -> float:
        total, correct = 0, 0
        for i in range(min(sample_size, len(index_lists))):
            for raw_idx in index_lists[i]:
                pos = raw_idx - offset
                if 0 <= pos < len(gallery_pids):
                    total += 1
                    if gallery_pids[pos] == query_pids[i]:
                        correct += 1
        return correct / total if total > 0 else 0.0

    agreement_1based = agreement(1)
    agreement_0based = agreement(0)

    print(
        f"  Проверка индексации: совпадение pid при 1-based = "
        f"{agreement_1based:.1%}, при 0-based = {agreement_0based:.1%}"
    )

    if max(agreement_1based, agreement_0based) < 0.5:
        print(
            "  ПРЕДУПРЕЖДЕНИЕ: совпадение pid низкое для ОБОИХ вариантов "
            "индексации. Результаты mAP/CMC могут быть некорректны. "
            "Откройте gt_index.txt вручную и сверьтесь с реальным форматом."
        )

    return 1 if agreement_1based >= agreement_0based else 0


# ---------------------------------------------------------------------------
# Извлечение эмбеддингов
# ---------------------------------------------------------------------------

def build_eval_transform(image_size) -> transforms.Compose:
    """
    Трансформации для инференса: БЕЗ аугментаций (детерминированный вход),
    только resize + нормализация под статистику ImageNet backbone'а.
    """
    h, w = image_size
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class EvalImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, pid, camid = self.samples[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, pid, camid


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    samples: List[Tuple[str, int, int]],
    transform,
    device: str,
    batch_size: int,
    num_workers: int,
    flip_tta: bool = False,
) -> torch.Tensor:
    """
    Прогоняет все изображения через модель, возвращает [N, D] эмбеддинги.

    flip_tta: если True, для каждого изображения дополнительно считается
        эмбеддинг его горизонтально отражённой версии, оба усредняются и
        результат заново L2-нормализуется. Отражение выполняется прямо
        на GPU-тензоре батча (torch.flip по оси ширины), без повторного
        прохода через Dataset/DataLoader - дешевле и не требует отдельной
        трансформации.
    """
    dataset = EvalImageDataset(samples, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.eval()
    all_embeddings = []
    for images, _pids, _camids in loader:
        images = images.to(device, non_blocking=True)
        _, _, norm_embedding = model(images)

        if flip_tta:
            flipped = torch.flip(images, dims=[3])  # отражение по ширине (NCHW)
            _, _, norm_embedding_flipped = model(flipped)
            combined = norm_embedding + norm_embedding_flipped
            norm_embedding = torch.nn.functional.normalize(combined, p=2, dim=1)

        all_embeddings.append(norm_embedding.cpu())

    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# k-reciprocal re-ranking (Zhong et al., 2017)
# ---------------------------------------------------------------------------

def _euclidean_dist_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Матрица попарных евклидовых расстояний [N, M] между x [N,D] и y [M,D].
    Для L2-нормализованных векторов монотонно связана с косинусной
    близостью (тот же принцип, что и в ml/losses/triplet.py).
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x_sq = np.sum(x ** 2, axis=1, keepdims=True)
    y_sq = np.sum(y ** 2, axis=1, keepdims=True)
    dist_sq = x_sq + y_sq.T - 2.0 * x @ y.T
    dist_sq = np.clip(dist_sq, 0.0, None)
    return np.sqrt(dist_sq)


def _k_reciprocal_neigh(initial_rank: np.ndarray, i: int, k: int) -> np.ndarray:
    """
    k-reciprocal множество для элемента i: те, кто входит в top-(k+1)
    соседей i, И для кого i сам входит в их top-(k+1) соседей (взаимность).
    """
    forward = initial_rank[i, : k + 1]
    backward = initial_rank[forward, : k + 1]
    fi = np.where(backward == i)[0]
    return forward[fi]


def k_reciprocal_rerank(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    k-reciprocal re-ranking (Zhong et al., "Re-ranking Person
    Re-identification with k-reciprocal Encoding", CVPR 2017).

    Идея: обычное косинусное ранжирование смотрит только на близость
    query-gallery. Здесь дополнительно учитывается структура соседства
    ВНУТРИ галереи: если два изображения взаимно оказываются в топе
    соседей друг друга (k-reciprocal), это более надёжный сигнал
    совпадения, чем просто малое расстояние.

    Возвращает итоговую матрицу расстояний [num_query, num_gallery] -
    чем меньше значение, тем более вероятное совпадение (в отличие от
    sim_matrix в compute_map_cmc, здесь сразу расстояние, не сходство).

    ПРЕДУПРЕЖДЕНИЕ ПО ПАМЯТИ/ВРЕМЕНИ: алгоритм строит полную матрицу
    расстояний размера (num_query+num_gallery)^2. Для VeRi-776
    (query=1678, gallery=11579) это ~13257^2 float32 ~ 700 МБ на матрицу,
    и вычисление может занять несколько минут на CPU. Это нормально для
    офлайн-оценки, но для продакшен-инференса такой полный пересчёт
    неприменим - там re-ranking обычно ограничивают top-N кандидатами.
    """
    query_np = query_embeddings.numpy() if isinstance(query_embeddings, torch.Tensor) else query_embeddings
    gallery_np = gallery_embeddings.numpy() if isinstance(gallery_embeddings, torch.Tensor) else gallery_embeddings

    num_query = query_np.shape[0]
    num_gallery = gallery_np.shape[0]
    feats = np.concatenate([query_np, gallery_np], axis=0).astype(np.float32)
    n = feats.shape[0]

    original_dist = _euclidean_dist_matrix(feats, feats)
    # Построчная нормализация в [0,1] - облегчает подбор весов на
    # следующих шагах (как в оригинальной статье).
    original_dist = original_dist / (original_dist.max(axis=1, keepdims=True) + 1e-12)
    initial_rank = np.argsort(original_dist, axis=1)

    # --- Строим k-reciprocal encoding V ---
    v = np.zeros((n, n), dtype=np.float32)
    half_k1 = int(round(k1 / 2.0))
    for i in range(n):
        k_reciprocal_idx = _k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_idx = k_reciprocal_idx.copy()

        # Локальное расширение множества: если у кандидата его собственное
        # (более узкое, half_k1) k-reciprocal множество почти целиком
        # лежит внутри множества i - тоже присоединяем его целиком.
        for candidate in k_reciprocal_idx:
            candidate_k_reciprocal_idx = _k_reciprocal_neigh(initial_rank, candidate, half_k1)
            overlap = np.intersect1d(candidate_k_reciprocal_idx, k_reciprocal_idx)
            if len(overlap) > 0.6667 * len(candidate_k_reciprocal_idx):
                k_reciprocal_expansion_idx = np.append(
                    k_reciprocal_expansion_idx, candidate_k_reciprocal_idx
                )

        k_reciprocal_expansion_idx = np.unique(k_reciprocal_expansion_idx)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_idx])
        v[i, k_reciprocal_expansion_idx] = weight / np.sum(weight)

    # --- Локальное query expansion: сглаживаем V по k2 ближайшим соседям ---
    if k2 > 1:
        v_qe = np.zeros_like(v)
        for i in range(n):
            v_qe[i, :] = np.mean(v[initial_rank[i, :k2], :], axis=0)
        v = v_qe
        del v_qe

    # --- Jaccard-расстояние между query и gallery через V ---
    # Используем разреженность V (у каждой строки лишь малая доля
    # ненулевых столбцов) через инвертированный индекс, иначе плотное
    # произведение N x N слишком дорого по памяти.
    inv_index = [np.where(v[:, j] != 0)[0] for j in range(n)]
    row_sums = v.sum(axis=1)  # sum_k V[i,k] для всех строк, считаем один раз

    jaccard_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    for i in range(num_query):
        temp_min = np.zeros(n, dtype=np.float32)
        nonzero_k = np.where(v[i, :] != 0)[0]
        for k in nonzero_k:
            j_list = inv_index[k]
            temp_min[j_list] += np.minimum(v[i, k], v[j_list, k])

        denom = row_sums[i] + row_sums - temp_min
        jaccard_dist[i, :] = 1.0 - temp_min[num_query:] / (denom[num_query:] + 1e-12)

    original_dist_qg = original_dist[:num_query, num_query:]
    final_dist = jaccard_dist * (1.0 - lambda_value) + original_dist_qg * lambda_value

    return final_dist


# ---------------------------------------------------------------------------
# mAP / CMC
# ---------------------------------------------------------------------------

def compute_map_cmc(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gt_index_lists: List[List[int]],
    jk_index_lists: List[List[int]],
    index_base: int,
    max_rank: int = 50,
    precomputed_dist_matrix: np.ndarray = None,
) -> Tuple[float, np.ndarray, int, int]:
    """
    Считает mAP и CMC-кривую по официальному протоколу VeRi-776
    (junk-фильтрация по jk_index.txt, good-совпадения по gt_index.txt).

    Если precomputed_dist_matrix не передан, ранжирование строится по
    обычной косинусной близости L2-нормализованных эмбеддингов (query
    @ gallery.T). Если передан (например, результат k_reciprocal_rerank),
    используется он - тогда query_embeddings/gallery_embeddings можно
    передать как None, они в этом случае не используются.

    Returns:
        mAP: усреднённый Average Precision по всем валидным query.
        cmc_curve: массив [max_rank], cmc_curve[k] = доля query, для
            которых верное совпадение найдено в пределах топ-(k+1).
        valid_query_count: сколько query реально участвовало в подсчёте
            (у части query может не оказаться валидных good-совпадений
            после фильтрации - такие query пропускаются, как и
            предписывает стандартный протокол).
        num_missing_good: сколько раз "good"-индекс из gt_index.txt не
            нашёл соответствия по pid в галерее (диагностика на случай
            неверно угаданной индексации).
    """
    if precomputed_dist_matrix is not None:
        # Меньше = более вероятное совпадение (обычное расстояние).
        dist_matrix = precomputed_dist_matrix
    else:
        # L2-нормализованные эмбеддинги -> скалярное произведение = косинусная
        # близость. Берём -sim, чтобы везде далее работать в терминах
        # "меньше = лучше", единообразно со случаем re-ranking выше.
        sim_matrix = (query_embeddings @ gallery_embeddings.t()).numpy()
        dist_matrix = -sim_matrix

    num_query = dist_matrix.shape[0]
    num_gallery = dist_matrix.shape[1]

    all_ap = []
    cmc_curve = np.zeros(max_rank)
    valid_query_count = 0
    num_missing_good = 0

    for i in range(num_query):
        good_set = {idx - index_base for idx in gt_index_lists[i]}
        junk_set = {idx - index_base for idx in jk_index_lists[i]}

        # Отбрасываем индексы, вышедшие за пределы галереи (защита от
        # некорректной индексации/повреждённых строк).
        good_set = {idx for idx in good_set if 0 <= idx < num_gallery}
        junk_set = {idx for idx in junk_set if 0 <= idx < num_gallery}

        order = np.argsort(dist_matrix[i])  # индексы галереи по возрастанию расстояния

        keep_mask = np.array([idx not in junk_set for idx in order])
        filtered_order = order[keep_mask]

        match = np.array([1 if idx in good_set else 0 for idx in filtered_order])

        if len(good_set) == 0 or match.sum() == 0:
            # Нет валидных good-совпадений для этого query после
            # фильтрации - пропускаем (так поступает и стандартный
            # протокол Market1501/VeRi-776 в подобных случаях).
            continue

        if match.sum() < len(good_set):
            num_missing_good += 1

        valid_query_count += 1

        # --- CMC ---
        cmc = np.cumsum(match)
        cmc[cmc > 1] = 1
        cmc = cmc[:max_rank]
        cmc_curve[: len(cmc)] += cmc
        if len(cmc) < max_rank:
            cmc_curve[len(cmc):] += cmc[-1]

        # --- AP (уравнения 22-24 обзорной статьи) ---
        hit_positions = np.where(match == 1)[0]
        precisions = [(k + 1) / (pos + 1) for k, pos in enumerate(hit_positions)]
        all_ap.append(float(np.mean(precisions)))

    if valid_query_count == 0:
        raise RuntimeError(
            "Ни один query не дал валидных ground-truth совпадений. "
            "Скорее всего, неверно определена индексация gt_index.txt "
            "(1-based/0-based) или файлы не соответствуют этому query/gallery."
        )

    mAP = float(np.mean(all_ap))
    cmc_curve = cmc_curve / valid_query_count

    return mAP, cmc_curve, valid_query_count, num_missing_good


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = args.device

    print(f"Загружаю чекпоинт: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_train_pids = checkpoint["num_train_pids"]
    embedding_dim = checkpoint.get("embedding_dim", 512)

    model = VehicleReIDNet(
        num_classes=num_train_pids, embedding_dim=embedding_dim, pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Эпоха чекпоинта: {checkpoint.get('epoch', '?')}")

    print("Загружаю query/gallery (в порядке name_query.txt / name_test.txt)...")
    query_samples = _load_named_split(args.data_root, "name_query.txt", "image_query")
    gallery_samples = _load_named_split(args.data_root, "name_test.txt", "image_test")
    print(f"  query: {len(query_samples)}, gallery: {len(gallery_samples)}")

    query_pids = np.array([s[1] for s in query_samples])
    gallery_pids = np.array([s[1] for s in gallery_samples])

    print("Читаю gt_index.txt / jk_index.txt...")
    gt_index_lists = _parse_index_file(
        os.path.join(args.data_root, "gt_index.txt"), len(query_samples)
    )
    jk_index_lists = _parse_index_file(
        os.path.join(args.data_root, "jk_index.txt"), len(query_samples)
    )

    index_base = _detect_index_base(gt_index_lists, query_pids, gallery_pids)
    print(f"  Используется индексация: {'1-based' if index_base == 1 else '0-based'}")

    transform = build_eval_transform(args.image_size)

    print("Извлекаю эмбеддинги query...")
    query_embeddings = extract_embeddings(
        model, query_samples, transform, device, args.batch_size, args.num_workers,
        flip_tta=args.flip_tta,
    )
    print("Извлекаю эмбеддинги gallery...")
    gallery_embeddings = extract_embeddings(
        model, gallery_samples, transform, device, args.batch_size, args.num_workers,
        flip_tta=args.flip_tta,
    )
    if args.flip_tta:
        print("  (flip-TTA включён: эмбеддинг усреднён с горизонтально отражённой версией)")

    if args.rerank:
        print(
            f"Считаю k-reciprocal re-ranking (k1={args.rerank_k1}, "
            f"k2={args.rerank_k2}, lambda={args.rerank_lambda})... "
            f"это может занять несколько минут на большой галерее."
        )
        dist_matrix = k_reciprocal_rerank(
            query_embeddings, gallery_embeddings,
            k1=args.rerank_k1, k2=args.rerank_k2, lambda_value=args.rerank_lambda,
        )
        print("Считаю mAP/CMC (по re-ranked дистанциям)...")
        mAP, cmc_curve, valid_query_count, num_missing_good = compute_map_cmc(
            None, None, gt_index_lists, jk_index_lists, index_base,
            max_rank=args.max_rank, precomputed_dist_matrix=dist_matrix,
        )
    else:
        print("Считаю mAP/CMC...")
        mAP, cmc_curve, valid_query_count, num_missing_good = compute_map_cmc(
            query_embeddings,
            gallery_embeddings,
            gt_index_lists,
            jk_index_lists,
            index_base,
            max_rank=args.max_rank,
        )

    print()
    print(f"Учтено query: {valid_query_count} / {len(query_samples)}")
    if num_missing_good > 0:
        print(
            f"  (В {num_missing_good} query часть good-индексов из gt_index.txt "
            f"не нашла совпадения по pid в галерее - возможно, стоит "
            f"перепроверить формат файла)"
        )
    print(f"mAP:    {mAP * 100:.2f}%")
    for r in (1, 5, 10, 20):
        if r <= args.max_rank:
            print(f"Rank-{r}: {cmc_curve[r - 1] * 100:.2f}%")


if __name__ == "__main__":
    main()
