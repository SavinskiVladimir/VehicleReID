"""
Скрипт обучения VehicleReIDNet на датасете VeRi-776.

Что происходит в этом файле, по шагам:

    1. Парсим аргументы командной строки (путь к данным, гиперпараметры).
    2. Строим VeRi776 (train/query/gallery сплиты из ml/datasets/veri_dataset.py).
    3. Оборачиваем dataset.train в VeRiImageDataset с аугментациями.
    4. Строим PKSampler на основе pid-ов train-сплита — каждый батч будет
       содержать P идентичностей по K изображений (нужно для
       Batch-Hard Triplet Loss).
    5. Создаём DataLoader на основе этого сэмплера.
    6. Строим модель VehicleReIDNet, две функции потерь (CE + Triplet),
       оптимизатор и scheduler.
    7. Обучаем нужное число эпох, логируем метрики, сохраняем чекпоинты.

Пример запуска:

    python ml/train.py \
        --data-root data/VeRi \
        --epochs 60 \
        --num-pids-per-batch 8 \
        --num-instances 4 \
        --lr 3.5e-4
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.veri_dataset import VeRi776, VeRiImageDataset
from samplers.pk_sampler import PKSampler
from models.efficientnetv2_reid import VehicleReIDNet
from losses.triplet import BatchHardTripletLoss


# ---------------------------------------------------------------------------
# Аргументы командной строки
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение VehicleReIDNet на VeRi-776")

    # Данные
    parser.add_argument("--data-root", type=str, required=True,
                         help="Путь к корню VeRi-776 (папка с image_train/, image_query/, image_test/)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                         help="Размер входного изображения (H W) после resize")

    # Батчинг по схеме P x K (см. pk_sampler.py)
    parser.add_argument("--num-pids-per-batch", type=int, default=8,
                         help="P: число идентичностей на батч")
    parser.add_argument("--num-instances", type=int, default=4,
                         help="K: число изображений на идентичность в батче")
    parser.add_argument("--num-workers", type=int, default=4,
                         help="Число процессов для DataLoader")

    # Обучение
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[30, 50],
                         help="Эпохи, на которых lr умножается на --lr-gamma")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplet-loss-weight", type=float, default=1.0,
                         help="Вес triplet loss относительно CE loss в общей сумме")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-pretrained", action="store_true",
                         help="Не использовать ImageNet-претрейн для backbone")

    # Прочее
    parser.add_argument("--checkpoint-dir", type=str, default="ml/checkpoints")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=(
            "Куда torch.hub должен скачивать предобученные веса backbone "
            "(EfficientNetV2-S). По умолчанию используется системный кэш "
            "torch (~/.cache/torch на Linux/macOS, C:\\Users\\<Имя>\\.cache\\torch "
            "на Windows). Укажите этот параметр, если системный диск "
            "ограничен в объёме или используется временный/сетевой профиль."
        ),
    )
    parser.add_argument("--save-every", type=int, default=10,
                         help="Сохранять чекпоинт каждые N эпох")
    parser.add_argument("--log-every", type=int, default=20,
                         help="Печатать статистику лосса каждые N итераций")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Аугментации
# ---------------------------------------------------------------------------

# Статистики нормализации ImageNet — используем их, т.к. backbone
# (EfficientNetV2-S) предобучен на ImageNet, и вход должен быть
# в том же распределении, к которому привык backbone.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size) -> transforms.Compose:
    """
    Аугментации для обучения.

    RandomHorizontalFlip — вагоны/машины симметричны по горизонтали
        в большинстве ракурсов, флип валиден и увеличивает разнообразие.
    Pad + RandomCrop — небольшой сдвиг кадрирования, имитирует неточность
        bounding box от YOLO-детектора (в реальном пайплайне padding
        боксов тоже не идеален).
    RandomErasing — случайно "стирает" прямоугольную область изображения.
        Для ReID это особенно полезно: приучает модель не полагаться на
        одну заметную деталь (например, номер или наклейку), что имитирует
        реальные окклюзии (грязь, другие машины в кадре, тени).
    ColorJitter — небольшие вариации яркости/контраста имитируют разные
        условия освещения между камерами.
    """
    h, w = image_size
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


# ---------------------------------------------------------------------------
# Одна эпоха обучения
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.Module,
    triplet_loss_fn: nn.Module,
    triplet_weight: float,
    device: str,
    epoch: int,
    log_every: int,
) -> None:
    model.train()

    running_ce = 0.0
    running_triplet = 0.0
    running_correct = 0
    running_total = 0
    t_start = time.time()

    for it, (images, pids, camids, _paths) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        pids = pids.to(device, non_blocking=True)

        # embedding      -> "сырой" вектор до BatchNorm, используется для triplet
        # logits         -> выход classification head, для CE loss
        # norm_embedding -> L2-нормализованный вектор
        embedding, logits, norm_embedding = model(images)

        ce_loss = ce_loss_fn(logits, pids)
        triplet_loss = triplet_loss_fn(norm_embedding, pids)

        loss = ce_loss + triplet_weight * triplet_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Метрики для логирования
        running_ce += ce_loss.item()
        running_triplet += triplet_loss.item()
        preds = logits.argmax(dim=1)
        running_correct += (preds == pids).sum().item()
        running_total += pids.size(0)

        if (it + 1) % log_every == 0:
            avg_ce = running_ce / (it + 1)
            avg_triplet = running_triplet / (it + 1)
            acc = running_correct / running_total
            elapsed = time.time() - t_start
            print(
                f"[epoch {epoch}] iter {it + 1}/{len(dataloader)} | "
                f"CE: {avg_ce:.4f} | Triplet: {avg_triplet:.4f} | "
                f"ID-acc: {acc:.3f} | {elapsed:.1f}s"
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Перенаправляем кэш torch.hub (сюда скачиваются предобученные веса
    # EfficientNetV2-S) в кастомную папку, если она указана. Это нужно
    # сделать ДО создания модели (см. ниже VehicleReIDNet(...)), т.к.
    # именно в момент создания backbone torchvision полезет качать
    # веса, если их ещё нет в кэше.
    if args.cache_dir is not None:
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.hub.set_dir(args.cache_dir)
        print(f"Кэш torch.hub перенаправлен в: {args.cache_dir}")

    # --- 1. Датасет ---
    print("Загружаю VeRi-776...")
    dataset = VeRi776(args.data_root)
    print(dataset.summary())

    train_transform = build_train_transform(args.image_size)
    train_dataset = VeRiImageDataset(dataset.train, transform=train_transform)

    # --- 2. PK-сэмплер ---
    # pids должны быть в том же порядке, что и индексы train_dataset,
    # поэтому берём их напрямую из dataset.train (список VeRiSample),
    # а не пересчитываем откуда-то ещё.
    train_pids = [sample.pid for sample in dataset.train]

    pk_sampler = PKSampler(
        pids=train_pids,
        num_pids_per_batch=args.num_pids_per_batch,
        num_instances=args.num_instances,
    )
    batch_size = pk_sampler.batch_size

    # --- 3. DataLoader ---
    # ВАЖНО: используем sampler=pk_sampler (а НЕ shuffle=True) и
    # batch_size, равный pk_sampler.batch_size = P * K.
    #
    # Как это работает вместе: PKSampler.__iter__() уже отдаёт индексы
    # СГРУППИРОВАННЫМИ по P*K (то есть первые P*K элементов — это ровно
    # один валидный PK-батч, следующие P*K — следующий батч, и т.д.).
    # DataLoader под капотом оборачивает переданный sampler в
    # `torch.utils.data.BatchSampler(sampler, batch_size, drop_last)`,
    # который просто нарезает поток индексов от sampler кусками по
    # batch_size элементов подряд, СОХРАНЯЯ ИХ ПОРЯДОК. Так как наш
    # PKSampler уже гарантирует, что каждый кусок такого размера — это
    # корректный PK-батч, итоговая нарезка BatchSampler'ом совпадает
    # с нарезкой, которую задумывал PKSampler.
    #
    # Если бы мы использовали `batch_sampler=` с СОБСТВЕННЫМ классом,
    # пришлось бы вручную группировать индексы в списки внутри самого
    # сэмплера — здесь в этом нет необходимости, т.к. PKSampler и так
    # выдаёт их в нужном порядке, и достаточно синхронизировать
    # batch_size у DataLoader с batch_size у PKSampler.
    #
    # drop_last=True на случай, если наберётся "хвост" из индексов
    # меньше batch_size (в текущей реализации PKSampler такого не
    # бывает, т.к. в final_idxs добавляются только полные PK-группы,
    # но drop_last оставлен для безопасности при будущих изменениях
    # логики сэмплера).
    train_loader = DataLoader(
        train_dataset,
        sampler=pk_sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(args.device == "cuda"),
    )

    print(
        f"DataLoader готов: batch_size={batch_size} "
        f"(P={args.num_pids_per_batch} x K={args.num_instances}), "
        f"~{len(pk_sampler) // batch_size} батчей на эпоху (оценка)"
    )

    # --- 4. Модель ---
    model = VehicleReIDNet(
        num_classes=dataset.num_train_pids,
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
        dropout_p=args.dropout,
    ).to(args.device)

    # --- 5. Функции потерь ---
    # label_smoothing встроен в nn.CrossEntropyLoss начиная с PyTorch 1.10+.
    # Он "размазывает" часть вероятностной массы с истинного класса на
    # остальные, что снижает переуверенность модели и улучшает обобщение —
    # особенно полезно при относительно небольшом числе примеров на класс,
    # как в VeRi-776 (в среднем ~64 изображения на идентичность).
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    triplet_loss_fn = BatchHardTripletLoss(margin=args.triplet_margin)

    # --- 6. Оптимизатор и scheduler ---
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # MultiStepLR: lr умножается на lr_gamma на заданных эпохах (--milestones).
    # Стандартная практика для ReID-бейзлайнов (например, "Bag of Tricks").
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)

    # --- 7. Цикл обучения ---
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            triplet_loss_fn=triplet_loss_fn,
            triplet_weight=args.triplet_loss_weight,
            device=args.device,
            epoch=epoch,
            log_every=args.log_every,
        )
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"== Эпоха {epoch}/{args.epochs} завершена. Текущий lr: {current_lr:.2e} ==")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_train_pids": dataset.num_train_pids,
                    "embedding_dim": args.embedding_dim,
                },
                checkpoint_path,
            )
            print(f"Чекпоинт сохранён: {checkpoint_path}")


if __name__ == "__main__":
    main()