"""dataset.py — загрузка и аугментация данных для сегментации облаков."""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> list[Path]:
    """Отсортированный список путей до изображений в папке (без рекурсии)."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Читает изображение в RGB uint8 (включая 16-битные TIFF -> нормировка)."""
    from utils import imread_unicode
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не удалось прочитать изображение: {path}")
    if img.dtype == np.uint16:
        img = (img / 256.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> np.ndarray:
    """Читает маску в градациях серого; >127 -> класс 1 (облако)."""
    from utils import imread_unicode
    m = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Не удалось прочитать маску: {path}")
    if m.dtype == np.uint16:
        m = (m / 256.0).astype(np.uint8)
    return (m > 127).astype(np.uint8)


def apply_haze(image: np.ndarray, max_alpha: float = 0.45) -> np.ndarray:
    """Синеватая атмосферная дымка (домен-адаптация под камеры кубсатов)."""
    a = random.uniform(0.05, max_alpha)
    haze_color = np.array([240, 232, 225], dtype=np.float32)
    hazy = image.astype(np.float32) * (1.0 - a) + haze_color[None, None, :] * a
    noise = np.random.normal(0, 3.0, hazy.shape).astype(np.float32)
    return np.clip(hazy + noise, 0, 255).astype(np.uint8)


def get_augmentations(use_albumentations: bool = True):
    """Возвращает функция(image, mask) -> (image, mask) одинакового размера."""
    if use_albumentations:
        try:
            import albumentations as A

            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                     val_shift_limit=20, p=0.3),
            ])
        except ImportError:
            print("[dataset] albumentations не установлен — numpy-аугментации.")

    def numpy_augment(image: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        k = random.choice([0, 1, 2, 3])
        if k:
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        f = 1.0 + random.uniform(-0.1, 0.1)
        image = np.clip(image.astype(np.float32) * f, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    return numpy_augment


class CloudSegmentationDataset(Dataset):
    """Пары изображение+маска из папок images/ и masks/ (одинаковые имена файлов)."""

    def __init__(self, images_dir: str | Path, masks_dir: str | Path,
                 img_size: int = 384, augment: bool = False,
                 use_albumentations: bool = True,
                 haze_aug: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment
        self.haze_aug = haze_aug
        self.aug_fn = get_augmentations(use_albumentations) if augment else None

        img_names = {p.name for p in list_images(self.images_dir)}
        mask_names = {p.name for p in list_images(self.masks_dir)}
        img_stems = {p.stem: p for p in list_images(self.images_dir)}
        mask_stems = {p.stem: p for p in list_images(self.masks_dir)}
        common = sorted(set(img_stems) & set(mask_stems))
        if not common and (img_names & mask_names):
            common = sorted(p.stem for p in list_images(self.images_dir)
                            if p.name in mask_names)
        self.pairs = [(img_stems[s], mask_stems[s]) for s in common]
        if not self.pairs:
            raise RuntimeError(
                f"Не найдено пар изображение+маска: {images_dir} / {masks_dir}. "
                f"Убедитесь, что файлы имеют одинаковые имена (без расширения).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = load_image_rgb(img_path)
        mask = load_mask(mask_path)

        if self.augment and self.aug_fn is not None:
            out = self.aug_fn(image=image, mask=mask)
            image, mask = out["image"], out["mask"] if isinstance(out, dict) else out
            if self.haze_aug and random.random() < 0.4:
                image = apply_haze(image)

        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(mask)).float()
        return x, y


def prepare_38cloud(parquet_dir: str | Path, out_dir: str | Path,
                    val_ratio: float = 0.1, limit: int | None = None,
                    seed: int = 42) -> None:
    """Распаковывает parquet-шарды 38-Cloud в PNG-папки train/val."""
    import pyarrow.parquet as pq

    parquet_dir, out_dir = Path(parquet_dir), Path(out_dir)
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"В {parquet_dir} нет .parquet файлов")

    rng = random.Random(seed)
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    total = 0
    from PIL import Image
    for shard in shards:
        table = pq.read_table(shard, columns=["image", "label"])
        n = table.num_rows
        for i in range(n):
            img_cell = table.column("image")[i].as_py()
            lbl_cell = table.column("label")[i].as_py()
            img_bytes = img_cell["bytes"] if isinstance(img_cell, dict) else img_cell
            lbl_bytes = lbl_cell["bytes"] if isinstance(lbl_cell, dict) else lbl_cell

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            lbl = Image.open(io.BytesIO(lbl_bytes)).convert("L")

            split = "val" if rng.random() < val_ratio else "train"
            name = f"{shard.stem[:16]}_{i:05d}.png"
            img.save(out_dir / split / "images" / name)
            lbl.save(out_dir / split / "masks" / name)
            total += 1
            if limit is not None and total >= limit:
                break
            if total % 1000 == 0:
                print(f"  ... распаковано {total}")
        if limit is not None and total >= limit:
            break

    n_train = len(list_images(out_dir / "train" / "images"))
    n_val = len(list_images(out_dir / "val" / "images"))
    print(f"Готово: {total} пар (train={n_train}, val={n_val}) -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка данных")
    parser.add_argument("--prepare-38cloud", action="store_true",
                        help="распаковать parquet-шарды 38-Cloud в PNG train/val")
    parser.add_argument("--parquet-dir", default="data/raw/38cloud")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None,
                        help="ограничить число пар (для быстрых экспериментов)")
    args = parser.parse_args()

    if args.prepare_38cloud:
        prepare_38cloud(args.parquet_dir, args.out_dir, args.val_ratio, args.limit)
    else:
        parser.print_help()
