"""inspect_data.py — инвентаризация скачанных данных."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from dataset import IMAGE_EXTS, list_images


def inspect_folder(folder: Path, show: int = 0, out_dir: Path | None = None) -> None:
    files = list_images(folder)
    if not files:
        print(f"[inspect] В {folder} нет изображений")
        return

    ext_counter = Counter(p.suffix.lower() for p in files)
    print(f"[inspect] Папка: {folder}")
    print(f"  Файлов: {len(files)}  |  форматы: {dict(ext_counter)}")
    print()

    rows = []
    for p in files:
        from utils import imread_unicode
        img = imread_unicode(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [ОШИБКА ЧТЕНИЯ] {p.name}")
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bright_and_desat = np.logical_and(gray > 200,
                                          hsv[:, :, 1] < 60)
        naive_cloud_pct = float(bright_and_desat.mean() * 100)
        rows.append((p.name, w, h, mean_brightness, naive_cloud_pct))

    print(f"{'Файл':<44}{'Размер':<14}{'Яркость':<10}{'Наивный % обл.':<15}")
    print("-" * 83)
    for name, w, h, br, cp in rows:
        print(f"{name:<44}{f'{w}x{h}':<14}{br:<10.1f}{cp:<15.1f}")

    if rows:
        cps = [r[4] for r in rows]
        print("-" * 83)
        print(f"Наивная эвристика (яркость>200, насыщ.<60): "
              f"мин {min(cps):.1f}%, макс {max(cps):.1f}%, среднее {np.mean(cps):.1f}%")
        print("  Важно: эвристика путает облака со снегом, льдом, дымкой и "
              "пересветами —\n  именно поэтому в кейсе используется нейросетевая сегментация.")

    if show > 0 and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in files[:show]:
            from utils import imread_unicode
            img = imread_unicode(p, cv2.IMREAD_COLOR)
            small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
            from utils import imwrite_unicode
            imwrite_unicode(out_dir / f"preview_{p.stem}.jpg", small)
        print(f"\n[inspect] Превью сохранены в {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инвентаризация изображений")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--show", type=int, default=0,
                        help="сохранить превью первых N файлов")
    parser.add_argument("--out", default="outputs/inspect_previews")
    args = parser.parse_args()
    inspect_folder(Path(args.dir), args.show, Path(args.out) if args.show else None)
