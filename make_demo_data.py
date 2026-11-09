"""make_demo_data.py — генератор синтетических «спутниковых» снимков с облаками."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def fbm_noise(shape: tuple[int, int], octaves: int = 5, seed: int = 0) -> np.ndarray:
    """Фрактальный шум (сумма сглаженных случайных полей разных масштабов)."""
    rng = np.random.default_rng(seed)
    result = np.zeros(shape, dtype=np.float32)
    amplitude, total = 1.0, 0.0
    for o in range(octaves):
        small = rng.random((max(2, shape[0] // (2 ** (octaves - o))),
                            max(2, shape[1] // (2 ** (octaves - o))))).astype(np.float32)
        up = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        result += amplitude * up
        total += amplitude
        amplitude *= 0.55
    result /= total
    result = (result - result.min()) / (result.max() - result.min() + 1e-9)
    return result


def make_surface(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Фон: поверхность Земли одной из палитр."""
    palette = rng.choice(["ocean", "land", "desert", "snow"])
    n = fbm_noise(shape, octaves=6, seed=int(rng.integers(1e9)))
    if palette == "ocean":
        base = np.array([20, 60, 120], dtype=np.float32)
        var = np.array([10, 30, 60], dtype=np.float32)
    elif palette == "land":
        base = np.array([40, 90, 50], dtype=np.float32)
        var = np.array([30, 40, 35], dtype=np.float32)
    elif palette == "desert":
        base = np.array([90, 160, 200], dtype=np.float32)
        var = np.array([25, 35, 40], dtype=np.float32)
    else:
        base = np.array([225, 230, 235], dtype=np.float32)
        var = np.array([15, 15, 15], dtype=np.float32)
    img = base[None, None, :] + (n[:, :, None] - 0.5) * 2 * var[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_cloud_mask(shape: tuple[int, int], cloud_fraction: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Бинарная маска облаков из fBm-поля с порогом под нужную долю."""
    n = fbm_noise(shape, octaves=6, seed=int(rng.integers(1e9)))
    thr = np.quantile(n, 1.0 - cloud_fraction)
    mask = (n >= thr).astype(np.uint8)
    if mask.any() and not mask.all():
        soft = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)
        target = cloud_fraction
        cur = soft.mean()
        if cur > 0 and abs(cur - target) > 0.02:
            thr2 = np.quantile(soft, 1.0 - target)
            mask = (soft >= thr2).astype(np.uint8)
    return mask


def render(shape: tuple[int, int], cloud_fraction: float,
           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (изображение RGB uint8, маска uint8 0/1)."""
    surface = make_surface(shape, rng)
    mask = make_cloud_mask(shape, cloud_fraction, rng)
    img = surface.astype(np.float32)

    m = mask.astype(bool)
    if m.any():
        cloud_noise = fbm_noise(shape, octaves=4, seed=int(rng.integers(1e9)))
        soft = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        alpha = np.clip(soft * (0.75 + 0.25 * cloud_noise), 0, 1)
        cloud_color = np.array([245, 248, 252], dtype=np.float32)[None, None, :]
        img = img * (1 - alpha[:, :, None]) + cloud_color * alpha[:, :, None] * 255 / 255

    img += rng.normal(0, 4, img.shape).astype(np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = shape[0] / 2, shape[1] / 2
    r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
    vign = 1.0 - 0.12 * r2
    img *= vign[:, :, None]

    return np.clip(img, 0, 255).astype(np.uint8), mask


def main():
    parser = argparse.ArgumentParser(description="Генератор синтетических данных")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--n", type=int, default=60, help="число пар снимок+маска")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    shape = (args.size, args.size)

    fractions = np.clip(rng.beta(1.2, 1.6, size=args.n), 0.0, 1.0)
    n_over75 = 0
    for i, frac in enumerate(fractions):
        img_rgb, mask = render(shape, float(frac), rng)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        from utils import imwrite_unicode
        imwrite_unicode(out / "images" / f"synth_{i:04d}.png", img_bgr)
        imwrite_unicode(out / "masks" / f"synth_{i:04d}.png",
                        (mask * 255).astype(np.uint8))
        if frac * 100 > 75:
            n_over75 += 1

    print(f"[demo-data] Сгенерировано {args.n} пар в {out}")
    print(f"  Размер: {args.size}x{args.size} | "
          f"кадров с облачностью >75%: {n_over75} "
          f"({n_over75/args.n*100:.0f}%)")
    print(f"  Диапазон облачности: {fractions.min()*100:.0f}–{fractions.max()*100:.0f}%")


if __name__ == "__main__":
    main()
