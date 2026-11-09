"""predict.py — обработка папки со спутниковыми снимками (инференс)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from dataset import IMAGE_EXTS, load_image_rgb
from model import build_model
from utils import (accept_file, cloud_percentage, compute_tiles,
                   format_operator_line, overlay_mask, reject_file,
                   valid_pixels_mask, write_reports)


def percentile_stretch(image_rgb: np.ndarray, lo_p: float = 2.0,
                       hi_p: float = 98.0) -> np.ndarray:
    """Перцентильная растяжка контраста по каналам (домен-адаптация)."""
    out = np.empty_like(image_rgb)
    for c in range(image_rgb.shape[2]):
        ch = image_rgb[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, lo_p), np.percentile(ch, hi_p)
        out[:, :, c] = np.clip((ch - lo) / max(hi - lo, 1.0) * 255.0, 0, 255)
    return out.astype(np.uint8)


def adaptive_preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Адаптивная предобработка кадра перед моделью."""
    import cv2
    lum = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    p2 = float(np.percentile(lum, 2))
    if p2 < 130.0:
        return percentile_stretch(image_rgb)
    return image_rgb


@torch.no_grad()
def predict_mask(model, image_rgb: np.ndarray, device: torch.device,
                 tile_size: int = 512, overlap: float = 0.25,
                 mask_threshold: float = 0.5) -> np.ndarray:
    """Полная маска облаков кадра; большие изображения — тайлами с усреднением."""
    h, w = image_rgb.shape[:2]
    use_amp = device.type == "cuda"

    def forward(patch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0)
        x = x.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            logits = model(x)
        return torch.sigmoid(logits.float())[0, 0].cpu().numpy()

    if h <= tile_size and w <= tile_size:
        return (forward(image_rgb) > mask_threshold).astype(np.uint8)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for (y0, x0, y1, x1) in compute_tiles(h, w, tile_size, overlap):
        patch = image_rgb[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        pad_h = (16 - ph % 16) % 16
        pad_w = (16 - pw % 16) % 16
        if pad_h or pad_w:
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        prob = forward(patch)[:ph, :pw]
        prob_sum[y0:y1, x0:x1] += prob
        weight[y0:y1, x0:x1] += 1.0
    avg_prob = prob_sum / np.maximum(weight, 1e-6)
    return (avg_prob > mask_threshold).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Обработка снимков: маска облаков -> процент облачности -> "
                    "отбраковка >75%% -> отчёт оператору")
    parser.add_argument("--input", required=True,
                        help="папка со снимками или один файл")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--out-dir", default="outputs/results")
    parser.add_argument("--cloud-threshold", type=float, default=75.0,
                        help="порог отбраковки, %% (по ТЗ 75)")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--hard-delete", action="store_true",
                        help="удалять отбракованные файлы безвозвратно "
                             "(по умолчанию — перенос в rejected/)")
    parser.add_argument("--move-accepted", action="store_true",
                        help="перемещать (а не копировать) принятые снимки")
    parser.add_argument("--save-masks", action="store_true",
                        help="сохранять визуализацию маски для каждого снимка")
    parser.add_argument("--no-stretch", action="store_true",
                        help="отключить растяжку контраста (домен-адаптацию)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    try:
        model, arch_used = build_model(ckpt.get("arch", "unet"),
                                       ckpt.get("encoder", "resnet18"),
                                       pretrained=False)
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        if "size mismatch" in str(e) or "Missing key" in str(e):
            raise SystemExit(
                "Чекпоинт обучен на U-Net с pretrained-энкодером, но библиотека\n"
                "segmentation-models-pytorch не установлена или её версия другая.\n"
                "Выполните: pip install -r requirements.txt")
        raise
    model.to(device).eval()
    print(f"[predict] Модель: {arch_used}, устройство: {device}, "
          f"порог отбраковки: облачность > {args.cloud_threshold}%\n")

    src = Path(args.input)
    files = ([src] if src.is_file()
             else sorted(p for p in src.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS))
    if not files:
        raise SystemExit(f"[predict] В {src} не найдено изображений")

    out_dir = Path(args.out_dir)
    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    vis_dir = out_dir / "visualizations"

    print(f"[predict] Обработка {len(files)} снимков из {src}\n")
    print("СООБЩЕНИЕ ОПЕРАТОРУ:")
    print("-" * 60)

    results = []
    t_start = time.time()
    for f in files:
        t0 = time.time()
        try:
            image = load_image_rgb(f)
            image_for_model = image if args.no_stretch else adaptive_preprocess(image)
            mask = predict_mask(model, image_for_model, device, args.tile_size,
                                args.tile_overlap, args.mask_threshold)
            valid = valid_pixels_mask(image)
            pct = cloud_percentage(mask, valid)
            n_cloud = int(mask.astype(bool).sum())
            n_valid = int(valid.sum())

            if pct > args.cloud_threshold:
                dst = reject_file(f, rejected_dir, hard_delete=args.hard_delete)
                decision, row_dst = "rejected", (str(dst) if dst else "DELETED")
            else:
                dst = accept_file(f, accepted_dir, copy=not args.move_accepted)
                decision, row_dst = "accepted", str(dst)

            row = {
                "filename": f.name,
                "cloud_pct": round(pct, 1),
                "decision": decision,
                "saved_path": row_dst if decision == "accepted" else "",
                "rejected_path": row_dst if decision == "rejected" else "",
                "cloud_pixels": n_cloud,
                "valid_pixels": n_valid,
                "time_sec": round(time.time() - t0, 2),
            }
            results.append(row)

            if args.save_masks:
                vis_dir.mkdir(parents=True, exist_ok=True)
                import cv2
                vis = overlay_mask(image, mask)
                label = f"Cloud: {pct:.1f}%  ->  {'REJECTED' if decision=='rejected' else 'OK'}"
                cv2.putText(vis, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (60, 60, 255), 3, cv2.LINE_AA)
                from utils import imwrite_unicode
                imwrite_unicode(vis_dir / f"{f.stem}_mask.jpg",
                                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            print("  " + format_operator_line(row))
        except Exception as e:
            print(f"  ОШИБКА обработки {f.name}: {type(e).__name__}: {e}")
            results.append({"filename": f.name, "cloud_pct": None,
                            "decision": "error", "saved_path": "",
                            "rejected_path": "", "cloud_pixels": 0,
                            "valid_pixels": 0,
                            "time_sec": round(time.time() - t0, 2)})

    total_time = time.time() - t_start

    n_ok = sum(1 for r in results if r["decision"] == "accepted")
    n_rej = sum(1 for r in results if r["decision"] == "rejected")
    print("-" * 60)
    print(f"ИТОГО: обработано {len(results)} снимков за {total_time:.1f} c")
    print(f"  Сохранено:  {n_ok}")
    print(f"  Отбраковано (облачность > {args.cloud_threshold}%): {n_rej}")
    print(f"  Папки: {accepted_dir} | {rejected_dir}")

    paths = write_reports(results, out_dir, run_meta={
        "cloud_threshold_pct": args.cloud_threshold,
        "hard_delete": args.hard_delete,
        "checkpoint": args.checkpoint,
        "input": str(src),
        "total_time_sec": round(total_time, 1),
    })
    print(f"\nОтчёты сохранены: {paths['csv']} | {paths['json']} | {paths['txt']}")


if __name__ == "__main__":
    main()
