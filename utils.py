"""utils.py — метрики, визуализация, работа с файлами и отчётами."""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

def seed_everything(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread с поддержкой кириллицы в путях (Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image, params=None) -> bool:
    """cv2.imwrite с поддержкой кириллицы в путях (Windows)."""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image, params or [])
    if ok:
        buf.tofile(str(path))
    return ok


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """IoU (Jaccard) для класса «облако»."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + eps))


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Dice coefficient (F1-мера пикселей) для класса «облако»."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() + gt.sum() == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + eps))


def precision_recall(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Precision и Recall для класса «облако»."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    return precision, recall


def cloud_percentage(mask: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Процент облачности = площадь облачных пикселей / площадь валидных * 100."""
    mask = mask.astype(bool)
    if valid is None:
        denom = mask.size
    else:
        mask = mask & valid.astype(bool)
        denom = int(valid.astype(bool).sum())
    if denom == 0:
        return 0.0
    return float(mask.sum()) / denom * 100.0


def valid_pixels_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Маска валидных пикселей: полностью чёрные (0,0,0) пиксели считаем NODATA."""
    return ~(np.all(image_rgb == 0, axis=2))


def threshold_confusion(pred_pct: np.ndarray | list, gt_pct: np.ndarray | list,
                        threshold: float = 75.0) -> dict:
    """Матрица ошибок решения «отбраковать снимок» (облачность > threshold)."""
    pred_pct = np.asarray(pred_pct, dtype=float)
    gt_pct = np.asarray(gt_pct, dtype=float)
    pred_reject = pred_pct > threshold
    gt_reject = gt_pct > threshold

    tp = int(np.logical_and(pred_reject, gt_reject).sum())
    fp = int(np.logical_and(pred_reject, ~gt_reject).sum())
    tn = int(np.logical_and(~pred_reject, ~gt_reject).sum())
    fn = int(np.logical_and(~pred_reject, gt_reject).sum())
    total = tp + fp + tn + fn
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "threshold_pct": threshold,
        "tp_rejected_correctly": tp,
        "fp_rejected_wrongly": fp,
        "tn_kept_correctly": tn,
        "fn_kept_wrongly": fn,
        "decision_accuracy": (tp + tn) / (total + eps),
        "decision_precision": precision,
        "decision_recall": recall,
        "decision_f1": 2 * precision * recall / (precision + recall + eps),
    }


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    """Оверлей: облака подсвечены бирюзовым, контур — красным."""
    color = np.array([0, 220, 255], dtype=np.uint8)
    overlay = image_rgb.copy()
    m = mask.astype(bool)
    overlay[m] = (image_rgb[m].astype(np.float32) * (1 - alpha)
                  + color.astype(np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 60, 60), 2)
    return overlay


def save_prediction_gallery(save_path: Path, image_rgb: np.ndarray,
                            gt_mask: np.ndarray | None, pred_mask: np.ndarray,
                            pred_pct: float, gt_pct: float | None = None) -> None:
    """Панель «снимок | истина | предсказание» для отчётов и презентации."""
    h, w = image_rgb.shape[:2]
    panels = [image_rgb]
    titles = ["Снимок"]
    if gt_mask is not None:
        panels.append(colorize_mask(gt_mask))
        titles.append(f"Разметка ({gt_pct:.1f}%)" if gt_pct is not None else "Разметка")
    panels.append(colorize_mask(pred_mask))
    titles.append(f"Предсказание ({pred_pct:.1f}%)")
    bar = np.full((34, w, 3), 18, dtype=np.uint8)

    rows = []
    for panel, title in zip(panels, titles):
        p = panel.copy()
        if p.ndim == 2:
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        bar_titled = bar.copy()
        cv2.putText(bar_titled, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.vstack([bar_titled, p]))
    gallery = np.hstack(rows)
    imwrite_unicode(save_path, cv2.cvtColor(gallery, cv2.COLOR_RGB2BGR))


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Маска 0/1 -> RGB: фон тёмно-синий, облака белые."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = (255, 255, 255)
    out[~mask.astype(bool)] = (25, 40, 70)
    return out


def compute_tiles(height: int, width: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Список прямоугольников (y0, x0, y1, x1) с перекрытием, покрывающих кадр."""
    stride = max(1, int(tile * (1 - overlap)))
    tiles = []
    for y0 in range(0, height, stride):
        for x0 in range(0, width, stride):
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)
            tiles.append((y0c, x0c, y1, x1))
            if x1 >= width:
                break
        if tiles[-1][2] >= height:
            break
    return tiles


def reject_file(src: Path, rejected_dir: Path, hard_delete: bool = False) -> Path | None:
    """Безопасная отбраковка: по умолчанию ПЕРЕМЕШЕНИЕ в rejected/,."""
    rejected_dir.mkdir(parents=True, exist_ok=True)
    if hard_delete:
        src.unlink()
        return None
    dst = rejected_dir / src.name
    if dst.exists():
        dst = rejected_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    shutil.move(str(src), str(dst))
    return dst


def accept_file(src: Path, accepted_dir: Path, copy: bool = True) -> Path:
    """Сохранение прошедшего отбор снимка: копия (по умолчанию) или перемещение."""
    accepted_dir.mkdir(parents=True, exist_ok=True)
    dst = accepted_dir / src.name
    if dst.exists():
        dst = accepted_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    if copy:
        shutil.copy2(str(src), str(dst))
    else:
        shutil.move(str(src), str(dst))
    return dst


def format_operator_line(row: dict) -> str:
    """Человекочитаемая строка по одному файлу — формат из ТЗ."""
    pct = row.get("cloud_pct")
    verb = "Удалён" if row.get("decision") == "rejected" else "Сохранён"
    if pct is None:
        return f"{verb}: {row.get('filename')} — не удалось обработать."
    return f"{verb}: {row['filename']} — облачность {pct:.1f}%."


def write_reports(results: list[dict], out_dir: Path, run_meta: dict | None = None) -> dict[str, Path]:
    """Пишет report.csv, report.json и operator_report.txt. Возвращает пути."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_path = out_dir / "report.csv"
    fields = ["filename", "cloud_pct", "decision", "saved_path", "rejected_path",
              "cloud_pixels", "valid_pixels", "time_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    n_rej = sum(1 for r in results if r["decision"] == "rejected")
    payload = {
        "run_timestamp": stamp,
        "meta": run_meta or {},
        "summary": {
            "total": len(results),
            "accepted": len(results) - n_rej,
            "rejected": n_rej,
            "cloud_threshold_pct": (run_meta or {}).get("cloud_threshold_pct"),
        },
        "files": results,
    }
    json_path = out_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    txt_path = out_dir / "operator_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ОТЧЁТ ОБРАБОТКИ СПУТНИКОВЫХ СНИМКОВ\n")
        f.write(f"Время обработки: {stamp}\n")
        thr = (run_meta or {}).get("cloud_threshold_pct", 75.0)
        f.write(f"Порог отбраковки: облачность > {thr}%\n")
        f.write("=" * 60 + "\n\n")
        f.write("Результаты обработки:\n")
        for r in results:
            f.write("  " + format_operator_line(r) + "\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"ИТОГО: обработано {len(results)} снимков\n")
        f.write(f"  Сохранено:  {len(results) - n_rej}\n")
        f.write(f"  Отбраковано (облачность > {thr}%): {n_rej}\n")

    return {"csv": csv_path, "json": json_path, "txt": txt_path}
