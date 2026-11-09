"""evaluate.py — оценка качества модели на валидационном наборе."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CloudSegmentationDataset
from model import build_model
from utils import (cloud_percentage, dice_score, iou_score,
                   precision_recall, save_prediction_gallery,
                   seed_everything, threshold_confusion)


@torch.no_grad()
def evaluate(model, loader, device, mask_threshold: float = 0.5):
    """Прогон по датасету; возвращает списки метрик по каждому изображению."""
    model.eval()
    per_image = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        gts = y.numpy()
        for prob, gt in zip(probs, gts):
            pred = (prob > mask_threshold).astype(np.uint8)
            gt_m = (gt > 0.5).astype(np.uint8)
            pr, rc = precision_recall(pred, gt_m)
            pred_b, gt_b = pred.astype(bool), gt_m.astype(bool)
            per_image.append({
                "iou": iou_score(pred, gt_m),
                "dice": dice_score(pred, gt_m),
                "precision": pr,
                "recall": rc,
                "tp": int(np.logical_and(pred_b, gt_b).sum()),
                "fp": int(np.logical_and(pred_b, ~gt_b).sum()),
                "fn": int(np.logical_and(~pred_b, gt_b).sum()),
                "pred_cloud_pct": cloud_percentage(pred),
                "gt_cloud_pct": cloud_percentage(gt_m),
            })
    return per_image


def main():
    parser = argparse.ArgumentParser(description="Оценка качества модели")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--cloud-threshold", type=float, default=75.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="outputs/metrics/metrics.json")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, arch_used = build_model(ckpt.get("arch", "unet"),
                                   ckpt.get("encoder", "resnet18"),
                                   pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"[eval] Модель загружена: {arch_used}, epoch {ckpt.get('epoch')}, "
          f"val_IoU на обучении: {ckpt.get('val_iou')}")

    ds = CloudSegmentationDataset(Path(args.data_dir) / args.split / "images",
                                  Path(args.data_dir) / args.split / "masks",
                                  img_size=args.img_size, augment=False)
    if args.limit is not None:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(ds))[:args.limit]
        from torch.utils.data import Subset
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[eval] Набор {args.split}: {len(ds)} изображений")

    rows = evaluate(model, loader, device, args.mask_threshold)

    ious = [r["iou"] for r in rows]
    dices = [r["dice"] for r in rows]
    precs = [r["precision"] for r in rows]
    recs = [r["recall"] for r in rows]
    pred_pcts = [r["pred_cloud_pct"] for r in rows]
    gt_pcts = [r["gt_cloud_pct"] for r in rows]
    mae_pct = float(np.mean([abs(p - g) for p, g in zip(pred_pcts, gt_pcts)]))

    conf = threshold_confusion(pred_pcts, gt_pcts, args.cloud_threshold)

    tp = sum(r.get("tp", 0) for r in rows)
    fp = sum(r.get("fp", 0) for r in rows)
    fn = sum(r.get("fn", 0) for r in rows)
    global_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    global_dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

    metrics = {
        "dataset": str(args.data_dir),
        "split": args.split,
        "n_images": len(rows),
        "checkpoint": str(args.checkpoint),
        "arch": arch_used,
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mask_metrics": {
            "IoU": round(float(np.mean(ious)), 4),
            "Dice": round(float(np.mean(dices)), 4),
            "Precision": round(float(np.mean(precs)), 4),
            "Recall": round(float(np.mean(recs)), 4),
        },
        "mask_metrics_global": {
            "IoU": round(global_iou, 4),
            "Dice": round(global_dice, 4),
            "note": "взвешено по площади пикселей; согласуется с логом обучения",
        },
        "cloud_pct_MAE": round(mae_pct, 2),
        "cloud_pct_pred_mean": round(float(np.mean(pred_pcts)), 1),
        "cloud_pct_gt_mean": round(float(np.mean(gt_pcts)), 1),
        "threshold_75_confusion": conf,
        "per_image": [
            {"iou": round(r["iou"], 4), "dice": round(r["dice"], 4),
             "pred_cloud_pct": round(r["pred_cloud_pct"], 1),
             "gt_cloud_pct": round(r["gt_cloud_pct"], 1)}
            for r in rows
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n========== МЕТРИКИ КАЧЕСТВА ==========")
    print(f"Изображений:                {metrics['n_images']}")
    print(f"IoU (Jaccard) средний:      {metrics['mask_metrics']['IoU']}")
    print(f"IoU глобальный (по площади):{metrics['mask_metrics_global']['IoU']}")
    print(f"Dice (F1) средний:          {metrics['mask_metrics']['Dice']}")
    print(f"Precision:                  {metrics['mask_metrics']['Precision']}")
    print(f"Recall:                     {metrics['mask_metrics']['Recall']}")
    print(f"MAE процента облачности:    {metrics['cloud_pct_MAE']} п.п.")
    print(f"\nМатрица ошибок решения (порог >{args.cloud_threshold}%):")
    print(f"  TP (верно отбраковано):    {conf['tp_rejected_correctly']}")
    print(f"  FP (зря отбраковано):      {conf['fp_rejected_wrongly']}")
    print(f"  TN (верно сохранено):      {conf['tn_kept_correctly']}")
    print(f"  FN (пропущена бракованная):{conf['fn_kept_wrongly']}")
    print(f"  Точность решения:          {conf['decision_accuracy']*100:.1f}%")
    print(f"======================================")
    print(f"[eval] Сохранено: {out_path}")


if __name__ == "__main__":
    main()
