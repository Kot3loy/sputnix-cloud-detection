"""train.py — обучение модели сегментации облаков (U-Net, PyTorch)."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CloudSegmentationDataset
from model import build_model, count_params
from utils import (dice_score, iou_score, save_prediction_gallery,
                   seed_everything)


class BCEDiceLoss(nn.Module):
    """BCE + Dice: устойчивый стандарт бинарной сегментации."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    @staticmethod
    def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                         eps: float = 1.0) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()

    def forward(self, logits, targets):
        return (self.bce_weight * self.bce(logits, targets)
                + (1 - self.bce_weight) * self.dice_from_logits(logits, targets))


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              max_steps=None, epoch_name="epoch"):
    """train (если передан optimizer) или validation. Возвращает метрики эпохи."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, steps = 0.0, 0
    inter_sum, union_sum = 0.0, 0.0

    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            steps += 1

            with torch.no_grad():
                pred = (torch.sigmoid(logits.float()) > 0.5)
                tgt = y.bool()
                inter_sum += torch.logical_and(pred, tgt).sum().item()
                union_sum += torch.logical_or(pred, tgt).sum().item()

            if steps % 50 == 0:
                iou_run = inter_sum / max(union_sum, 1)
                print(f"  [{epoch_name}] шаг {steps}/{len(loader)} | "
                      f"loss {total_loss/steps:.4f} | IoU(тек.) {iou_run:.4f}")

    eps = 1e-6
    return {
        "loss": total_loss / max(steps, 1),
        "iou": inter_sum / (union_sum + eps),
    }


@torch.no_grad()
def save_val_samples(model, val_ds, device, out_dir: Path, epoch: int,
                     n_samples: int = 3):
    """Сохраняет несколько галерей «снимок|разметка|предсказание»."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(val_ds) - 1, min(n_samples, len(val_ds))).astype(int)
    for k, idx in enumerate(idxs):
        x, y = val_ds[idx]
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(x.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits.float())[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        img = (x.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        gt = y.numpy().astype(np.uint8)
        save_prediction_gallery(out_dir / f"val_epoch{epoch:02d}_{k}.png",
                                img, gt, pred,
                                pred_pct=float(pred.mean() * 100),
                                gt_pct=float(gt.mean() * 100))


def main():
    parser = argparse.ArgumentParser(description="Обучение U-Net для сегментации облаков")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--arch", default="unet", choices=["unet", "small_unet"])
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="отключить смешанную точность")
    parser.add_argument("--haze-aug", action="store_true",
                        help="эксперимент: аугментация дымки (домен-адаптация)")
    parser.add_argument("--init-from", default=None,
                        help="путь к чекпоинту для дообучения (fine-tune)")
    parser.add_argument("--limit", type=int, default=None, help="ограничить размер train-выборки")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir, log_dir, pred_dir = (out_dir / "checkpoints", out_dir / "logs",
                                   out_dir / "predictions")
    for d in (ckpt_dir, log_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_ds = CloudSegmentationDataset(Path(args.data_dir) / "train" / "images",
                                        Path(args.data_dir) / "train" / "masks",
                                        img_size=args.img_size, augment=True,
                                        haze_aug=args.haze_aug)
    val_ds = CloudSegmentationDataset(Path(args.data_dir) / "val" / "images",
                                      Path(args.data_dir) / "val" / "masks",
                                      img_size=args.img_size, augment=False)
    if args.limit is not None:
        from torch.utils.data import Subset
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_ds))[:args.limit]
        train_ds = Subset(train_ds, idx.tolist())
        idx_v = rng.permutation(len(val_ds))[:max(8, args.limit // 10)]
        val_ds = Subset(val_ds, idx_v.tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, args.num_workers - 1) or 0,
                            pin_memory=True,
                            persistent_workers=args.num_workers > 1)
    print(f"[train] Обучающая выборка: {len(train_ds)} пар, "
          f"валидация: {len(val_ds)} пар")

    model, arch_used = build_model(args.arch, args.encoder,
                                   pretrained=not args.no_pretrained)
    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=device,
                               weights_only=False)
        model.load_state_dict(init_ckpt["model_state"])
        print(f"[train] Инициализация с чекпоинта: {args.init_from} "
              f"(эпоха {init_ckpt.get('epoch')}, "
              f"val IoU {init_ckpt.get('val_iou')})")
    model.to(device)
    print(f"[train] Модель: {arch_used} ({args.encoder}), "
          f"параметров: {count_params(model):,}, устройство: {device}")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"[train] AMP (смешанная точность): {'вкл' if use_amp else 'выкл'}")

    log_path = log_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_iou",
                                "val_dice", "lr", "epoch_time_sec"])

    best_iou = -1.0
    t0_all = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n=== Эпоха {epoch}/{args.epochs} ===")
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler,
                       max_steps=args.max_steps_per_epoch, epoch_name=f"train {epoch}")
        va = run_epoch(model, val_loader, criterion, device,
                       max_steps=args.max_steps_per_epoch, epoch_name=f"val {epoch}")
        scheduler.step()

        val_iou = va["iou"]
        print(f"[epoch {epoch}] train_loss={tr['loss']:.4f} | "
              f"val_loss={va['loss']:.4f} | val_IoU={val_iou:.4f} | "
              f"время {time.time()-t0:.0f} c")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{tr['loss']:.4f}", f"{va['loss']:.4f}",
                                    f"{val_iou:.4f}", f"{val_iou:.4f}",
                                    f"{scheduler.get_last_lr()[0]:.2e}",
                                    f"{time.time()-t0:.0f}"])

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt = {"epoch": epoch, "arch": arch_used, "encoder": args.encoder,
                    "img_size": args.img_size, "val_iou": val_iou,
                    "model_state": model.state_dict()}
            torch.save(ckpt, ckpt_dir / "best_model.pth")
            print(f"[checkpoint] Новый лучший val_IoU={val_iou:.4f} -> best_model.pth")

        if epoch % 2 == 0 or epoch == args.epochs:
            save_val_samples(model, val_ds, device, pred_dir, epoch)

    total_min = (time.time() - t0_all) / 60
    print(f"\n[train] Обучение завершено за {total_min:.1f} мин. "
          f"Лучший val IoU = {best_iou:.4f}")
    print(f"[train] Чекпоинт: {ckpt_dir / 'best_model.pth'}")
    print(f"[train] Лог: {log_path}")

    with open(out_dir / "logs" / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({"arch": arch_used, "encoder": args.encoder,
                   "epochs_trained": args.epochs, "best_val_iou": round(best_iou, 4),
                   "train_pairs": len(train_ds), "val_pairs": len(val_ds),
                   "batch_size": args.batch_size, "lr": args.lr,
                   "img_size": args.img_size,
                   "total_time_min": round(total_min, 1),
                   "device": str(device), "amp": use_amp,
                   "dataset": str(args.data_dir)},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
