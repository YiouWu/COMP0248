"""
Training script with early stopping for RGB-D baseline.

- Uses only annotated keyframes
- Early stopping on combined validation score.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from .dataloader import RGBDGestureDataset, collate_fn, GESTURES
from .evaluate import evaluate_loader
from .utils import (
    set_seed, EarlyStopping, AvgMeter, dice_loss_from_logits,
    ensure_dir, count_trainable_params
)

from .model import build_model  


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cls_w: float,
    seg_w: float,
    box_w: float,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()

    loss_m = AvgMeter()
    cls_m = AvgMeter()
    seg_m = AvgMeter()
    box_m = AvgMeter()

    bce = nn.BCEWithLogitsLoss(reduction="mean")
    smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    
    pbar = tqdm(loader, desc="Train", ncols=110, leave=False)
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y_cls"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)  
        gt_mask = batch["mask"].to(device, non_blocking=True)       
        gt_box = batch["box"].to(device, non_blocking=True)         

        optim.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x)
            cls_logits = out["cls_logits"]
            seg_logits = out["seg_logits"]
            box_pred = out["box"].clamp(0, 1)

            # Classification loss
            cls_loss = F.cross_entropy(cls_logits, y)

            # Seg+Det only on labeled samples
            idx = (has_mask == 1).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                seg_bce = bce(seg_logits[idx], gt_mask[idx])
                seg_dice = dice_loss_from_logits(seg_logits[idx], gt_mask[idx])
                seg_loss = seg_bce + seg_dice

                box_loss = smooth_l1(box_pred[idx], gt_box[idx])
            else:
                seg_loss = torch.zeros((), device=device)
                box_loss = torch.zeros((), device=device)

            loss = cls_w * cls_loss + seg_w * seg_loss + box_w * box_loss

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_m.update(float(loss.detach().cpu()))
        cls_m.update(float(cls_loss.detach().cpu()))
        seg_m.update(float(seg_loss.detach().cpu()))
        box_m.update(float(box_loss.detach().cpu()))
        
        pbar.set_postfix({
            "loss": f"{loss_m.avg:.3f}",
            "cls": f"{cls_m.avg:.3f}",
            "seg": f"{seg_m.avg:.3f}",
            "box": f"{box_m.avg:.3f}",
        })
    return {
        "train_loss": loss_m.avg,
        "train_cls_loss": cls_m.avg,
        "train_seg_loss": seg_m.avg,
        "train_box_loss": box_m.avg,
    }


def save_ckpt(path: Path, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--weights_dir", type=str, default="weights")
    ap.add_argument("--variant", type=str, default="baseline", choices=["baseline", "innovation"])

    ap.add_argument("--epochs", type=int, default=100)      # use early stopping anyway
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--cls_w", type=float, default=1.0)
    ap.add_argument("--seg_w", type=float, default=1.0)
    ap.add_argument("--box_w", type=float, default=1.0)

    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--include_unlabeled_for_cls", action="store_true",
                    help="If set, include unlabeled frames in TRAIN for cls-only learning.")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] device={device} | torch_cuda={torch.version.cuda} | cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    out_dir = Path(args.out_dir)
    weights_dir = Path(args.weights_dir)
    ensure_dir(out_dir)
    ensure_dir(weights_dir)

    split_info = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    train_subjects = split_info["train_subjects"]
    val_subjects = split_info["val_subjects"]

    # Datasets
    ds_train = RGBDGestureDataset(
        data_root=args.data_root,
        subjects=train_subjects,
        split="train",
        target_size=args.target_size,
        include_unlabeled_for_cls=args.include_unlabeled_for_cls,
        augment=True,
    )
    ds_val = RGBDGestureDataset(
        data_root=args.data_root,
        subjects=val_subjects,
        split="val",
        target_size=args.target_size,
        include_unlabeled_for_cls=False,
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # Model
    model = build_model(args.variant, num_classes=len(GESTURES), base=32).to(device)
    n_params = count_trainable_params(model)
    print(f"[Model] trainable params = {n_params}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(not args.no_amp))
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # CSV log
    log_path = out_dir / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "lr", "train_loss",
            "val_det_acc@0.5", "val_det_mean_iou",
            "val_seg_mean_iou", "val_seg_dice",
            "val_cls_top1", "val_cls_macro_f1",
            "val_score"
        ])

    best_score = -1e9

    for epoch in range(1, args.epochs + 1):
        lr_now = optim.param_groups[0]["lr"]

        train_stats = train_one_epoch(
            model, train_loader, optim, scaler, device,
            cls_w=args.cls_w, seg_w=args.seg_w, box_w=args.box_w,
            use_amp=(not args.no_amp),
        )

        val_metrics = evaluate_loader(model, val_loader, device)

        # Save last
        save_ckpt(weights_dir / "last.pt", model, optim, epoch, val_metrics)

        # Save best
        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            save_ckpt(weights_dir / "best.pt", model, optim, epoch, val_metrics)

        # Log
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, lr_now, train_stats["train_loss"],
                val_metrics["det_acc05"], val_metrics["det_iou"],
                val_metrics["seg_iou"], val_metrics["seg_dice"],
                val_metrics["cls_acc"], val_metrics["cls_macro_f1"],
                val_metrics["score"]
            ])

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_stats['train_loss']:.4f} | "
            f"VAL det_acc@0.5={val_metrics['det_acc05']:.4f}, det_mean_iou={val_metrics['det_iou']:.4f} | "
            f"seg_mean_iou={val_metrics['seg_iou']:.4f}, seg_dice={val_metrics['seg_dice']:.4f} | "
            f"cls_top1={val_metrics['cls_acc']:.4f}, cls_macro_f1={val_metrics['cls_macro_f1']:.4f} | "
            f"score={val_metrics['score']:.4f}"
        )

        # Early stop
        if stopper.step(val_metrics["score"]):
            print(f"[EarlyStopping] stop at epoch {epoch} (best={stopper.best:.4f})")
            break

    print(f"[Done] best_score={best_score:.4f}")
    print(f"       best weights: {weights_dir / 'best.pt'}")
    print(f"       last weights: {weights_dir / 'last.pt'}")


if __name__ == "__main__":
    main()