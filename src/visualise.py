"""
Visualisation (qualitative overlays + confusion matrix + curves).
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .dataloader import RGBDGestureDataset, collate_fn, GESTURES
from .utils import ensure_dir
from .model import build_model

def _load_split_subjects(split_json: str, split: str) -> List[str]:
    info = json.loads(Path(split_json).read_text(encoding="utf-8"))
    if split == "train":
        return info.get("train_subjects", [])
    if split == "val":
        return info.get("val_subjects", [])
    if split == "test":
        return info.get("test_subjects", info.get("val_subjects", []))
    raise ValueError(f"Unknown split: {split}")


def _tensor_to_rgb(x4: torch.Tensor) -> np.ndarray:
    rgb = x4[:3].detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def _tensor_to_depth01(x4: torch.Tensor) -> np.ndarray:
    d = x4[3].detach().cpu().numpy()
    d = (d * 0.5) + 0.5
    d = np.clip(d, 0.0, 1.0)
    return d


def _mask_from_logits(logits: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    if logits.ndim == 3:
        logits = logits[0]
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    return (prob >= thr).astype(np.float32)


def _draw_box(ax, box_xyxy: np.ndarray, H: int, W: int, color: str, label: str) -> None:
    x1, y1, x2, y2 = box_xyxy
    x1p, x2p = x1 * W, x2 * W
    y1p, y2p = y1 * H, y2 * H
    rect = patches.Rectangle(
        (x1p, y1p), max(0.0, x2p - x1p), max(0.0, y2p - y1p),
        linewidth=2, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(x1p, max(0, y1p - 5), label, color=color, fontsize=9, weight="bold")


def _overlay_mask(rgb: np.ndarray, mask01: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    m = mask01[..., None]
    out = out * (1 - m * alpha) + (np.array(color)[None, None, :] * (m * alpha))
    return np.clip(out, 0.0, 1.0)


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()
    ys, ps = [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y_cls"].to(device)
        out = model(x)
        pred = out["cls_logits"].argmax(dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> np.ndarray:
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curves(log_csv: Path, out_path: Path) -> None:
    
    # Reads results/log.csv written by train.py and plots curves.
    if not log_csv.exists():
        return

    epochs, seg_dice, det_acc, cls_acc, score = [], [], [], [], []
    with log_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            seg_dice.append(float(row["val_seg_dice"]))
            det_acc.append(float(row["val_det_acc@0.5"]))
            cls_acc.append(float(row["val_cls_top1"]))
            score.append(float(row["val_score"]))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, seg_dice, label="seg_dice")
    ax.plot(epochs, det_acc, label="det_acc@0.5")
    ax.plot(epochs, cls_acc, label="cls_acc")
    ax.plot(epochs, score, label="score")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title("Validation metrics over epochs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def save_overlays(
    model: torch.nn.Module,
    dataset: RGBDGestureDataset,
    device: torch.device,
    out_dir: Path,
    split: str,
    num_samples: int,
    seed: int,
    mask_thr: float
) -> None:
    rng = np.random.default_rng(seed)

    # Prefer samples that have mask 
    idx_mask = [i for i in range(len(dataset)) if int(dataset[i]["has_mask"].item()) == 1]
    idx_nomask = [i for i in range(len(dataset)) if int(dataset[i]["has_mask"].item()) == 0]

    chosen: List[int] = []
    if len(idx_mask) > 0:
        chosen.extend(rng.choice(idx_mask, size=min(num_samples, len(idx_mask)), replace=False).tolist())

    # If still not enough, pad with no-mask samples 
    if len(chosen) < num_samples and len(idx_nomask) > 0:
        need = num_samples - len(chosen)
        chosen.extend(rng.choice(idx_nomask, size=min(need, len(idx_nomask)), replace=False).tolist())

    ensure_dir(out_dir)

    model.eval()

    for k, i in enumerate(chosen):
        item = dataset[i]
        x = item["x"].to(device).unsqueeze(0)          # (1,4,H,W)
        y = int(item["y_cls"].item())
        has_mask = int(item["has_mask"].item())
        gt_mask = item["mask"][0].detach().cpu().numpy()  # (H,W)
        gt_box = item["box"].detach().cpu().numpy()       # (4,) normalized or -1

        out = model(x)
        cls_logits = out["cls_logits"][0]
        pred_cls = int(torch.argmax(cls_logits).item())
        conf = float(torch.softmax(cls_logits, dim=0)[pred_cls].item())

        pred_box = out["box"][0].detach().cpu().numpy()
        pred_mask = _mask_from_logits(out["seg_logits"][0], thr=mask_thr)

        x4 = x[0]
        rgb = _tensor_to_rgb(x4)
        depth = _tensor_to_depth01(x4)

        H, W = rgb.shape[0], rgb.shape[1]

        # Build panels 
        gt_overlay = rgb.copy()
        pred_overlay = rgb.copy()

        if has_mask == 1:
            gt_overlay = _overlay_mask(gt_overlay, gt_mask, color=(0.0, 1.0, 0.0), alpha=0.45)   # green
            if (gt_box >= 0).all():
                # draw GT box
                fig_dummy, ax_dummy = plt.subplots(1, 1)  
                plt.close(fig_dummy)

        pred_overlay = _overlay_mask(pred_overlay, pred_mask, color=(1.0, 0.0, 0.0), alpha=0.45) # red

        # Plot 
        fig = plt.figure(figsize=(11, 7))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(rgb)
        ax1.set_title(f"RGB | GT: {GESTURES[y]}")
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(depth, cmap="gray")
        ax2.set_title("Depth (normalized)")
        ax2.axis("off")

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(gt_overlay)
        title3 = "GT overlay (mask+box)" if has_mask == 1 else "GT overlay (no mask available)"
        ax3.set_title(title3)
        ax3.axis("off")
        if has_mask == 1 and (gt_box >= 0).all():
            _draw_box(ax3, gt_box, H, W, color="lime", label="GT box")

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(pred_overlay)
        ax4.set_title(f"Pred overlay | pred={GESTURES[pred_cls]} (p={conf:.2f})")
        ax4.axis("off")
        _draw_box(ax4, pred_box, H, W, color="red", label="Pred box")

        fig.tight_layout()
        out_path = out_dir / f"overlay_{split}_{k:03d}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)

    ap.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask_thr", type=float, default=0.5)

    ap.add_argument("--plot_curves", action="store_true",
                    help="Also plot training curves from results/log.csv.")
    
    ap.add_argument("--variant", type=str, default="baseline", choices=["baseline", "innovation"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    subjects = _load_split_subjects(args.split_json, args.split)
    ds = RGBDGestureDataset(
        data_root=args.data_root,
        subjects=subjects if len(subjects) > 0 else None,
        split=args.split,
        target_size=args.target_size,
        include_unlabeled_for_cls=False,
        augment=False,
    )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    model = build_model(args.variant, num_classes=len(GESTURES), base=32).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visuals" / args.split
    ensure_dir(vis_dir)

    # Overlays 
    save_overlays(
        model=model, dataset=ds, device=device,
        out_dir=vis_dir,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        mask_thr=args.mask_thr,
    )
    print(f"[OK] overlays saved to: {vis_dir}")

    # Confusion matrix 
    y_true, y_pred = _collect_predictions(model, loader, device, num_classes=len(GESTURES))
    cm = _confusion_matrix(y_true, y_pred, C=len(GESTURES))
    cm_path = out_dir / f"confusion_{args.split}.png"
    plot_confusion_matrix(cm, GESTURES, cm_path)
    print(f"[OK] confusion matrix saved: {cm_path}")

    # Training curves
    if args.plot_curves:
        log_csv = out_dir / "log.csv"
        curves_path = out_dir / "curves.png"
        plot_training_curves(log_csv, curves_path)
        if log_csv.exists():
            print(f"[OK] curves saved: {curves_path}")
        else:
            print(f"[skip] {log_csv} not found, curves not plotted.")


if __name__ == "__main__":
    main()