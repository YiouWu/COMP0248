"""
Evaluation for detection / segmentation / classification metrics.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

from .dataloader import RGBDGestureDataset, collate_fn, GESTURES
from .utils import box_iou_xyxy, seg_iou_and_dice_from_logits, ensure_dir
from .model import build_model

@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    all_y = []
    all_pred = []

    # For det/seg only count samples with has_mask=1
    det_iou_list = []
    det_acc05_list = []
    seg_iou_list = []
    seg_dice_list = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y_cls"].to(device)
        has_mask = batch["has_mask"].to(device)  
        gt_box = batch["box"].to(device)         
        gt_mask = batch["mask"].to(device)       

        out = model(x)
        cls_logits = out["cls_logits"]
        box_pred = out["box"].clamp(0, 1)
        seg_logits = out["seg_logits"]

        pred = cls_logits.argmax(dim=1)
        all_y.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

        # Select only labeled samples for det/seg
        idx = (has_mask == 1).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            # detection IoU
            iou = box_iou_xyxy(box_pred[idx], gt_box[idx])
            det_iou_list.append(iou.detach().cpu().numpy())
            det_acc05_list.append((iou >= 0.5).float().detach().cpu().numpy())

            # segmentation
            siou, sdice = seg_iou_and_dice_from_logits(seg_logits[idx], gt_mask[idx])
            seg_iou_list.append(float(siou.detach().cpu()))
            seg_dice_list.append(float(sdice.detach().cpu()))

    y_true = np.concatenate(all_y, axis=0) if len(all_y) else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred, axis=0) if len(all_pred) else np.array([], dtype=np.int64)

    cls_acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    cls_f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true.size else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GESTURES)))) if y_true.size else None

    if len(det_iou_list) > 0:
        det_iou = float(np.concatenate(det_iou_list).mean())
        det_acc05 = float(np.concatenate(det_acc05_list).mean())
        seg_iou = float(np.mean(seg_iou_list)) if len(seg_iou_list) else 0.0
        seg_dice = float(np.mean(seg_dice_list)) if len(seg_dice_list) else 0.0
    else:
        det_iou, det_acc05, seg_iou, seg_dice = 0.0, 0.0, 0.0, 0.0

    # score for early stopping 
    score = (seg_dice + det_acc05 + cls_acc) / 3.0

    return {
        "seg_iou": seg_iou,
        "seg_dice": seg_dice,
        "det_iou": det_iou,
        "det_acc05": det_acc05,
        "cls_acc": cls_acc,
        "cls_macro_f1": cls_f1,
        "score": score,
        "confusion_matrix": cm.tolist() if cm is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["val", "train", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--variant", type=str, default="baseline", choices=["baseline", "innovation"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split_info = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    subjects = split_info.get(f"{args.split}_subjects", None)
    if subjects is None:
        # For val/train we stored train_subjects/val_subjects, test optional.
        subjects = split_info.get(f"{args.split}_subjects", None)

    if args.split == "train":
        subjects = split_info["train_subjects"]
    elif args.split == "val":
        subjects = split_info["val_subjects"]
    elif args.split == "test":
        subjects = split_info.get("test_subjects", split_info["val_subjects"])

    ds = RGBDGestureDataset(
        data_root=args.data_root,
        subjects=subjects,
        split=args.split,
        target_size=args.target_size,
        include_unlabeled_for_cls=False,  # eval only on full labels
        augment=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(args.variant, num_classes=len(GESTURES), base=32).to(device)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    metrics = evaluate_loader(model, loader, device)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_path = out_dir / f"metrics_{args.split}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[EVAL]", metrics)
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()