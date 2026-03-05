"""
Utility functions: bbox from mask, IoU, Dice, EarlyStopping, seeding.

- Derive bbox from GT mask
- Compute IoU/Dice metrics
- Early stopping on validation score
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Optional, Tuple, Dict

import numpy as np
import torch



# Reproducibility 
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Mask -> BBox
def bbox_from_mask(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bx2, by2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter = inter_w * inter_h

    area_a = torch.clamp(ax2 - ax1, min=0) * torch.clamp(ay2 - ay1, min=0)
    area_b = torch.clamp(bx2 - bx1, min=0) * torch.clamp(by2 - by1, min=0)
    union = area_a + area_b - inter + 1e-6
    return inter / union


# Segmentation metrics
def seg_iou_and_dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    # IoU
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3)) + 1e-6
    iou = inter / union

    # Dice
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice = (2.0 * inter) / denom
    return iou.mean(), dice.mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice = (2.0 * inter) / denom
    return 1.0 - dice.mean()


# Average meter
@dataclass
class AvgMeter:
    total: float = 0.0
    n: int = 0

    def update(self, v: float, k: int = 1) -> None:
        self.total += float(v) * k
        self.n += int(k)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.n)



# Early stopping
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)