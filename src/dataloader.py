"""
RGB-D Dataset + DataLoader utilities.

- Input is 4-channel RGB-D 
- only uses annotated keyframes for full metrics.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF



# Gesture mapping 
GESTURES = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",
]
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURES)}


def _is_subject_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name
    if name.startswith("__") or name.lower() == "__macosx":
        return False
    return any((p / g).exists() for g in GESTURES)


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _load_depth_png(path: Path) -> np.ndarray:
    # normalize depth png into [0,1] for a baseline.
    
    d = np.array(Image.open(path))
    if d.dtype == np.uint16:
        d = d.astype(np.float32) / 65535.0
    elif d.dtype == np.uint8:
        d = d.astype(np.float32) / 255.0
    else:
        d = d.astype(np.float32)
        mx = float(np.max(d)) if np.max(d) > 0 else 1.0
        d = d / mx
    d = np.clip(d, 0.0, 1.0)
    return d


def _load_mask(path: Path) -> np.ndarray:
    
    # annotation is single-channel png: 0 background, 255 hand.

    # returns float32 mask in {0,1}
    
    m = np.array(Image.open(path))
    m01 = (m > 127).astype(np.float32)
    return m01


def _resize_pad_pil(img: Image.Image, target: int, is_mask: bool = False) -> Image.Image:
    """
    Keep aspect ratio, resize so max(H,W)=target, then pad to (target,target).
    """
    w, h = img.size
    scale = target / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if is_mask:
        img2 = img.resize((new_w, new_h), resample=Image.NEAREST)
    else:
        img2 = img.resize((new_w, new_h), resample=Image.BILINEAR)

    pad_left = (target - new_w) // 2
    pad_top = (target - new_h) // 2
    pad_right = target - new_w - pad_left
    pad_bottom = target - new_h - pad_top

    # PIL padding uses (left, top, right, bottom)
    img3 = TF.pad(img2, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    return img3


@dataclass
class Sample:
    rgb_path: Path
    depth_path: Path
    label: int
    mask_path: Optional[Path]  # None if not annotated


class RGBDGestureDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: Optional[List[str]],
        split: str,
        target_size: int = 256,
        include_unlabeled_for_cls: bool = True,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.include_unlabeled_for_cls = include_unlabeled_for_cls
        self.augment = augment

        # Build subject list
        all_subjects = sorted([p.name for p in self.data_root.iterdir() if _is_subject_dir(p)])
        if subjects is None:
            subjects = all_subjects
        self.subjects = subjects

        self.samples: List[Sample] = []
        self._index()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split} in {self.data_root}")

    def _index(self) -> None:    # Scan folder tree and build sample list.

        for subj in self.subjects:
            subj_dir = self.data_root / subj
            if not subj_dir.exists():
                continue

            for gesture in GESTURES:
                gdir = subj_dir / gesture
                if not gdir.exists():
                    continue
                label = GESTURE_TO_IDX[gesture]

                for clip_dir in sorted(gdir.glob("clip*")):
                    rgb_dir = clip_dir / "rgb"
                    depth_dir = clip_dir / "depth"
                    ann_dir = clip_dir / "annotation"
                    if not rgb_dir.exists() or not depth_dir.exists():
                        continue

                    # All rgb frames
                    rgb_frames = sorted(rgb_dir.glob("frame_*.png"))
                    for rgb_path in rgb_frames:
                        depth_path = depth_dir / rgb_path.name
                        if not depth_path.exists():
                            continue

                        mask_path = (ann_dir / rgb_path.name) if ann_dir.exists() else None
                        has_mask = (mask_path is not None and mask_path.exists())

                        if self.split in {"val", "test"}:
                            if has_mask:
                                self.samples.append(Sample(rgb_path, depth_path, label, mask_path))
                        else:
                            # train
                            if has_mask:
                                self.samples.append(Sample(rgb_path, depth_path, label, mask_path))
                            elif self.include_unlabeled_for_cls:
                                self.samples.append(Sample(rgb_path, depth_path, label, None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        rgb = _load_rgb(s.rgb_path)
        depth = _load_depth_png(s.depth_path)  # HxW float [0,1]

        mask01 = None
        if s.mask_path is not None and s.mask_path.exists():
            mask01 = _load_mask(s.mask_path)  # HxW {0,1}

        # Augmentation (train only) 
        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                rgb = TF.hflip(rgb)
                depth = np.fliplr(depth).copy()
                if mask01 is not None:
                    mask01 = np.fliplr(mask01).copy()

        # Resize+Pad to square 
        rgb = _resize_pad_pil(rgb, self.target_size, is_mask=False)
        depth_pil = Image.fromarray((depth * 255.0).astype(np.uint8))  # temporary for geometry ops
        depth_pil = _resize_pad_pil(depth_pil, self.target_size, is_mask=False)
        depth = np.array(depth_pil).astype(np.float32) / 255.0  # back to [0,1]

        if mask01 is not None:
            mask_pil = Image.fromarray((mask01 * 255.0).astype(np.uint8))
            mask_pil = _resize_pad_pil(mask_pil, self.target_size, is_mask=True)
            mask01 = (np.array(mask_pil) > 127).astype(np.float32)

        # To tensor 
        rgb_t = TF.to_tensor(rgb)  # (3,H,W) in [0,1]
        depth_t = torch.from_numpy(depth).unsqueeze(0)  # (1,H,W)

        # Normalize depth to [-1,1] 
        depth_t = (depth_t - 0.5) / 0.5

        x = torch.cat([rgb_t, depth_t], dim=0)  # (4,H,W)

        y_cls = torch.tensor(s.label, dtype=torch.long)

        # Targets for seg/det only if mask exists
        has_mask = mask01 is not None
        if has_mask:
            mask_t = torch.from_numpy(mask01).unsqueeze(0)  # (1,H,W)
            # bbox in normalized xyxy
            ys, xs = np.where(mask01 > 0)
            if len(xs) == 0:
                # rare: empty mask
                box = torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
                has_mask = False
                mask_t = torch.zeros((1, self.target_size, self.target_size), dtype=torch.float32)
            else:
                x1, x2 = float(xs.min()), float(xs.max() + 1)
                y1, y2 = float(ys.min()), float(ys.max() + 1)
                # normalize to [0,1]
                S = float(self.target_size)
                box = torch.tensor([x1 / S, y1 / S, x2 / S, y2 / S], dtype=torch.float32)
        else:
            mask_t = torch.zeros((1, self.target_size, self.target_size), dtype=torch.float32)
            box = torch.tensor([-1, -1, -1, -1], dtype=torch.float32)

        return {
            "x": x,
            "y_cls": y_cls,
            "mask": mask_t,
            "box": box,
            "has_mask": torch.tensor(1 if has_mask else 0, dtype=torch.long),
            "meta": {
                "rgb_path": str(s.rgb_path),
                "depth_path": str(s.depth_path),
                "mask_path": str(s.mask_path) if s.mask_path is not None else "",
            }
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y_cls = torch.stack([b["y_cls"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    box = torch.stack([b["box"] for b in batch], dim=0)
    has_mask = torch.stack([b["has_mask"] for b in batch], dim=0)
    meta = [b["meta"] for b in batch]
    return {"x": x, "y_cls": y_cls, "mask": mask, "box": box, "has_mask": has_mask, "meta": meta}