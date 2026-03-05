"""
Baseline RGB-D multitask network: detection + segmentation + classification.

- 4-channel input RGB-D.
- Shared encoder, U-Net-like decoder for segmentation.
- Global pooled bottleneck for classification and bbox regression.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RGBDMultitaskNet(nn.Module):
    def __init__(self, num_classes: int = 10, base: int = 32, dropout: float = 0.2):
        super().__init__()
        # Encoder 
        self.stem = nn.Sequential(
            ConvBNReLU(4, base, 3, 1, 1),
            ConvBNReLU(base, base, 3, 1, 1),
        )
        self.enc1 = nn.Sequential(
            ConvBNReLU(base, base * 2, 3, 2, 1),
            ConvBNReLU(base * 2, base * 2, 3, 1, 1),
        )
        self.enc2 = nn.Sequential(
            ConvBNReLU(base * 2, base * 4, 3, 2, 1),
            ConvBNReLU(base * 4, base * 4, 3, 1, 1),
        )
        self.enc3 = nn.Sequential(
            ConvBNReLU(base * 4, base * 8, 3, 2, 1),
            ConvBNReLU(base * 8, base * 8, 3, 1, 1),
        )
        self.bottleneck = nn.Sequential(
            ConvBNReLU(base * 8, base * 8, 3, 1, 1),
        )

        # Decoder 
        self.dec3 = nn.Sequential(
            ConvBNReLU(base * 8 + base * 4, base * 4, 3, 1, 1),
            ConvBNReLU(base * 4, base * 4, 3, 1, 1),
        )
        self.dec2 = nn.Sequential(
            ConvBNReLU(base * 4 + base * 2, base * 2, 3, 1, 1),
            ConvBNReLU(base * 2, base * 2, 3, 1, 1),
        )
        self.dec1 = nn.Sequential(
            ConvBNReLU(base * 2 + base, base, 3, 1, 1),
            ConvBNReLU(base, base, 3, 1, 1),
        )
        self.seg_head = nn.Conv2d(base, 1, kernel_size=1)

        # Heads for classification + bbox 
        feat_dim = base * 8
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )
        self.box_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 4),
        )

    def forward(self, x: torch.Tensor) -> dict:
        s0 = self.stem(x)     # (B, base, H, W)
        s1 = self.enc1(s0)    # (B, 2b, H/2, W/2)
        s2 = self.enc2(s1)    # (B, 4b, H/4, W/4)
        s3 = self.enc3(s2)    # (B, 8b, H/8, W/8)
        b = self.bottleneck(s3)

        # Global pooled features 
        gp = F.adaptive_avg_pool2d(b, 1).flatten(1)
        cls_logits = self.cls_head(gp)
        
        #  valid box order (x1<x2, y1<y2) 
        box_raw = torch.sigmoid(self.box_head(gp))  # (B,4) in [0,1]

        x1 = torch.minimum(box_raw[:, 0], box_raw[:, 2])
        y1 = torch.minimum(box_raw[:, 1], box_raw[:, 3])
        x2 = torch.maximum(box_raw[:, 0], box_raw[:, 2])
        y2 = torch.maximum(box_raw[:, 1], box_raw[:, 3])

        # avoid zero-area boxes 
        eps = 1e-4
        x2 = torch.maximum(x2, x1 + eps)
        y2 = torch.maximum(y2, y1 + eps)
        x2 = torch.minimum(x2, torch.ones_like(x2))
        y2 = torch.minimum(y2, torch.ones_like(y2))

        box = torch.stack([x1, y1, x2, y2], dim=1)

        # Decoder for segmentation 
        u3 = F.interpolate(b, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.dec3(torch.cat([u3, s2], dim=1))

        u2 = F.interpolate(u3, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(torch.cat([u2, s1], dim=1))

        u1 = F.interpolate(u2, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, s0], dim=1))

        seg_logits = self.seg_head(u1)

        return {"seg_logits": seg_logits, "cls_logits": cls_logits, "box": box}



# Innovation: Mask-guided pooling + Inference-time Mask2Box
def mask_to_box_hard(mask_prob: torch.Tensor, thr: float = 0.5):
    """
    mask_prob: (B,1,H,W) in [0,1]
    return:
      box_mask: (B,4) normalized xyxy in [0,1]
      valid:    (B,) bool, whether mask has any foreground

    inference-time mask->bbox using thresholded mask.
    """
    B, _, H, W = mask_prob.shape
    m = (mask_prob > thr)

    area = m.flatten(1).sum(dim=1)         
    valid = area > 0

    cols = m.any(dim=2).squeeze(1)        
    rows = m.any(dim=3).squeeze(1)        

    # first true index
    x1 = cols.float().argmax(dim=1)       
    y1 = rows.float().argmax(dim=1)       

    # last true index
    x2 = (W - 1) - cols.flip(1).float().argmax(dim=1)
    y2 = (H - 1) - rows.flip(1).float().argmax(dim=1)

    # normalize, make x2,y2 exclusive (+1)
    x1n = x1.float() / float(W)
    y1n = y1.float() / float(H)
    x2n = (x2.float() + 1.0) / float(W)
    y2n = (y2.float() + 1.0) / float(H)

    # enforce valid order + non-zero area
    eps = 1e-4
    x_left = torch.minimum(x1n, x2n)
    y_top = torch.minimum(y1n, y2n)
    x_right = torch.maximum(x1n, x2n)
    y_bottom = torch.maximum(y1n, y2n)

    x_right = torch.maximum(x_right, x_left + eps)
    y_bottom = torch.maximum(y_bottom, y_top + eps)
    x_right = torch.clamp(x_right, max=1.0)
    y_bottom = torch.clamp(y_bottom, max=1.0)

    box = torch.stack([x_left, y_top, x_right, y_bottom], dim=1)
    return box, valid


class RGBDMultitaskNet_Innovation(RGBDMultitaskNet):
    """
    - Mask-guided pooling for cls/det features
    - Inference-time mask->box refinement (stable + effective for detection)
    """
    def __init__(
        self,
        num_classes: int = 10,
        base: int = 32,
        dropout: float = 0.2,
        detach_attn: bool = True,
        box_mix: float = 0.25,
        mask_thr: float = 0.5
    ):
        super().__init__(num_classes=num_classes, base=base, dropout=dropout)
        self.detach_attn = detach_attn
        self.box_mix = box_mix
        self.mask_thr = mask_thr

    def forward(self, x: torch.Tensor) -> dict:
        # enc

        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)  

        # Decoder -> seg 
        u3 = F.interpolate(b, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.dec3(torch.cat([u3, s2], dim=1))

        u2 = F.interpolate(u3, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(torch.cat([u2, s1], dim=1))

        u1 = F.interpolate(u2, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, s0], dim=1))

        seg_logits = self.seg_head(u1)  
        A = torch.sigmoid(seg_logits)   

        # mask-guided pooling 
        A_pool = A.detach() if self.detach_attn else A
        A_b = F.interpolate(A_pool, size=b.shape[-2:], mode="bilinear", align_corners=False)  

        eps = 1e-6
        den = A_b.sum(dim=(2, 3)) + eps  # (B,1)
        num = (b * A_b).sum(dim=(2, 3))  # (B,C)
        gp_mask = num / den  # (B,C)

        gap = F.adaptive_avg_pool2d(b, 1).flatten(1)   # (B,C)
        mask_area = A.sum(dim=(2, 3))                  # (B,1)
        use_gap = (mask_area < 1e-3).float()           # (B,1)
        gp = gp_mask * (1.0 - use_gap) + gap * use_gap

        cls_logits = self.cls_head(gp)

        # box_head 
        box_raw = torch.sigmoid(self.box_head(gp))    
        x1 = torch.minimum(box_raw[:, 0], box_raw[:, 2])
        y1 = torch.minimum(box_raw[:, 1], box_raw[:, 3])
        x2 = torch.maximum(box_raw[:, 0], box_raw[:, 2])
        y2 = torch.maximum(box_raw[:, 1], box_raw[:, 3])

        eps_box = 1e-4
        x2 = torch.maximum(x2, x1 + eps_box)
        y2 = torch.maximum(y2, y1 + eps_box)
        x2 = torch.clamp(x2, max=1.0)
        y2 = torch.clamp(y2, max=1.0)

        box_head = torch.stack([x1, y1, x2, y2], dim=1)

        # mask->box refinement
        if not self.training:
            box_mask, valid = mask_to_box_hard(A, thr=self.mask_thr)  # (B,4), (B,)
            valid = valid.unsqueeze(1).float()     # (B,1)

            box = self.box_mix * box_head + (1.0 - self.box_mix) * box_mask
            # if mask invalid, fallback to head
            box = box * valid + box_head * (1.0 - valid)
        else:
            box = box_head

        return {"seg_logits": seg_logits, "cls_logits": cls_logits, "box": box}

# variant (baseline/innovation)
def build_model(variant: str, num_classes: int, base: int = 32) -> nn.Module:

    variant = variant.lower()
    if variant in ["baseline", "rgbd", "rgb-d"]:
        return RGBDMultitaskNet(num_classes=num_classes, base=base)
    if variant in ["depth_innovation", "inn", "innovation"]:
        return RGBDMultitaskNet_Innovation(num_classes=num_classes, base=base)
    raise ValueError(f"Unknown variant: {variant}")