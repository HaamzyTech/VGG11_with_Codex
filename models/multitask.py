"""Unified multi-task model."""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    A single forward pass computes shared encoder features once and branches into:
    - breed classification logits
    - bbox localization regression
    - semantic segmentation logits
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_breeds),
        )

        # Localization head
        self.loc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # Segmentation decoder head
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _DoubleConv(512 + 512, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(64 + 64, 64)

        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bottleneck, feats = self.encoder(x, return_features=True)

        cls_feat = self.cls_pool(bottleneck)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_logits = self.cls_head(cls_feat)

        loc_boxes = self.loc_head(bottleneck)

        d5 = self.up5(bottleneck)
        d5 = torch.cat([d5, feats["enc5"]], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([d4, feats["enc4"]], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, feats["enc3"]], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, feats["enc2"]], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, feats["enc1"]], dim=1)
        d1 = self.dec1(d1)
        seg_logits = self.seg_head(d1)

        return {
            "classification": cls_logits,
            "localization": loc_boxes,
            "segmentation": seg_logits,
        }
