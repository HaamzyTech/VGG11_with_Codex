"""VGG11 encoder."""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    VGG11 channel progression:
    [64] -> [128] -> [256, 256] -> [512, 512] -> [512, 512]
    with max-pooling between each stage.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Keep a canonical torchvision-like `features` stack so shape checks
        # based on layer order/indexing remain straightforward.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        f1 = self.features[2](self.features[1](self.features[0](x)))
        p1 = self.features[3](f1)

        f2 = self.features[6](self.features[5](self.features[4](p1)))
        p2 = self.features[7](f2)

        f3 = self.features[13](self.features[12](self.features[11](self.features[10](self.features[9](self.features[8](p2))))))
        p3 = self.features[14](f3)

        f4 = self.features[20](self.features[19](self.features[18](self.features[17](self.features[16](self.features[15](p3))))))
        p4 = self.features[21](f4)

        f5 = self.features[27](self.features[26](self.features[25](self.features[24](self.features[23](self.features[22](p4))))))
        bottleneck = self.features[28](f5)

        if not return_features:
            return bottleneck

        feature_dict = {
            "enc1": f1,
            "enc2": f2,
            "enc3": f3,
            "enc4": f4,
            "enc5": f5,
        }
        return bottleneck, feature_dict
