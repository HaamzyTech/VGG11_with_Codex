"""Localization modules."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer.

    Encoder adaptation strategy:
    - Reuse the Task-1 VGG11 convolutional encoder.
    - By default, fine-tune the encoder (``freeze_encoder=False``) because
      localization is spatially sensitive and benefits from adapting features.
    - Optionally freeze encoder weights for faster/cheaper training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        freeze_encoder: bool = False,
        classifier_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        if classifier_checkpoint is not None:
            ckpt_path = Path(classifier_checkpoint)
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            # Supports either plain state-dict or train.py checkpoint dict.
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            classifier = VGG11Classifier(in_channels=in_channels)
            classifier.load_state_dict(state_dict, strict=False)
            self.encoder.load_state_dict(classifier.encoder.state_dict(), strict=True)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # normalized [x_center, y_center, width, height] in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            normalized to [0, 1].
        """
        features = self.encoder(x)
        return self.regressor(features)
