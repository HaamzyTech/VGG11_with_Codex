"""Custom IoU loss."""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Inputs must be shape [B, 4] in (x_center, y_center, width, height) format.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.eps = eps
        self.reduction = reduction

    def _cxcywh_to_xyxy(self, boxes: torch.Tensor):
        cx, cy, w, h = boxes.unbind(dim=-1)
        w = w.clamp_min(self.eps)
        h = h.clamp_min(self.eps)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return x1, y1, x2, y2

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height).
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height).
        """
        if pred_boxes.shape != target_boxes.shape or pred_boxes.ndim != 2 or pred_boxes.size(-1) != 4:
            raise ValueError(
                "pred_boxes and target_boxes must both have shape [B, 4]. "
                f"Got {tuple(pred_boxes.shape)} and {tuple(target_boxes.shape)}"
            )

        px1, py1, px2, py2 = self._cxcywh_to_xyxy(pred_boxes)
        tx1, ty1, tx2, ty2 = self._cxcywh_to_xyxy(target_boxes)

        inter_x1 = torch.maximum(px1, tx1)
        inter_y1 = torch.maximum(py1, ty1)
        inter_x2 = torch.minimum(px2, tx2)
        inter_y2 = torch.minimum(py2, ty2)

        inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
        inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
        inter_area = inter_w * inter_h

        pred_area = (px2 - px1).clamp_min(0.0) * (py2 - py1).clamp_min(0.0)
        target_area = (tx2 - tx1).clamp_min(0.0) * (ty2 - ty1).clamp_min(0.0)
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + self.eps)
        loss = 1.0 - iou

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
