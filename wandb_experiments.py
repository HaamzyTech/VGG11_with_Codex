"""W&B utilities for the Oxford-IIIT Pet experiments (Tasks 2.1 - 2.8)."""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from losses.iou_loss import IoULoss


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean()


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    scores = []
    num_classes = logits.shape[1]
    for c in range(num_classes):
        pred_c = (pred == c).float()
        tgt_c = (target == c).float()
        inter = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()
        scores.append((2.0 * inter + eps) / (denom + eps))
    return torch.stack(scores).mean()


def apply_transfer_strategy(model: torch.nn.Module, strategy: str) -> None:
    """Apply segmentation transfer strategy for Task 2.3.

    strategy:
      - 'strict_feature_extractor': freeze whole encoder
      - 'partial_finetune': freeze early encoder blocks, unfreeze later blocks
      - 'full_finetune': unfreeze everything
    """
    if strategy not in {"strict_feature_extractor", "partial_finetune", "full_finetune"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    for p in model.encoder.parameters():
        p.requires_grad = True

    if strategy == "strict_feature_extractor":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == "partial_finetune":
        # VGG11 features indices: early blocks roughly < 15.
        for idx, module in enumerate(model.encoder.features):
            if idx < 15:
                for p in module.parameters():
                    p.requires_grad = False


def log_detection_table(
    wandb,
    images: Iterable[np.ndarray],
    pred_boxes: Iterable[List[float]],
    gt_boxes: Iterable[List[float]],
    conf_scores: Iterable[float],
):
    """Log at least 10 detection samples with IoU and confidence to W&B."""
    iou_fn = IoULoss(reduction="none")
    table = wandb.Table(columns=["image", "pred_box", "gt_box", "confidence", "iou"])

    for img, pred, gt, conf in zip(images, pred_boxes, gt_boxes, conf_scores):
        pred_t = torch.tensor([pred], dtype=torch.float32)
        gt_t = torch.tensor([gt], dtype=torch.float32)
        iou = float(1.0 - iou_fn(pred_t, gt_t).item())

        boxes = {
            "predictions": {
                "box_data": [
                    {
                        "position": {
                            "middle": [pred[0], pred[1]],
                            "width": pred[2],
                            "height": pred[3],
                        },
                        "class_id": 0,
                        "box_caption": f"Pred conf={conf:.3f} IoU={iou:.3f}",
                    }
                ],
                "class_labels": {0: "pred"},
            },
            "ground_truth": {
                "box_data": [
                    {
                        "position": {
                            "middle": [gt[0], gt[1]],
                            "width": gt[2],
                            "height": gt[3],
                        },
                        "class_id": 1,
                        "box_caption": "Ground Truth",
                    }
                ],
                "class_labels": {1: "gt"},
            },
        }

        wb_img = wandb.Image(img, boxes=boxes)
        table.add_data(wb_img, pred, gt, conf, iou)

    wandb.log({"detection/table": table})


def log_segmentation_samples(wandb, images, gt_masks, pred_masks, max_samples: int = 5):
    rows = []
    for idx, (img, gt, pred) in enumerate(zip(images, gt_masks, pred_masks)):
        if idx >= max_samples:
            break
        rows.append(
            [
                wandb.Image(img, caption=f"sample_{idx}_input"),
                wandb.Image(gt, caption=f"sample_{idx}_gt"),
                wandb.Image(pred, caption=f"sample_{idx}_pred"),
            ]
        )
    table = wandb.Table(columns=["image", "gt_trimap", "pred_trimap"], data=rows)
    wandb.log({"segmentation/examples": table})


def activation_histogram(model, input_tensor: torch.Tensor, conv_index: int = 2) -> np.ndarray:
    """Extract flattened activations from a chosen conv layer for histogram plotting."""
    convs = [m for m in model.encoder.features if isinstance(m, torch.nn.Conv2d)]
    activations: Dict[str, torch.Tensor] = {}

    def hook(_m, _i, o):
        activations["a"] = o.detach().flatten().cpu()

    handle = convs[conv_index].register_forward_hook(hook)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()
    return activations["a"].numpy()
