"""Training entrypoint for Oxford-IIIT Pet breed classification with optional W&B logging."""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def _get_third_conv_module(model: VGG11Classifier) -> nn.Module:
    convs = [m for m in model.encoder.features if isinstance(m, nn.Conv2d)]
    if len(convs) < 3:
        raise RuntimeError("Expected at least 3 convolution layers in encoder")
    return convs[2]


def _capture_activation(model: VGG11Classifier, batch: torch.Tensor) -> torch.Tensor:
    activations = {}

    def hook(_module, _inp, out):
        activations["act"] = out.detach().flatten().cpu()

    h = _get_third_conv_module(model).register_forward_hook(hook)
    with torch.no_grad():
        _ = model(batch)
    h.remove()
    return activations["act"]


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), labels) * batch_size

    return total_loss / total_samples, total_acc / total_samples


def maybe_init_wandb(args) -> Optional[object]:
    if not args.use_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=args.wandb_run_name if args.wandb_run_name else None,
        config=vars(args),
    )
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Path to Oxford-IIIT Pet root directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="oxford-pets-vgg11")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--log-activations", action="store_true", help="Log 3rd-conv activation histogram to W&B")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    wandb_run = maybe_init_wandb(args)

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", transform=train_tfms)
    val_size = int(len(full_train) * args.val_ratio)
    train_size = len(full_train) - val_size

    gen = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=gen)
    val_subset.dataset.transform = eval_tfms

    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", transform=eval_tfms)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "vgg11_classifier_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if wandb_run is not None:
            payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if args.log_activations:
                images, _ = next(iter(val_loader))
                images = images.to(device)
                acts = _capture_activation(model, images)
                import wandb
                payload["activations/conv3_hist"] = wandb.Histogram(acts.numpy())
            wandb_run.log(payload)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved new best checkpoint to: {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(f"Test metrics | loss={test_loss:.4f} acc={test_acc:.4f}")

    if wandb_run is not None:
        wandb_run.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb_run.finish()


if __name__ == "__main__":
    main()
