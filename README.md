# Oxford-IIIT Pet: VGG11 from Scratch (PyTorch)

This repository now includes a **from-scratch VGG11 classifier** for 37 Oxford-IIIT Pet breeds.

## What was implemented

- **VGG11 Encoder (from scratch)** using only standard `torch.nn` modules (`Conv2d`, `ReLU`, `MaxPool2d`).
- **Batch Normalization** integrated after convolutional layers and fully connected layers.
- **Custom Dropout layer** implemented by inheriting from `torch.nn.Module` (without using `torch.nn.Dropout` or `torch.nn.functional.dropout`).
- **Classifier training pipeline** with train/val/test loops.
- **Oxford-IIIT Pet dataset loader** reading official split files.

## Architectural choices and justification

### 1) BatchNorm placement

BatchNorm is placed **immediately after each Conv2d** (and before ReLU), plus after the first two fully connected layers in the classifier head.

**Reasoning**:
- Convolutional BN reduces internal covariate shift and smooths optimization, which is especially useful for deeper conv stacks like VGG.
- BN in fully connected layers stabilizes activation scale entering subsequent nonlinearities, often improving convergence speed and reducing sensitivity to initialization/learning rate.

### 2) Dropout placement

Custom dropout is applied in the classifier head after each ReLU on the first two large FC layers.

**Reasoning**:
- The densest part of VGG is the FC head, where parameter count and co-adaptation risk are highest.
- Applying dropout in FC layers is a classic VGG regularization strategy and typically gives stronger regularization benefits than dropping early convolutional activations.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py \
  --data-root /path/to/oxford-iiit-pet \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --dropout 0.5
```

Expected dataset layout:

```text
/path/to/oxford-iiit-pet/
  images/
    *.jpg
  annotations/
    trainval.txt
    test.txt
```

The best model checkpoint is saved to:

```text
checkpoints/vgg11_classifier_best.pt
```

## Task 2: Object localization extension

- **Encoder adaptation:** `VGG11Localizer` reuses the Task-1 `VGG11Encoder` backbone and optionally loads encoder weights from a trained classifier checkpoint.
- **Freeze vs fine-tune:** default is **fine-tuning** (`freeze_encoder=False`) because localization depends on spatially precise features; freezing is supported for faster training when compute is constrained.
- **Regression head:** a compact MLP head predicts exactly four values `[x_center, y_center, width, height]`.
- **Coordinate scaling:** final `Sigmoid` constrains predictions to `[0, 1]` normalized image coordinates for stable training and bounded outputs.
- **Custom loss:** `IoULoss` is implemented from scratch for center-format boxes with numerically stable eps handling.

