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


## Running on a PBS GPU cluster

If your HPC uses PBS/Torque (or PBS Pro), the easiest pattern is to submit a batch script that:

1. Activates your Python environment.
2. Verifies CUDA is visible.
3. Launches `train.py` with your dataset path.

### 1) Create a job script

Save this as `train_vgg11.pbs` and edit placeholders for your cluster (`<ACCOUNT>`, queue name, module names, and paths):

```bash
#!/bin/bash
#PBS -N vgg11_pets_train
#PBS -A <ACCOUNT>
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"

# Optional: load site-specific modules
module purge
module load cuda/12.1
module load python/3.10

# Create/activate virtual environment (first run)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Confirm GPU visibility
python - <<'CHECK'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
CHECK

# Launch training
python train.py \
  --data-root /path/to/oxford-iiit-pet \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 8 \
  --lr 1e-3 \
  --dropout 0.5
```

Submit with:

```bash
qsub train_vgg11.pbs
```

Monitor with:

```bash
qstat -u $USER
```

### 2) Interactive debugging on a GPU node (optional)

Useful for quick checks before long runs:

```bash
qsub -I -q gpu -l select=1:ncpus=4:ngpus=1:mem=16gb,walltime=02:00:00
```

Then activate your env and run a short training smoke test (e.g., fewer epochs, smaller batch size).

### 3) Common PBS cluster tips

- Use `$PBS_O_WORKDIR` to run from the directory where you submitted `qsub`.
- Keep datasets on a high-throughput shared filesystem (not small home quotas).
- Increase `--num-workers` gradually to match allocated CPUs and avoid I/O bottlenecks.
- If your cluster uses `conda` instead of `venv`, replace activation commands accordingly.
- Some clusters require different resource keys (`nodes:ppn:gpus` vs `select=...`). If `qsub` rejects your script, check your site docs and adapt the `#PBS -l` line.

## Task 2: Object localization extension

- **Encoder adaptation:** `VGG11Localizer` reuses the Task-1 `VGG11Encoder` backbone and optionally loads encoder weights from a trained classifier checkpoint.
- **Freeze vs fine-tune:** default is **fine-tuning** (`freeze_encoder=False`) because localization depends on spatially precise features; freezing is supported for faster training when compute is constrained.
- **Regression head:** a compact MLP head predicts exactly four values `[x_center, y_center, width, height]`.
- **Coordinate scaling:** final `Sigmoid` constrains predictions to `[0, 1]` normalized image coordinates for stable training and bounded outputs.
- **Custom loss:** `IoULoss` is implemented from scratch for center-format boxes with numerically stable eps handling.

## Task 3: U-Net style semantic segmentation

- **Architecture:** `VGG11UNet` uses `VGG11Encoder` as the contracting path and a symmetric expansive decoder.
- **Learnable upsampling:** each decoder stage uses `ConvTranspose2d` (stride 2) for upsampling.
- **Feature fusion:** upsampled decoder features are concatenated with spatially aligned encoder skip maps at every level.
- **Loss choice:** use **pixel-wise `CrossEntropyLoss`** over segmentation logits. This is appropriate because trimap segmentation is a 3-class per-pixel classification problem (foreground / boundary / background), and CE provides stable gradients for mutually-exclusive classes without requiring manual thresholding.

## Task 4: Unified multi-task pipeline

`MultiTaskPerceptionModel` performs a **single shared-backbone forward pass** and branches into three heads:
- **Classification head:** outputs 37-class breed logits.
- **Localization head:** outputs normalized bbox coordinates `[x_center, y_center, width, height]`.
- **Segmentation head:** outputs dense segmentation logits map.

The model returns a dictionary with keys:
`{"classification": ..., "localization": ..., "segmentation": ...}`.


## W&B integration and experiment playbook (Tasks 2.1 - 2.8)

### 0) One-time setup

```bash
pip install -r requirements.txt
wandb login
```

Use a single project name for all phases, e.g. `oxford-pets-multitask`.

---

### 2.1 BatchNorm effect (convergence + stable LR + activation distribution)

1. **Run without BN** (set a BN-disabled variant in code or branch where BN layers are removed).
2. **Run with BN** (current implementation).
3. Log 3rd-conv activation histogram with `--log-activations`.

Example command:

```bash
python train.py --data-root /path/to/pets --epochs 30 --lr 1e-3 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name cls_bn_on --log-activations
```

Repeat with BN-off model and multiple learning rates (`1e-4, 5e-4, 1e-3, 3e-3`) to find max stable LR.

**Expected analysis:** BN should reduce activation drift, speed convergence, and allow a higher stable LR before divergence.

---

### 2.2 Dropout internal dynamics (p=0.0 / 0.2 / 0.5)

Run three classification jobs:

```bash
python train.py --data-root /path/to/pets --dropout 0.0 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name cls_do_00
python train.py --data-root /path/to/pets --dropout 0.2 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name cls_do_02
python train.py --data-root /path/to/pets --dropout 0.5 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name cls_do_05
```

In W&B, overlay `train/loss` and `val/loss` across runs.

**Expected analysis:** Increasing dropout usually raises training loss but can reduce overfitting and narrow train-vs-val gap.

---

### 2.3 Transfer-learning showdown for segmentation

Use `apply_transfer_strategy(...)` in `wandb_experiments.py` with:
- `strict_feature_extractor`
- `partial_finetune`
- `full_finetune`

Track per epoch:
- `val/loss`
- `val/dice`
- `val/pixel_acc`
- `time/epoch_sec`

Code snippet:

```python
from wandb_experiments import apply_transfer_strategy, dice_score, pixel_accuracy

apply_transfer_strategy(model, strategy)  # one of the 3 modes
# during validation: log dice_score(logits, masks), pixel_accuracy(logits, masks)
```

**Expected analysis:** strict freezing converges faster per epoch but usually underfits domain details; partial/full fine-tuning typically improves final Dice.

---

### 2.4 Feature-map visualization (first vs last conv)

Use a forward hook on first and last conv of classifier encoder, then log maps to W&B as images.

Code snippet:

```python
convs = [m for m in model.encoder.features if isinstance(m, torch.nn.Conv2d)]
first_conv, last_conv = convs[0], convs[-1]
```

**Expected analysis:** early maps emphasize edges/textures; late maps emphasize coarse semantic regions (e.g., head/ears/snout structure).

---

### 2.5 Detection table with confidence + IoU

Use `log_detection_table(...)` from `wandb_experiments.py` for >=10 test images. It logs per-row:
- overlaid GT and predicted boxes
- confidence score
- IoU

```python
from wandb_experiments import log_detection_table
log_detection_table(wandb, images, pred_boxes, gt_boxes, conf_scores)
```

Failure-case writeup: describe occlusion, tiny scale, unusual pose, or cluttered background.

---

### 2.6 Segmentation: Dice vs Pixel Accuracy

Use `log_segmentation_samples(...)` for 5 qualitative samples and track both metrics every epoch.

```python
from wandb_experiments import log_segmentation_samples, dice_score, pixel_accuracy
```

**Mathematical explanation:** with class imbalance, background dominates pixels, so pixel accuracy can stay high even if foreground overlap is poor. Dice directly emphasizes overlap quality and is more sensitive to minority regions.

---

### 2.7 Final pipeline showcase on 3 internet images

For each novel image:
1. Run multitask forward.
2. Draw bbox + class label + segmentation overlay.
3. Upload result image to W&B report.

Discuss generalization limits: lighting shift, backgrounds, uncommon breed poses/scales.

---

### 2.8 Meta-analysis template (put in W&B report)

1. **Comprehensive curves:** training/validation loss and metrics for all tasks.
2. **Architecture reflection:** BN + custom dropout effects on stability/generalization.
3. **Encoder adaptation reflection:** frozen vs partial vs full fine-tuning trade-offs and task interference.
4. **Loss reflection:** why CE (segmentation) + IoU (localization) behaved as observed.

Recommended report sections:
- Setup and splits
- Ablations (BN/dropout/transfer strategy)
- Quantitative comparisons
- Qualitative examples (feature maps, boxes, trimaps)
- Final conclusions and failure analysis

For the full beginner-friendly experiment guide, see `EXPERIMENTS_WANDB_GUIDE.md`.
