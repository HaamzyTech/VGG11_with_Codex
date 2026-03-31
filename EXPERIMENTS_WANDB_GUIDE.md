# Oxford-IIIT Pet: W&B Experiment Guide (Tasks 2.1–2.8)

This guide explains **exactly what to run**, **what to change in code**, and **how to interpret results** for each required experiment.

> Audience: beginners and non-technical users.  
> Goal: finish Tasks 2.1 to 2.8 with reproducible W&B logs and clear conclusions.

---

## 1) Before you start (one-time setup)

### 1.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.2 Login to Weights & Biases (W&B)

```bash
wandb login
```

Paste your API key when asked.

### 1.3 Create one W&B project for everything

Use one project name in all runs, e.g.:

- `oxford-pets-multitask`

This keeps all comparisons in one dashboard.

---

## 2) Shared command template (classification)

Use this base command and only change what is mentioned in each task:

```bash
python train.py \
  --data-root /path/to/oxford-iiit-pet \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --use-wandb \
  --wandb-project oxford-pets-multitask \
  --wandb-run-name <your_run_name>
```

---

## 3) Task 2.1 — BatchNorm effect

## Objective
Compare **with BatchNorm** vs **without BatchNorm**, then inspect:
1. convergence speed,
2. maximum stable learning rate,
3. activation distribution in 3rd convolution layer.

### 3.1 Run with BatchNorm (current code)

```bash
python train.py --data-root /path/to/oxford-iiit-pet --epochs 30 --lr 1e-3 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name t21_bn_on --log-activations
```

### 3.2 Create a BN-off model variant (small code edit)

Create a file `models/vgg11_no_bn.py` (copy from `models/vgg11.py`) and remove all `nn.BatchNorm2d(...)` layers.

Then create a `VGG11ClassifierNoBN` in `models/classification_no_bn.py` that uses `VGG11EncoderNoBN`.

In `train.py`, replace:

```python
from models.classification import VGG11Classifier
```

with:

```python
from models.classification_no_bn import VGG11ClassifierNoBN as VGG11Classifier
```

Run BN-off experiment:

```bash
python train.py --data-root /path/to/oxford-iiit-pet --epochs 30 --lr 1e-3 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name t21_bn_off --log-activations
```

### 3.3 Find maximum stable learning rate

Run both BN-on and BN-off with multiple LRs:

- `1e-4`, `5e-4`, `1e-3`, `3e-3`

### 3.4 How to interpret

- Faster drop in `train/loss` + `val/loss` = faster convergence.
- Training loss exploding/NaN = unstable learning rate.
- BN usually allows higher LR and smoother activation histograms.

---

## 4) Task 2.2 — Dropout internal dynamics

## Objective
Compare three runs:
1. No Dropout (`p=0.0`)
2. Custom Dropout (`p=0.2`)
3. Custom Dropout (`p=0.5`)

### Commands

```bash
python train.py --data-root /path/to/oxford-iiit-pet --dropout 0.0 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name t22_do_00
python train.py --data-root /path/to/oxford-iiit-pet --dropout 0.2 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name t22_do_02
python train.py --data-root /path/to/oxford-iiit-pet --dropout 0.5 --use-wandb --wandb-project oxford-pets-multitask --wandb-run-name t22_do_05
```

### W&B view

Overlay `train/loss` and `val/loss` for all 3 runs.

### How to interpret

- If training loss is very low but validation loss stays high → overfitting.
- Increasing dropout often raises training loss but improves validation behavior.
- Generalization gap = `val/loss - train/loss`; better regularization usually reduces this gap.

---

## 5) Task 2.3 — Transfer learning showdown (segmentation)

## Objective
Compare 3 strategies on segmentation:
1. strict feature extractor,
2. partial fine-tuning,
3. full fine-tuning.

Use helper in `wandb_experiments.py`:

```python
from wandb_experiments import apply_transfer_strategy
apply_transfer_strategy(model, strategy)
```

Where `strategy` is one of:
- `"strict_feature_extractor"`
- `"partial_finetune"`
- `"full_finetune"`

### Example training-loop snippet (segmentation)

```python
import wandb
from wandb_experiments import dice_score, pixel_accuracy, apply_transfer_strategy

run = wandb.init(project="oxford-pets-multitask", name=f"t23_{strategy}")
apply_transfer_strategy(model, strategy)

# inside each validation epoch:
wandb.log({
    "val/loss": val_loss,
    "val/dice": float(dice_score(val_logits, val_masks)),
    "val/pixel_acc": float(pixel_accuracy(val_logits, val_masks)),
    "time/epoch_sec": epoch_time_sec,
})
```

### How to interpret

- **Strict freeze**: fastest per epoch, usually lower final Dice.
- **Partial fine-tune**: good trade-off (speed vs quality).
- **Full fine-tune**: often best final quality, but slower and needs careful LR.

---

## 6) Task 2.4 — Feature maps (inside the black box)

## Objective
Visualize feature maps from:
- first conv layer,
- last conv layer.

### Example snippet

```python
import torch
import wandb
from models.classification import VGG11Classifier

model = VGG11Classifier().eval().to(device)
image = image_tensor.unsqueeze(0).to(device)  # [1,3,H,W]

acts = {}
convs = [m for m in model.encoder.features if isinstance(m, torch.nn.Conv2d)]

h1 = convs[0].register_forward_hook(lambda m,i,o: acts.setdefault("first", o.detach().cpu()))
h2 = convs[-1].register_forward_hook(lambda m,i,o: acts.setdefault("last", o.detach().cpu()))

with torch.no_grad():
    _ = model(image)

h1.remove(); h2.remove()

# log first 16 channels
for k in ["first", "last"]:
    fmap = acts[k][0, :16]  # [16,H,W]
    wandb.log({f"feature_maps/{k}": [wandb.Image(ch.numpy()) for ch in fmap]})
```

### How to interpret

- Early maps: edges, corners, textures.
- Late maps: coarse semantic regions (pet face/head/body structure).

---

## 7) Task 2.5 — Detection confidence + IoU table

## Objective
Log at least 10 test images with:
- GT box (green),
- predicted box (red),
- confidence,
- IoU.

### Use helper

```python
from wandb_experiments import log_detection_table

# images: list of numpy images
# pred_boxes / gt_boxes: [xc,yc,w,h] normalized
# conf_scores: list of floats
log_detection_table(wandb, images, pred_boxes, gt_boxes, conf_scores)
```

### Confidence definition (simple and acceptable)

If your localizer does not output confidence directly, use a proxy confidence such as:
- `confidence = max softmax probability` from classification branch, or
- `confidence = exp(-L1_box_error)`.

Just clearly document which one you used.

### Failure-case interpretation

Pick one high-confidence / low-IoU sample and explain likely cause:
- occlusion,
- small object,
- cluttered background,
- unusual pose.

---

## 8) Task 2.6 — Dice vs pixel accuracy

## Objective
For 5 samples, log:
1. input image,
2. GT trimap,
3. predicted trimap.

### Use helper

```python
from wandb_experiments import log_segmentation_samples
log_segmentation_samples(wandb, images, gt_masks, pred_masks, max_samples=5)
```

Track metrics each epoch:

```python
wandb.log({
    "val/dice": float(dice_value),
    "val/pixel_acc": float(pixel_acc_value),
})
```

### How to interpret (important)

Pixel accuracy can look high even with poor foreground quality because background dominates pixel counts.  
Dice is better for imbalance because it directly measures overlap of predicted vs true region.

---

## 9) Task 2.7 — Final pipeline showcase on novel web images

## Objective
Run your final multitask model on **3 pet images not in Oxford-IIIT Pet**.

### Steps
1. Download 3 images from the internet.
2. Preprocess to model input size.
3. Run one forward pass in `MultiTaskPerceptionModel`.
4. Draw:
   - predicted bbox,
   - predicted breed label,
   - segmentation overlay.
5. Upload final composed images to W&B report.

### Example snippet (high-level)

```python
outputs = multitask_model(x)
cls_logits = outputs["classification"]
box = outputs["localization"]
seg = outputs["segmentation"].argmax(1)
# draw and wandb.log(wandb.Image(...))
```

### How to interpret

Discuss where generalization failed:
- lighting shift,
- shadows,
- unusual camera angle,
- heavy background clutter.

---

## 10) Task 2.8 — Meta-analysis and reflection (report template)

Use this structure in your W&B report:

1. **All curves in one place**
   - Classification: loss + macro-F1/accuracy
   - Localization: IoU/mAP + loss
   - Segmentation: Dice + pixel accuracy + loss

2. **Architectural reflection**
   - BN placement impact on stability and LR.
   - Custom Dropout impact on generalization gap.

3. **Encoder adaptation reflection**
   - Strict vs partial vs full fine-tuning.
   - Whether shared backbone showed task interference.

4. **Loss reflection**
   - Why IoU loss helped localization.
   - Why CE + Dice tracking is useful for segmentation imbalance.

5. **Final takeaway**
   - Best setting chosen,
   - why it wins empirically,
   - and where it still fails.

---

## 11) Recommended run names (copy/paste)

- `t21_bn_on`, `t21_bn_off`
- `t22_do_00`, `t22_do_02`, `t22_do_05`
- `t23_strict_feature_extractor`, `t23_partial_finetune`, `t23_full_finetune`
- `t24_feature_maps`
- `t25_detection_table`
- `t26_seg_dice_vs_pixacc`
- `t27_novel_images`
- `t28_meta_analysis`

These names make W&B grouping and overlays much easier.
