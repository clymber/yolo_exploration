# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (yolo-dev)
#     language: python
#     name: yolo-dev
# ---

# %% [markdown]
# # Train yolo11 model for object detection on custom dataset
#
# This notebook trains YOLO11 object detection models locally on macOS using Apple
# Silicon MPS. The experiment follows the Roboflow YOLO11 custom dataset notebook, but
# adapts it for a local Conda/Jupytext workflow.

# %% [markdown]
# ## Import and setup

# %%
"""
Train yolo11 model(s) for object detection on custom dataset(s) from COCO.
"""

import csv
import textwrap
from functools import partial

from IPython.display import Image as IPyImage
from IPython.display import display

from yolo_exploration import (
    DEVICE,
    PROJECT_ROOT,
    cache_download,
    configure_ultralytics_privacy,
    directory_tree,
    relative_to_project_root,
)

configure_ultralytics_privacy()  # must be called before importing ultralytics
from ultralytics import YOLO  # noqa: E402

# %%
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
DATA_EXTERNAL = DATA_DIR / "external"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

for directory in (PRETRAINED_DIR, RUNS_DIR, DATA_EXTERNAL, PREDICTIONS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

YOLO11N_MODEL = PRETRAINED_DIR / "yolo11n.pt"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("RUNS_DIR:", relative_to_project_root(RUNS_DIR))
print("YOLO11N_MODEL:", relative_to_project_root(YOLO11N_MODEL))

# %% [markdown]
# ## Inference with a pretrained model
#
# ### Download and cache a test image
#
# Download a sample image, run the pretrained YOLO11n model against it, and save the
# prediction output under `outputs/predictions/` for a quick sanity check.

# %%
DOG_IMAGE = cache_download(
    DATA_EXTERNAL / "dog.jpeg",
    "https://media.roboflow.com/notebooks/examples/dog.jpeg"
)

print("Dog image:", relative_to_project_root(DOG_IMAGE))
display(IPyImage(filename=str(DOG_IMAGE), width=400))

# %% [markdown]
# ### Inference with pretrained YOLO11

# %%
model = YOLO(YOLO11N_MODEL)
pred_results = model.predict(
    source=DOG_IMAGE,
    device=DEVICE,
    project=PREDICTIONS_DIR,
    name="nb02_pretrained_dog_prediction",
    exist_ok=True,
    save=True,
    show=False,
)

print("Prediction completed.")
print("Number of result objects:", len(pred_results))

# %% [markdown]
# ### Inspect YOLO prediction output
#
# The `model.predict()` method returns a list of
# `ultralytics.yolo.engine.results.Results` objects, which contain the prediction
# results and metadata for each input image.

# %%
results = pred_results[0]

print("Image path:", relative_to_project_root(results.path))
print("Original image shape:", results.orig_shape)

# Display the predicted image with bounding boxes overlaid.
pred_image_path = str(PREDICTIONS_DIR / "nb02_pretrained_dog_prediction.jpg")
pred_image_path = results.save(pred_image_path)
print("Predicted image with bounding boxes:", relative_to_project_root(pred_image_path))
display(IPyImage(filename=str(pred_image_path), width=400))

# %%
shorten_text = partial(textwrap.shorten, width=120, placeholder="...")
boxes = results.boxes

if boxes is not None:
    print("Number of predicted boxes:", len(boxes))
    print(shorten_text(f"Class IDs: {boxes.cls}"))
    print(shorten_text(f"Confidences: {boxes.conf.tolist()}"))
    print(shorten_text(f"Bounding box coordinates: {boxes.xyxy.tolist()}"))
else:
    print("No boxes predicted.")

# %% [markdown]
# To map class IDs to names:

# %%
names = results.names

if boxes is not None:
    for cls_id, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy, strict=True):
        cls_id = int(cls_id)
        label = names[cls_id]
        conf = float(conf)
        print(f"{label}\t: confidence={conf:.2f}, box={xyxy.tolist()}")

# %% [markdown]
# ## Train YOLO11n on COCO128
#
# Now that pretrained inference works, we train YOLO11n on a small COCO128 dataset (made
# from the first 128 images of COCO train2027) as a local training smoke test.
#
# ### Fine-tune the current YOLO11n model on a small object-detection dataset
#
# The `model.train(...)` starts training from those pretrained weights, not from scratch.
#
# So conceptually:
#
# ```plaintext
# pretrained YOLO11n checkpoint
#         ↓
# train for 5 epochs on COCO128
#         ↓
# save fine-tuned model to outputs/runs/nb02_coco128_yolo11n_smoke_test/
# ```
#
# `coco128.yaml` is a built-in dataset configuration file known by Ultralytics.
# Ultralytics searches its internal dataset configuration files for it. The official
# Ultralytics repository includes ultralytics/cfg/datasets/coco128.yaml.
#
#

# %%
results = model.train(
    data="coco128.yaml",
    epochs=5,
    imgsz=640,
    batch=2,
    device=DEVICE,
    project=str(PROJECT_ROOT / "outputs" / "runs"),
    name="nb02_coco128_yolo11n_smoke_test",
    exist_ok=True,
    plots=True,
    conf=0.25,      # filters weak boxes earlier
    iou=0.7,        # NMS overlap threshold
    max_det=100,    # fewer final detections per image
)

# %% [markdown]
# ### Inspect the training output

# %%
output_dir = RUNS_DIR / "nb02_coco128_yolo11n_smoke_test"
print(directory_tree(output_dir, max_depth=3, max_children=5))

# %% [markdown]
# The training run folder is the main audit trail for this experiment:
#
# - `args.yaml`: the resolved training configuration, including data, model, device,
#   hyperparameters, and output paths.
# - `results.csv`: per-epoch losses and validation metrics for analysis or
#   custom plots.
# - `results.png`: a quick visual summary of the loss and metric curves.
# - `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, and
#   `BoxPR_curve.png`: precision, recall, F1, and precision-recall behavior
#   across confidence thresholds.
# - `confusion_matrix*.png`: class-level error summaries, shown as raw counts
#   and normalized proportions.
# - `labels.jpg`: dataset label and bounding-box distribution checks.
# - `train_batch*.jpg`: examples of augmented training batches, useful for
#   spotting annotation or transform issues.
# - `val_batch*_labels.jpg` and `val_batch*_pred.jpg`: validation labels next
#   to model predictions for qualitative inspection.
# - `weights/best.pt`: the checkpoint with the best validation score; use this
#   for inference or export.
# - `weights/last.pt`: the final checkpoint from the last epoch; use this when
#   resuming training.

# %% [markdown]
# ### What precision, recall, and mAP mean
#
# Object-detection metrics start by matching predicted boxes to labeled
# ground-truth boxes. A prediction is usually counted as a **true positive** only
# when it has the right class and overlaps the ground-truth box enough. That
# overlap is measured with IoU, or intersection over union:
#
# $$
# \mathrm{IoU} =
# \frac{
#   \operatorname{area}(B_\mathrm{pred} \cap B_\mathrm{true})
# }{
#   \operatorname{area}(B_\mathrm{pred} \cup B_\mathrm{true})
# }
# $$
#
# Once predictions are matched to labels:
#
# - **Precision** answers: when the model predicts an object, how often is that
#   prediction correct?
#
#   $$
#   \mathrm{precision} =
#   \frac{\text{true positives}}
#        {\text{true positives} + \text{false positives}}
#   $$
#
#   Low precision means the model is producing too many wrong or duplicate
#   detections.
#
# - **Recall** answers: of the objects that are actually present, how many did
#   the model find?
#
#   $$
#   \mathrm{recall} =
#   \frac{\text{true positives}}
#        {\text{true positives} + \text{false negatives}}
#   $$
#
#   Low recall means the model is missing many real objects.
#
# Precision and recall usually trade off as the confidence threshold changes.
# A high threshold accepts only confident boxes, which can improve precision but
# miss objects. A low threshold accepts more boxes, which can improve recall but
# may add false positives.
#
# **AP**, or average precision, summarizes the precision-recall curve for one
# class. **mAP**, or mean average precision, averages AP across classes.
#
# - **mAP50** is mAP when a predicted box only needs IoU of at least 0.50 to
#   match a ground-truth box. This is the more forgiving headline metric.
# - **mAP50-95** averages mAP across IoU thresholds from 0.50 to 0.95 in steps
#   of 0.05. This is stricter because it rewards both correct classification and
#   tight localization.
#
# In this notebook, zero precision, zero recall, zero mAP50, and zero mAP50-95
# mean the evaluator accepted no correct detections. Since the validation label
# images clearly contain objects, all-zero metrics should be treated as a warning
# sign to inspect the evaluation setup before trusting the run as a measure of
# model quality.

# %% [markdown]
# ### Interpret the smoke-test results
#
# The run artifacts show two different stories:
#
# 1. **The training process itself probably ran successfully.** Ultralytics wrote
#    the full run directory, `results.csv`, diagnostic plots, and both
#    `weights/best.pt` and `weights/last.pt`. The training and validation losses
#    also moved downward over the five recorded epochs.
# 2. **The validation/evaluation result is suspicious and should not be trusted
#    as a model-quality estimate.** Precision, recall, mAP50, and mAP50-95 are
#    exactly zero for every epoch. The validation label images contain many
#    objects, but the matching prediction images contain no boxes, and the
#    confusion matrix pushes true classes into the background row.
#
# So this smoke test is useful as an execution check: the local environment can
# train, validate, plot, and save checkpoints. It is not yet useful as evidence
# that the fine-tuned model learned a good detector.

# %%
metrics_path = output_dir / "results.csv"
weights_dir = output_dir / "weights"

with metrics_path.open(newline="") as metrics_file:
    metric_rows = list(csv.DictReader(metrics_file))

numeric_rows = [
    {key.strip(): float(value) for key, value in row.items()}
    for row in metric_rows
]
first_epoch = numeric_rows[0]
last_epoch = numeric_rows[-1]

print(f"Metrics file: {relative_to_project_root(metrics_path)}")
print(f"Epochs recorded: {len(numeric_rows)}")
print(f"Final epoch: {last_epoch['epoch']:.0f}")
print(f"best.pt exists: {(weights_dir / 'best.pt').exists()}")
print(f"last.pt exists: {(weights_dir / 'last.pt').exists()}")
print()

loss_columns = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]

print("Loss movement from epoch 1 to epoch 5:")
for column in loss_columns:
    print(f"- {column}: {first_epoch[column]:.4f} -> {last_epoch[column]:.4f}")

metric_columns = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

print()
print("Validation detection metrics:")
for column in metric_columns:
    values = {row[column] for row in numeric_rows}
    unique_values = ", ".join(f"{value:.4f}" for value in sorted(values))
    print(f"- {column}: final={last_epoch[column]:.4f}; all={unique_values}")

all_detection_metrics_zero = all(
    row[column] == 0.0
    for row in numeric_rows
    for column in metric_columns
)
print()
print("All validation detection metrics are zero:", all_detection_metrics_zero)

# %% [markdown]
# ### Demonstrate the decisive artifacts
#
# The most important visual checks are:
#
# - `results.png`: losses decrease, but all validation metrics stay flat at zero.
# - `val_batch0_labels.jpg`: the validation images do have ground-truth objects.
# - `val_batch0_pred.jpg`: the corresponding prediction images are blank.
# - `confusion_matrix_normalized.png`: true classes are counted as background,
#   which is consistent with no accepted detections.

# %%
important_artifacts = [
    ("Training curves", "results.png", 900),
    ("Validation labels", "val_batch0_labels.jpg", 650),
    ("Validation predictions", "val_batch0_pred.jpg", 650),
    ("Normalized confusion matrix", "confusion_matrix_normalized.png", 800),
]

for title, filename, width in important_artifacts:
    artifact_path = output_dir / filename
    print(f"{title}: {relative_to_project_root(artifact_path)}")
    display(IPyImage(filename=str(artifact_path), width=width))

# %% [markdown]
# ### Working interpretation
#
# Treat this as a successful **training pipeline smoke test**, not as a successful
# detector evaluation. The run completed on the intended local setup and produced
# the expected audit trail, so the mechanics are healthy enough to keep iterating.
#
# The model-quality signal is the part that looks wrong. A YOLO11n checkpoint
# fine-tuned from COCO weights on COCO128 should not normally produce zero
# precision, zero recall, and zero mAP across every epoch while the label batches
# clearly contain common COCO objects. The most likely immediate suspect is the
# evaluation configuration, especially the explicit `conf=0.25` setting used
# during training/validation, because mAP evaluation is usually inspected with a
# very low confidence threshold before integrating over the precision-recall
# curve. Re-run validation with a lower/default confidence threshold before
# interpreting these numbers as model performance.

# %% [markdown]
# ### Trouble-shooting
#
# The training losses decreased, indicating that the training loop executed. However,
# precision, recall, mAP50, and mAP50-95 remained at zero. The validation label plots
# also appear to show no ground-truth boxes, while prediction plots contain detected
# objects. This suggests a possible validation label loading or dataset-cache issue,
# rather than simply poor model learning.
#
# Before proceeding to a custom COCO-derived dataset, the next step is to validate the
# pretrained YOLO11n model directly on `coco128.yaml` and confirm that validation
# labels are loaded correctly.

# %%
baseline_model = YOLO(YOLO11N_MODEL)

baseline_metrics = baseline_model.val(
    data="coco128.yaml",
    device=DEVICE,
    project=str(RUNS_DIR),
    name="nb02_pretrained_coco128_val",
    exist_ok=True,
    plots=True,
)

# %%
print("mAP50:", baseline_metrics.box.map50)
print("mAP50-95:", baseline_metrics.box.map)
print("precision:", baseline_metrics.box.mp)
print("recall:", baseline_metrics.box.mr)

# %% [markdown]
# The output concludes that,
#
# - COCO128 is installed correctly.
# - coco128.yaml is being resolved correctly.
# - validation labels are being loaded correctly.
# - The pretrained YOLO11n checkpoint is valid.
#
# Possible issues:
#
# - very small dataset
# - tiny batch size
# - aggressive augmentation
# - short training
# - learning-rate/warmup behaviour
