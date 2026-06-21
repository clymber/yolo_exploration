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
# # Train YOLO11 model for object detection on custom dataset
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

# Reloads all modules every time before executing code, except explicitly excluded using
# # %aimport -<package>, like %aimport -numpy.
# %load_ext autoreload
# %autoreload 2
# %aimport -csv
# %aimport -textwrap
# %aimport -functools
# %aimport -IPython
# %aimport -ultralytics

from yolo_exploration import configure_stdio_relative_path, PROJECT_ROOT

# Display project paths relatively for consistent output across environments.
# Must be called before other imports to setup filters.
configure_stdio_relative_path(PROJECT_ROOT)


# %%
import csv
import textwrap
from functools import partial

from IPython.display import Image as IPyImage
from IPython.display import display

from yolo_exploration import (
    cache_download,
    configure_ultralytics_privacy,
    directory_tree,
    ensure_dir,
    get_device,
    relative_to_userhome,
)

# Must be called before importing ultralytics
configure_ultralytics_privacy()  
from ultralytics import YOLO  # noqa: E402

# %%
DATA_DIR = ensure_dir(PROJECT_ROOT / "data")
MODELS_DIR = ensure_dir(PROJECT_ROOT / "models")
PRETRAINED_DIR = ensure_dir(MODELS_DIR / "pretrained")
OUTPUTS_DIR = ensure_dir(PROJECT_ROOT / "outputs")
RUNS_DIR = ensure_dir(OUTPUTS_DIR / "runs")
DATA_EXTERNAL = ensure_dir(DATA_DIR / "external")
PREDICTIONS_DIR = ensure_dir(OUTPUTS_DIR / "predictions")

YOLO11N_MODEL = PRETRAINED_DIR / "yolo11n.pt"
DEVICE = get_device()

print("PROJECT_ROOT:", relative_to_userhome(PROJECT_ROOT))
print("RUNS_DIR:", RUNS_DIR)
print("YOLO11N_MODEL:", YOLO11N_MODEL)

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

print("Dog image:", DOG_IMAGE)
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

print("Prediction completed. Number of result objects:", len(pred_results))

# %% [markdown]
# ### Inspect YOLO prediction output
#
# The `model.predict()` method returns a list of
# `ultralytics.yolo.engine.results.Results` objects, which contain the prediction
# results and metadata for each input image.

# %%
results = pred_results[0]

print("Image path:", results.path)
print("Original image shape:", results.orig_shape)

# Display the predicted image with bounding boxes overlaid.
pred_image_path = str(PREDICTIONS_DIR / "nb02_pretrained_dog_prediction.jpg")
pred_image_path = results.save(pred_image_path)
print("Predicted image with bounding boxes:", pred_image_path)
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
# Now that pretrained inference works, we train YOLO11n on a small COCO128
# dataset (made from the first 128 images of COCO train2017) as a local training
# smoke test.
#
# ### Fine-tune the current YOLO11n model on a small object-detection dataset
#
# The `model.train(...)` starts training from those pretrained weights, not from
# scratch.
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
COCO128_SMOKE_RUN_NAME = "nb02_coco128_yolo11n_smoke_test"

smoke_model = YOLO(YOLO11N_MODEL)
results = smoke_model.train(
    data="coco128.yaml",
    epochs=5,
    imgsz=640,
    batch=2,
    device=DEVICE,
    project=str(RUNS_DIR),
    name=COCO128_SMOKE_RUN_NAME,
    exist_ok=True,
    plots=True,
)

# %% [markdown]
# ### Inspect the training output

# %%
output_dir = RUNS_DIR / COCO128_SMOKE_RUN_NAME
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
# If precision, recall, mAP50, and mAP50-95 are all zero, the evaluator accepted
# no correct detections. Since the validation label images should contain
# objects, all-zero metrics are a warning sign to inspect the evaluation setup
# before trusting the run as a measure of model quality.

# %% [markdown]
# ### Interpret the smoke-test results
#
# The current smoke test is healthy. Ultralytics wrote the expected run
# directory, `results.csv`, diagnostic plots, and both `weights/best.pt` and
# `weights/last.pt`. The training losses move downward over the five recorded
# epochs, and the validation detection metrics are nonzero throughout the run.
#
# This is the expected result for a YOLO11n checkpoint fine-tuned from COCO
# weights on COCO128. It confirms that the local environment can train,
# validate, plot diagnostics, and save checkpoints before we move on to a custom
# dataset.

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

print(f"Metrics file: {metrics_path}")
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
# - `results.png`: training losses trend downward and validation metrics remain
#   nonzero.
# - `val_batch0_labels.jpg`: the validation images contain ground-truth boxes.
# - `val_batch0_pred.jpg`: the corresponding predictions contain detected boxes.
# - `confusion_matrix_normalized.png`: class-level matches and errors are visible
#   instead of a background-only failure pattern.

# %%
important_artifacts = [
    ("Training curves", "results.png", 900),
    ("Validation labels", "val_batch0_labels.jpg", 650),
    ("Validation predictions", "val_batch0_pred.jpg", 650),
    ("Normalized confusion matrix", "confusion_matrix_normalized.png", 800),
]

for title, filename, width in important_artifacts:
    artifact_path = output_dir / filename
    print(f"{title}: {artifact_path}")
    display(IPyImage(filename=str(artifact_path), width=width))

# %% [markdown]
# ### Working interpretation
#
# Treat this as a successful **training pipeline smoke test**. The run completed
# on the intended local setup and produced a plausible detector-quality signal:
# mAP50 ends near 0.67, mAP50-95 ends near 0.49, and the validation prediction
# images show boxes.
#
# The earlier all-zero run should now be treated as superseded. Its artifacts
# were overwritten by this corrected run because the notebook uses
# `exist_ok=True`. For future investigations, use a unique run name when
# preserving failed artifacts matters.

# %% [markdown]
# ### Trouble-shooting
#
# The next checks keep the troubleshooting evidence explicit. They compare the
# current smoke-test checkpoint with the original pretrained checkpoint at two
# confidence thresholds, then validate the pretrained model directly on
# `coco128.yaml`.

# %%
def print_box_metrics(title, metrics):
    """Print the high-level box metrics from an Ultralytics validation result."""
    print(title)
    print(f"- precision: {metrics.box.mp:.4f}")
    print(f"- recall: {metrics.box.mr:.4f}")
    print(f"- mAP50: {metrics.box.map50:.4f}")
    print(f"- mAP50-95: {metrics.box.map:.4f}")


threshold_checks = [
    (
        "Smoke checkpoint at conf=0.001",
        weights_dir / "best.pt",
        0.001,
        "nb02_smoke_best_val_conf_0p001",
    ),
    (
        "Smoke checkpoint at conf=0.25",
        weights_dir / "best.pt",
        0.25,
        "nb02_smoke_best_val_conf_0p25",
    ),
    (
        "Pretrained checkpoint at conf=0.25",
        YOLO11N_MODEL,
        0.25,
        "nb02_pretrained_val_conf_0p25",
    ),
]

for title, checkpoint_path, conf_threshold, run_name in threshold_checks:
    checkpoint_metrics = YOLO(checkpoint_path).val(
        data="coco128.yaml",
        device=DEVICE,
        conf=conf_threshold,
        iou=0.7,
        max_det=100,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    print_box_metrics(title, checkpoint_metrics)
    print()

# %% [markdown]
# The current smoke checkpoint validates normally at both thresholds. The
# `conf=0.001` result is closer to mAP-style evaluation because it lets the
# precision-recall curve include low-confidence detections. The `conf=0.25`
# result is stricter and therefore reports lower mAP, but it is still clearly
# nonzero. The pretrained checkpoint provides a baseline on the same labels.

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
print_box_metrics("Pretrained COCO128 baseline", baseline_metrics)

# %% [markdown]
# The pretrained validation output confirms that,
#
# - COCO128 is installed correctly.
# - coco128.yaml is being resolved correctly.
# - validation labels are being loaded correctly.
# - The pretrained YOLO11n checkpoint is valid.
#
# Its metrics are in the same range as the smoke-test checkpoint, so this
# section is a dataset and checkpoint sanity check rather than an error
# diagnosis.

# %% [markdown]
# ### CPU control fine-tune
#
# Before tuning a custom dataset, run one conservative CPU epoch. This is slower
# per optimizer step than MPS, but it gives a useful control path: labels,
# optimizer setup, checkpoint saving, and validation should all produce plausible
# metrics from the same pretrained weights.

# %%
CPU_CONTROL_RUN_NAME = "nb02_coco128_yolo11n_cpu_one_epoch_control"

cpu_control_model = YOLO(YOLO11N_MODEL)
cpu_control_results = cpu_control_model.train(
    data="coco128.yaml",
    epochs=1,
    imgsz=640,
    batch=16,
    device="cpu",
    workers=0,
    project=str(RUNS_DIR),
    name=CPU_CONTROL_RUN_NAME,
    exist_ok=True,
    plots=True,
    amp=False,
)

# %%
cpu_control_checkpoint = RUNS_DIR / CPU_CONTROL_RUN_NAME / "weights" / "best.pt"
cpu_control_metrics = YOLO(cpu_control_checkpoint).val(
    data="coco128.yaml",
    device="cpu",
    conf=0.25,
    iou=0.7,
    max_det=100,
    project=str(RUNS_DIR),
    name=f"{CPU_CONTROL_RUN_NAME}_conf_0p25",
    exist_ok=True,
    plots=False,
    verbose=False,
)

print_box_metrics("CPU one-epoch control at conf=0.25", cpu_control_metrics)

# %% [markdown]
# The CPU control is healthy too. Its one-epoch metrics are close to the MPS
# smoke test and pretrained baseline, which gives confidence that the dataset and
# training code path are sound. Its runtime is not directly comparable to the MPS
# run because it uses one epoch with `batch=16`, while the MPS smoke test uses
# five epochs with `batch=2`.
