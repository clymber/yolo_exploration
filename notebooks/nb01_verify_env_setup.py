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

# %%
"""
Verify the YOLO development environment setup.
"""

# Reloads all modules every time before executing code, except explicitly excluded using
# # %aimport -<package>, like %aimport -numpy.
# %load_ext autoreload
# %autoreload 2

# %aimport -os
# %aimport -sys
# %aimport -platform
# %aimport -torch
# %aimport -torchvision
# %aimport -torchaudio
# %aimport -ultralytics


# %%
from yolo_exploration import configure_stdio_relative_path, PROJECT_ROOT

# Display project paths relatively for consistent output across environments.
configure_stdio_relative_path(PROJECT_ROOT)

# %%
import os
import platform
import sys

import torch
import torchaudio
import torchvision
import ultralytics

from yolo_exploration import (
    cache_download,
    ensure_dir,
)

# %% [markdown]
# # YOLO development environment setup verification

# %%
WEIGHTS_DIR = ensure_dir(PROJECT_ROOT / "models" / "pretrained")
DATA_EXTERNAL = ensure_dir(PROJECT_ROOT / "data" / "external")
PREDICTIONS_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "predictions")

MODEL_PATH = WEIGHTS_DIR / "yolo11n.pt"
BUS_IMAGE = DATA_EXTERNAL / "bus.jpg"

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Machine:", platform.machine())
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Project root:", PROJECT_ROOT)
print("Weights directory:", WEIGHTS_DIR)
print("External data directory:", DATA_EXTERNAL)
print("Predictions directory:", PREDICTIONS_DIR)


# %% [markdown]
# ## 1. Verify PyTorch MPS (Metal Performance Shaders)

# %%
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("PYTORCH_ENABLE_MPS_FALLBACK =", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))

if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Selected device:", DEVICE)

    x = torch.ones(1, device=DEVICE)
    print("Example tensor:", x)
else:
    DEVICE = "cpu"
    print("WARNING: MPS is not available. Falling back to CPU.", file=sys.stderr)

# %% [markdown]
# `torch.backends.mps.is_available()` returned `True`, and this indicates that Apple M1
# Pro GPU is available for ML training.

# %% [markdown]
# ## 2. Verify Ultralytics YOLO

# %%
# !yolo checks

# %% [markdown]
# GPU in Ultralytics checks is basically CUDA-style GPU reporting. Current Apple M1 Pro is
# not an NVIDIA/CUDA GPU, so the `yolo checks` output is normal:
#
# ```plaintext
# GPU                    None
# GPU count              None
# ```

# %% [markdown]
# ## 3. Minimal Python test

# %%
BUS_IMAGE = cache_download(BUS_IMAGE, "https://ultralytics.com/images/bus.jpg")

print("Bus image:", BUS_IMAGE)
print("Bus image exists:", BUS_IMAGE.exists())

# %%
from ultralytics import YOLO

model_path = WEIGHTS_DIR / "yolo11n.pt"
model = YOLO(model_path)

results = model.predict(
    source=BUS_IMAGE,
    device=DEVICE,
    project=PREDICTIONS_DIR,
    name="nb01_bus_smoke_test",
    exist_ok=True,
    show=False
)

print("Prediction completed.")
print("Number of result objects:", len(results))

# %% [markdown]
# The warning shown above indicates that operator `torchvision::nms` was doing Non-Maximum Suppression with CPU instead of GPU. This is because current `torchvision::nms` does not support MPS yet, and the conda virtual environment set up enables MPS fallback:
#
# ```yml
# variables:
#   # Let PyTorch run unsupported MPS ops, such as torchvision NMS, on CPU.
#   PYTORCH_ENABLE_MPS_FALLBACK: "1"
# ```
#
# Enabling MPS fallback means: Use MPS/GPU whenever the operation is supported.
# If one specific operation is not supported on MPS, run only that operation on CPU instead of crashing.

# %%
result = results[0]

print("Input image:", result.path)
print("Original image shape:", result.orig_shape)

if result.boxes is not None:
    print("Number of detected boxes:", len(result.boxes))
    print("Class IDs:", result.boxes.cls.tolist())
    print("Confidences:", result.boxes.conf.tolist())
else:
    print("No boxes detected.")
