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

from yolo_exploration import PROJECT_ROOT, configure_stdio_relative_path

# Display project paths relatively for consistent output across environments.
# Should be called before other imports.
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
    aligned_print,
    cache_download,
    ensure_dir,
)
from yolo_exploration.utils import image as image_utils

# %% [markdown]
# # YOLO development environment setup verification

# %%
WEIGHTS_DIR = ensure_dir(PROJECT_ROOT / "models" / "pretrained")
DATA_EXTERNAL = ensure_dir(PROJECT_ROOT / "data" / "external")
PREDICTIONS_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "predictions")

MODEL_PATH = WEIGHTS_DIR / "yolo11n.pt"
BUS_IMAGE = DATA_EXTERNAL / "bus.jpg"

aligned_print({
    "Python": sys.version,
    "Platform": platform.platform(),
    "Machine": platform.machine(),
    "PyTorch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "Ultralytics": ultralytics.__version__,
})
aligned_print({
    "Project root": PROJECT_ROOT,
    "Weights directory": WEIGHTS_DIR,
    "External data directory": DATA_EXTERNAL,
    "Predictions directory": PREDICTIONS_DIR,
})


# %% [markdown]
# ## 1. Verify PyTorch MPS (Metal Performance Shaders)

# %%
aligned_print({
    "Model path": MODEL_PATH,
    "Bus image path": BUS_IMAGE,
    "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
})

if torch.backends.mps.is_available():
    DEVICE = "mps"
    x = torch.ones(1, device=DEVICE)

    aligned_print({
        "Selected device": DEVICE,
        "Example tensor": x,
    })
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

aligned_print({
    "Bus image": BUS_IMAGE,
    "Bus image exists": BUS_IMAGE.exists(),
})

if BUS_IMAGE.exists():
    image_utils.display(BUS_IMAGE, width=300)

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

print("Prediction completed. Number of result objects:", len(results))

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
messages = {
    "Input image": result.path,
    "Original image shape": result.orig_shape,
}

if result.boxes is not None:
    messages.update({
        "Number of detected boxes": len(result.boxes),
        "Class IDs": result.boxes.cls.tolist(),
        "Confidences": result.boxes.conf.tolist(),
    })
else:
    messages.update({"Number of detected boxes": 0})

aligned_print(messages)

# %%
for result in results:
    annotated_img = result.plot(
        pil=True,
        labels=True,
        conf=True,
        boxes=True,
    )
    image_utils.display(annotated_img, width=300)
