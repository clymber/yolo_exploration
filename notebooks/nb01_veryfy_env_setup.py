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
# # YOLO development environment setup verification

# %%
"""
The notebook is used to verify the YOLO development enviroment setup.
"""
import os
import sys
from urllib.request import urlretrieve

import torch

from yolo_exploration import PROJECT_ROOT

WEIGHTS_DIR = PROJECT_ROOT / "models" / "pretrained"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 1. Verify PyTorch MPS (Metal Performance Shaders)

# %%
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    print("MPS is available")
    x = torch.ones(1, device="mps")
    print("Example: ", x)
else:
    print("ERROR: MPS is not available", file=sys.stderr)

# %% [markdown]
# `torch.backends.mps.is_available()` returned `True`, and this indicates that Apple M1
# Pro GPU is avaliable for ML training.

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
bus_img = DATA_EXTERNAL / "bus.jpg"
if not bus_img.exists():
    souce = "https://ultralytics.com/images/bus.jpg"
    urlretrieve(souce, bus_img)

# %%
from ultralytics import YOLO

model_path = WEIGHTS_DIR / "yolo11n.pt"
model = YOLO(model_path)

results = model.predict(
    source=bus_img,
    device="mps",
    project=PREDICTIONS_DIR,
    show=False
)

# %% [markdown]
# The warning shown above indicates that operator `torchvision:mns` was doing Non-Maximum Suppression with CPU instead of GPU. This is because current `torchvision::mns` does not support MPS yet, and the conda virtual environment set up enables MPS fallback:
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
print("PYTORCH_ENABLE_MPS_FALLBACK =", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
print("PyTorch:", torch.__version__)
print("MPS available: ", torch.backends.mps.is_available())
