"""
Runtime configuration for YOLO experiments.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch

from .paths import PROJECT_ROOT

ULTRALYTICS_PRIVACY_SETTINGS = {
    "sync": False,
    "hub": False,
    "clearml": False,
    "comet": False,
    "dvc": False,
    "mlflow": False,
    "neptune": False,
    "raytune": False,
    "tensorboard": False,
    "wandb": False,
    "vscode_msg": False,
    "openvino_msg": False,
}


def get_device() -> str:
    """
    Select the best available device for local YOLO experiments.

    Priority:
    1. Apple Silicon MPS
    2. CPU fallback
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_ultralytics_privacy(
    *,
    offline: bool = True,
    config_dir: Path | str | None = PROJECT_ROOT / "outputs" / "ultralytics",
    settings_overrides: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    """
    Disable Ultralytics telemetry and optional experiment integrations.

    Call this before importing ``YOLO`` so Ultralytics initializes with the
    project privacy settings from the start.
    """
    if offline:
        os.environ.setdefault("YOLO_OFFLINE", "true")

    if config_dir is not None:
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_path))

    from ultralytics import settings

    updates = dict(ULTRALYTICS_PRIVACY_SETTINGS)
    if settings_overrides is not None:
        updates.update(settings_overrides)

    settings.update(updates)
    return updates


DEVICE = get_device()
