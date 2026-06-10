# YOLO Exploration

YOLO object-detection experiments and Apple Silicon MPS environment checks.

The Conda environment is defined in `environment.yml`. Python package metadata,
editable-install configuration, optional dependency groups, and developer tool
settings are defined in `pyproject.toml`.

## Environment Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate yolo-dev
```

Install this project as an editable local package:

```bash
python -m pip install --no-deps -e .
```

The `--no-deps` flag is intentional for the Conda workflow. `environment.yml`
already installs the main runtime packages, including PyTorch, torchvision,
torchaudio, Jupyter, Ultralytics, and common scientific Python tools. Installing
the local package in editable mode makes imports such as this work from notebooks
and scripts:

```python
from yolo_exploration import PROJECT_ROOT
```

## Verify the Setup

Run the environment verification notebook:

```bash
jupyter lab notebooks/nb01_veryfy_env_setup.py
```

The environment enables:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

This lets PyTorch use Apple Silicon MPS where supported and fall back to CPU for
unsupported operations, such as some torchvision NMS paths.

Optional dependency groups are also declared in `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[notebook]"
python -m pip install -e ".[data,vision]"
python -m pip install -e ".[yolo]"
```

For the recommended Conda workflow, prefer `environment.yml` for heavy packages
such as PyTorch and OpenCV. The extras are most useful for pip-only environments
or quick tool installation.

## Development Commands

Run Ruff:

```bash
python -m ruff check .
```

Format with Black:

```bash
python -m black src notebooks
```

Check formatting:

```bash
python -m black --check src notebooks
```

Run tests:

```bash
python -m pytest
```

If there is no `tests/` directory yet, pytest will not have project tests to run.

## Updating the Environment

After editing `environment.yml`, update the Conda environment:

```bash
conda env update -f environment.yml --prune
```

After editing package metadata or dependencies in `pyproject.toml`, refresh the
editable install:

```bash
python -m pip install --no-deps -e .
```

For a pip-managed environment, install the desired extras explicitly:

```bash
python -m pip install -e ".[dev,notebook,data,vision]"
```
