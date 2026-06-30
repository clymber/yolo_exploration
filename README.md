# YOLO Exploration

YOLO object-detection experiments and Apple Silicon MPS environment checks.

The Conda environment is defined in `environment.yml`. Python package metadata,
editable-install configuration, optional dependency groups, and developer tool
settings are defined in `pyproject.toml`.

## Environment Requirements

This project is currently intended for a local Conda-based workflow, especially
for Apple Silicon MPS environment checks and local Jupytext notebook
development.

Google Colab is not a convenient target at this stage. The repo keeps notebooks
as Jupytext `.py` files, ignores generated `.ipynb` files, and relies on the
`yolo-dev` Conda environment. Colab support would need a separate setup path.

## Environment Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate it and load the project runtime environment variables:

```bash
source env_setup.sh
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

## Jupytext Notebook Workflow

This repo stores notebooks as Jupytext `.py` files using the percent format,
for example `notebooks/nb01_verify_env_setup.py`. This keeps notebooks easier
to review in Git while still allowing them to be opened and run in Jupyter.

After creating and activating the Conda environment, register the Jupyter kernel
once:

```bash
python -m ipykernel install --user --name yolo-dev \
  --display-name "Python (yolo-dev)"
```

Sync the Jupytext notebooks to Jupyter `.ipynb` notebooks with Make:

```bash
make sync-notebooks
```

The Makefile runs Jupytext in the `yolo-dev` Conda environment and uses
`jupytext.toml`, which pairs each notebook as `ipynb,py:percent`. After
syncing, each notebook has both a Git-friendly `.py` file and a Jupyter
`.ipynb` file. The generated `.ipynb` files are ignored by Git, so commit
changes to the `.py` notebook files.

Start JupyterLab from the activated environment:

```bash
jupyter lab
```

Open the `.ipynb` files from the `notebooks/` directory and select the
`Python (yolo-dev)` kernel if Jupyter does not select it automatically. When
you save a paired notebook in Jupyter, Jupytext updates the `.py` file too.

If you edit the `.py` notebook directly, run the sync command again before
opening it in Jupyter:

```bash
make sync-notebooks
```

To execute all synced notebooks from the command line:

```bash
make run-notebooks
```

## Verify the Setup

After syncing Jupytext notebooks, run the environment verification notebook:

```bash
jupyter lab notebooks/nb01_verify_env_setup.ipynb
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
