# %%
"""
Application runtime paths helper and config.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = (start or Path.cwd()).resolve()

    for path in [current, *current.parents]:
        if (path / "pyproject.toml").is_file():
            return path
        project_child = path / "yolo_exploration"
        if (project_child / "pyproject.toml").is_file():
            return project_child

    raise RuntimeError("Project root not found")


def relative_to_project_root(
    path: Path | str, *, project_root: Path | str | None = None
) -> str:
    """
    Return a stable POSIX-style path relative to the project root.
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = candidate.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is not inside project root: {resolved}") from exc


# Basic runtime directory configuration
PROJECT_ROOT = find_project_root()
