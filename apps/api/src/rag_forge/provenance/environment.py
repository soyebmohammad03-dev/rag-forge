"""Capture the execution environment so every run can be reconstructed."""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rag_forge import __version__
from rag_forge.domain.models import EnvironmentSnapshot

TRACKED_PACKAGES = ("fastapi", "pydantic", "uvicorn")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


@cache  # the environment does not change within a process
def capture_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        rag_forge_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_commit=_git_commit(),
        packages={name: _package_version(name) for name in TRACKED_PACKAGES},
    )
