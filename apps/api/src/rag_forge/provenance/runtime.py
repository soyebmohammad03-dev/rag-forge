"""Capture what a run executed on beyond the Python environment: toolchain versions, lockfile
hashes, settings (secrets redacted) and the exact model files in the Hugging Face cache.

Model file hashes cost nothing for weights: the Hugging Face cache stores LFS files under their
sha256, so the blob name is the content hash. Small non-LFS files are hashed directly. Nothing
is downloaded here; a file that is not cached is reported as such.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

from rag_forge.domain.models import ModelFile, ModelRecord, RuntimeSnapshot

REPO_ROOT = Path(__file__).resolve().parents[5]
LOCKFILES = ("apps/api/uv.lock", "package-lock.json")
RUNTIME_PACKAGES = (
    "fastapi",
    "pydantic",
    "uvicorn",
    "starlette",
    "numpy",
    "onnxruntime",
    "tokenizers",
    "huggingface-hub",
    "pypdf",
)
SETTING_PREFIXES = ("RAG_FORGE_", "HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_CACHE")
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SMALL_FILE = 64 * 1024 * 1024  # hash non-LFS files up to this size


def _run(*cmd: str) -> str | None:
    try:
        out = subprocess.run(
            list(cmd), cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@cache
def _toolchain() -> tuple[str | None, str | None]:
    uv = _run("uv", "--version")
    return _run("node", "--version"), uv.split()[1] if uv and len(uv.split()) > 1 else uv


def _package(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def settings() -> dict[str, str]:
    """Settings that change behaviour, with secret values replaced. Never returns a secret."""
    return {
        k: ("<redacted>" if SECRET.search(k) else v)
        for k, v in sorted(os.environ.items())
        if k.startswith(SETTING_PREFIXES)
    }


def model_files(repo: str, revision: str | None, files: list[str]) -> list[ModelFile]:
    out = []
    for name in files:
        cached = try_to_load_from_cache(repo, name, revision=revision) if revision else None
        if not isinstance(cached, str):
            out.append(ModelFile(path=name, sha256=None, bytes=None, source="not-cached"))
            continue
        blob = Path(cached).resolve()
        size = blob.stat().st_size
        if SHA256.match(blob.name):
            out.append(ModelFile(path=name, sha256=blob.name, bytes=size, source="lfs-blob-id"))
        elif size <= SMALL_FILE:
            out.append(ModelFile(path=name, sha256=_sha256(blob), bytes=size, source="computed"))
        else:
            out.append(ModelFile(path=name, sha256=None, bytes=size, source="not-hashed"))
    return out


def capture_runtime(models: list[ModelRecord]) -> RuntimeSnapshot:
    node, uv = _toolchain()
    status = _run("git", "status", "--porcelain", "--untracked-files=no")
    in_repo = _run("git", "rev-parse", "--is-inside-work-tree") == "true"
    return RuntimeSnapshot(
        git_commit=_run("git", "rev-parse", "HEAD") if in_repo else None,
        node_version=node,
        uv_version=uv,
        git_dirty=bool(status) if in_repo else None,
        lockfiles={p: _sha256(REPO_ROOT / p) for p in LOCKFILES if (REPO_ROOT / p).is_file()},
        packages={p: _package(p) for p in RUNTIME_PACKAGES},
        settings=settings(),
        models=models,
    )
