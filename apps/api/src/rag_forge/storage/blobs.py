"""Content-addressed store for original uploaded bytes. Identical content is stored once."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def put(self, data: bytes) -> str:
        digest = sha256_hex(data)
        target = self.path(digest)
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            # write-then-rename so a crash never leaves a truncated blob under a valid hash
            fd, tmp = tempfile.mkstemp(dir=target.parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, target)
        return digest

    def get(self, digest: str) -> bytes:
        return self.path(digest).read_bytes()
