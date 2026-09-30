from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_forge.domain.models import EmbedderSpec
from rag_forge.ingestion.service import IngestionService
from rag_forge.main import create_app
from rag_forge.retrieval.embedding import OnnxSentenceEmbedder
from rag_forge.storage.blobs import BlobStore
from rag_forge.storage.sqlite import SqliteStore


def make_pdf(pages: list[str]) -> bytes:
    """A minimal, valid PDF with one line of Helvetica text per page ("" = blank page)."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for pid, text in zip(page_ids, pages, strict=True):
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET".encode() if text else b""
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offsets:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture(scope="session")
def embedder() -> OnnxSentenceEmbedder:
    """The real pinned model, loaded once per session (downloaded once into the HF cache)."""
    return OnnxSentenceEmbedder(EmbedderSpec())


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "db.sqlite3")


@pytest.fixture
def service(store: SqliteStore, tmp_path: Path) -> IngestionService:
    return IngestionService(store, BlobStore(tmp_path / "blobs"), max_bytes=10_000)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(tmp_path)) as c:
        yield c
