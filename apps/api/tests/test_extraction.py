from pathlib import Path

import pytest

from rag_forge.ingestion.extraction import ExtractionError, extract
from rag_forge.storage.blobs import BlobStore, sha256_hex
from tests.conftest import make_pdf


def test_sha256_is_stable_and_content_addressed(tmp_path: Path) -> None:
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    blobs = BlobStore(tmp_path)
    d1, d2 = blobs.put(b"same"), blobs.put(b"same")
    assert d1 == d2 == sha256_hex(b"same")
    assert blobs.get(d1) == b"same"
    assert len(list(tmp_path.rglob("*"))) == 2  # one shard dir, one blob


def test_plain_text_normalises_newlines_and_bom() -> None:
    extractor, ex = extract("a.txt", "﻿line one\r\nline two\rthree".encode())
    assert extractor.media_type == "text/plain"
    assert ex.text == "line one\nline two\nthree"


def test_markdown_title_and_headings() -> None:
    _, ex = extract("n.md", b"intro\n# Title Here\n## Sub\ntext\n")
    assert ex.metadata == {"title": "Title Here", "headings": 2}
    assert "# Title Here" in ex.text  # source preserved


def test_pdf_pages_offsets_and_blank_page_warning() -> None:
    extractor, ex = extract("p.pdf", make_pdf(["First page text", "", "Third page"]))
    assert extractor.media_type == "application/pdf"
    assert ex.metadata["pages"] == 3
    assert ex.text.startswith("First page text")
    assert ex.page_offsets is not None and len(ex.page_offsets) == 3
    assert ex.text[ex.page_offsets[2] :].startswith("Third page")
    assert ex.warnings == ["1 of 3 pages had no extractable text"]


@pytest.mark.parametrize(
    ("name", "data", "reason"),
    [
        ("a.docx", b"PK\x03\x04", "unsupported file type '.docx'"),
        ("noext", b"hello", "unsupported file type"),
        ("a.txt", b"\xff\xfe\x00bad", "binary content"),
        ("a.txt", b"caf\xe9", "not valid UTF-8"),
        ("a.md", b"  \n\t\n", "no extractable text"),
        ("a.pdf", b"hello, not a pdf", "missing %PDF- header"),
        ("a.pdf", b"%PDF-1.4\ngarbage without structure", "corrupt or unreadable PDF"),
        ("a.pdf", make_pdf(["", ""]), "no extractable text (scanned PDF?"),
    ],
)
def test_invalid_files_are_refused_with_a_reason(name: str, data: bytes, reason: str) -> None:
    with pytest.raises(ExtractionError, match=reason.replace("(", r"\(").replace("?", r"\?")):
        extract(name, data)
