"""Turn uploaded bytes into text, or refuse loudly.

Each format is an `Extractor`. Adding DOCX, HTML, OCR or table-aware parsers means adding one
class to `EXTRACTORS`; nothing else changes. Extractors never guess: anything they cannot read
faithfully raises `ExtractionError`, and the file is rejected with that message.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Protocol

import pypdf


class ExtractionError(Exception):
    """The file cannot be ingested. The message is shown to the user."""


@dataclass(frozen=True)
class Extraction:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_offsets: list[int] | None = None  # char offset where each page starts
    warnings: list[str] = field(default_factory=list)


class Extractor(Protocol):
    name: str
    version: str
    media_type: str
    extensions: tuple[str, ...]

    def extract(self, data: bytes) -> Extraction: ...


def _decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise ExtractionError("binary content (NUL bytes) in a text file")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"not valid UTF-8 text (invalid byte at offset {exc.start})") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


class PlainTextExtractor:
    name = "plaintext"
    version = "1"
    media_type = "text/plain"
    extensions: tuple[str, ...] = (".txt", ".text")

    def extract(self, data: bytes) -> Extraction:
        text = _decode_text(data)
        return Extraction(text, {"lines": text.count("\n") + 1})


_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


class MarkdownExtractor:
    """Keeps Markdown source as-is (headings and emphasis carry signal for retrieval)."""

    name = "markdown"
    version = "1"
    media_type = "text/markdown"
    extensions: tuple[str, ...] = (".md", ".markdown")

    def extract(self, data: bytes) -> Extraction:
        text = _decode_text(data)
        headings = _ATX_HEADING.findall(text)
        title = next((t for level, t in headings if level == "#"), None)
        return Extraction(text, {"title": title, "headings": len(headings)})


class PdfExtractor:
    name = "pypdf"
    version = pypdf.__version__
    media_type = "application/pdf"
    extensions: tuple[str, ...] = (".pdf",)

    def extract(self, data: bytes) -> Extraction:
        if not data.startswith(b"%PDF-"):
            raise ExtractionError("not a PDF (missing %PDF- header)")
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ExtractionError("encrypted PDFs are not supported")
            pages = list(reader.pages)
            title = reader.metadata.title if reader.metadata else None
        except ExtractionError:
            raise
        except Exception as exc:  # pypdf raises many types for malformed input
            raise ExtractionError(f"corrupt or unreadable PDF: {exc}") from exc
        if not pages:
            raise ExtractionError("PDF has no pages")

        parts: list[str] = []
        offsets: list[int] = []
        warnings: list[str] = []
        pos = 0
        for i, page in enumerate(pages, 1):
            try:
                page_text = (page.extract_text() or "").strip()
            except Exception as exc:
                warnings.append(f"page {i}: text extraction failed ({exc})")
                page_text = ""
            offsets.append(pos)
            parts.append(page_text)
            pos += len(page_text) + 2  # "\n\n" joiner
        empty = sum(1 for p in parts if not p)
        if empty and empty < len(parts):
            warnings.append(f"{empty} of {len(parts)} pages had no extractable text")
        return Extraction(
            "\n\n".join(parts),
            {"pages": len(pages), "title": title, "empty_pages": empty},
            page_offsets=offsets,
            warnings=warnings,
        )


EXTRACTORS: tuple[Extractor, ...] = (PlainTextExtractor(), MarkdownExtractor(), PdfExtractor())


def extractor_for(filename: str) -> Extractor:
    ext = PurePath(filename).suffix.lower()
    for extractor in EXTRACTORS:
        if ext in extractor.extensions:
            return extractor
    supported = ", ".join(e for x in EXTRACTORS for e in x.extensions)
    raise ExtractionError(f"unsupported file type '{ext or filename}' (supported: {supported})")


def extract(filename: str, data: bytes) -> tuple[Extractor, Extraction]:
    extractor = extractor_for(filename)
    result = extractor.extract(data)
    if not result.text.strip():
        hint = (
            " (scanned PDF? OCR is not supported yet)"
            if extractor.media_type.endswith("pdf")
            else ""
        )
        raise ExtractionError(f"no extractable text{hint}")
    return extractor, result
