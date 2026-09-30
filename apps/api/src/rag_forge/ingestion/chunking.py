"""Chunking strategies. Every chunk records exact character offsets into the extracted text,
so `text[chunk.char_start:chunk.char_end] == chunk.text` always holds.
"""

from __future__ import annotations

import bisect
import hashlib

from rag_forge.domain.models import Chunk, ChunkingConfig, ChunkingStrategy

Span = tuple[int, int]

# Coarse to fine. A chunk ends at the coarsest boundary that still fills half the window.
SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def _ends_on_heading(text: str, start: int, end: int) -> bool:
    """True if the chunk's last line is a Markdown heading, which belongs with what follows."""
    body = text[start:end].rstrip()
    return body[body.rfind("\n") + 1 :].startswith("#")


def _snap_end(text: str, start: int, size: int, prev_end: int) -> int:
    limit = start + size
    if limit >= len(text):
        return len(text)
    min_end = max(start + size // 2, prev_end + 1)
    for sep in SEPARATORS:
        j = text.rfind(sep, start, limit)
        while j != -1 and j + len(sep) >= min_end:
            if not _ends_on_heading(text, start, j + len(sep)):
                return j + len(sep)
            j = text.rfind(sep, start, j)
    return limit  # no usable boundary: hard cut


def _snap_start(text: str, lo: int, end: int) -> int:
    """First boundary in [lo, end) so the overlap starts on a sentence/word, not mid-word."""
    while end > lo and text[end - 1].isspace():  # the chunk's own trailing break is not overlap
        end -= 1
    for sep in SEPARATORS:
        k = text.find(sep, max(lo - len(sep), 0), end)
        if k != -1 and lo <= k + len(sep) < end:
            return k + len(sep)
    return end


def _recursive(text: str, size: int, overlap: int) -> list[Span]:
    spans: list[Span] = []
    start, prev_end = 0, 0
    while start < len(text):
        end = _snap_end(text, start, size, prev_end)
        spans.append((start, end))
        if end >= len(text):
            break
        start = _snap_start(text, max(end - overlap, start + 1), end) if overlap else end
        prev_end = end
    return spans


def _strip(text: str, span: Span) -> Span | None:
    a, b = span
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return (a, b) if a < b else None


def chunk_spans(text: str, config: ChunkingConfig) -> list[Span]:
    size, overlap = config.chunk_size, config.chunk_overlap
    if config.strategy is ChunkingStrategy.FIXED:
        step = size - overlap
        raw = [(p, min(p + size, len(text))) for p in range(0, len(text), step)]
        # drop a trailing window that lies entirely inside the previous one
        raw = [s for i, s in enumerate(raw) if i == 0 or s[1] > raw[i - 1][1]]
    else:
        raw = _recursive(text, size, overlap)
    return [s for span in raw if (s := _strip(text, span))]


def chunk_id(document_version_id: str, chunking_hash: str, ordinal: int) -> str:
    """Deterministic: re-chunking the same version with the same config yields the same ids."""
    digest = hashlib.sha256(f"{document_version_id}|{chunking_hash}|{ordinal}".encode())
    return f"chk_{digest.hexdigest()[:20]}"


def chunk_text(
    text: str,
    config: ChunkingConfig,
    document_version_id: str,
    page_offsets: list[int] | None = None,
) -> list[Chunk]:
    chash = config.config_hash()
    chunks = []
    for i, (a, b) in enumerate(chunk_spans(text, config)):
        body = text[a:b]
        meta: dict[str, int] = {"words": len(body.split())}
        if page_offsets:
            meta["page_start"] = bisect.bisect_right(page_offsets, a)
            meta["page_end"] = bisect.bisect_right(page_offsets, b - 1)
        chunks.append(
            Chunk(
                id=chunk_id(document_version_id, chash, i),
                document_version_id=document_version_id,
                chunking_hash=chash,
                ordinal=i,
                text=body,
                char_start=a,
                char_end=b,
                metadata=meta,
            )
        )
    return chunks
