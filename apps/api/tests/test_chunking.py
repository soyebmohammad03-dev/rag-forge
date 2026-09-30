from itertools import pairwise

import pytest
from pydantic import ValidationError

from rag_forge.domain.models import ChunkingConfig, ChunkingStrategy
from rag_forge.ingestion.chunking import chunk_spans, chunk_text

PARAGRAPHS = "\n\n".join(
    f"Paragraph {p}. " + " ".join(f"Sentence {p}.{s} has several ordinary words." for s in range(6))
    for p in range(8)
)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
@pytest.mark.parametrize(("size", "overlap"), [(200, 0), (300, 60), (1000, 150), (80, 20)])
def test_invariants(strategy: ChunkingStrategy, size: int, overlap: int) -> None:
    cfg = ChunkingConfig(strategy=strategy, chunk_size=size, chunk_overlap=overlap)
    chunks = chunk_text(PARAGRAPHS, cfg, "dv_x")
    covered = [False] * len(PARAGRAPHS)
    for i, c in enumerate(chunks):
        assert PARAGRAPHS[c.char_start : c.char_end] == c.text  # exact offsets
        assert 0 < len(c.text) <= size
        assert c.text == c.text.strip()
        assert c.ordinal == i
        if i:
            assert c.char_start > chunks[i - 1].char_start  # always makes progress
        for j in range(c.char_start, c.char_end):
            covered[j] = True
    missing = [
        ch for ch, ok in zip(PARAGRAPHS, covered, strict=True) if not ok and not ch.isspace()
    ]
    assert missing == []  # every non-whitespace character lands in some chunk


def test_recursive_prefers_paragraph_and_sentence_boundaries() -> None:
    spans = chunk_spans(PARAGRAPHS, ChunkingConfig(chunk_size=400, chunk_overlap=0))
    for _, end in spans:
        assert PARAGRAPHS[end - 1] == "."  # never cut mid-sentence when sentences fit


def test_overlap_repeats_trailing_text() -> None:
    chunks = chunk_text(PARAGRAPHS, ChunkingConfig(chunk_size=300, chunk_overlap=100), "dv")
    overlaps = [prev.char_end - cur.char_start for prev, cur in pairwise(chunks)]
    assert all(0 < o <= 100 for o in overlaps)
    for _, cur in pairwise(chunks):  # overlap starts on a word, not mid-word
        assert PARAGRAPHS[cur.char_start - 1].isspace()


def test_hard_cut_when_no_separator() -> None:
    spans = chunk_spans("x" * 250, ChunkingConfig(chunk_size=100, chunk_overlap=0))
    assert spans == [(0, 100), (100, 200), (200, 250)]


def test_ids_are_deterministic_and_config_sensitive() -> None:
    a = chunk_text(PARAGRAPHS, ChunkingConfig(), "dv_1")
    b = chunk_text(PARAGRAPHS, ChunkingConfig(), "dv_1")
    c = chunk_text(PARAGRAPHS, ChunkingConfig(chunk_size=999), "dv_1")
    assert [x.id for x in a] == [x.id for x in b]
    assert a[0].id != c[0].id
    assert a[0].chunking_hash == ChunkingConfig().config_hash()


def test_page_metadata_from_offsets() -> None:
    text = "page one text\n\npage two text"
    chunks = chunk_text(text, ChunkingConfig(chunk_size=50, chunk_overlap=0), "dv", [0, 15])
    assert chunks[0].metadata == {"words": 6, "page_start": 1, "page_end": 2}


def test_config_validation() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=10)


def test_empty_text() -> None:
    assert chunk_text("   \n\n  ", ChunkingConfig(), "dv") == []


def test_chunks_do_not_end_on_a_markdown_heading() -> None:
    md = (
        "# Title\n\n"
        + "Intro sentence here. " * 10
        + "\n\n## Next section\n\n"
        + "Body text. " * 30
    )
    chunks = chunk_text(md, ChunkingConfig(chunk_size=260, chunk_overlap=0), "dv")
    for c in chunks:
        assert not c.text.rstrip().split("\n")[-1].startswith("#"), c.text
    assert any(c.text.startswith("## Next section") for c in chunks)


def test_config_hash_includes_algorithm_version(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_forge.domain import models

    before = ChunkingConfig().config_hash()
    monkeypatch.setattr(models, "CHUNKER_VERSION", models.CHUNKER_VERSION + 1)
    assert ChunkingConfig().config_hash() != before
