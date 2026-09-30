"""Inverted index over chunk text, in the same SQLite database as the chunks.

The index is keyed by chunk id and analyzer, not by corpus: chunks are immutable and shared by
every corpus version that contains them, so each chunk is indexed once and a corpus version is
searched by joining through its membership. Statistics (chunk count, average length, document
frequency) are computed over that membership only, so scores for a corpus version never depend
on any other corpus or version. That is why this is not SQLite FTS5, whose bm25() uses
table-wide statistics.

Only terms and counts are stored; chunk text stays in `chunks`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from rag_forge.domain.models import Corpus
from rag_forge.retrieval.analysis import ANALYZER, analyze
from rag_forge.retrieval.bm25 import Posting
from rag_forge.storage.sqlite import SqliteStore

# Chunk ids in one corpus version under the corpus's chunking config.
_MEMBERS = """
SELECT c.id AS chunk_id FROM corpus_version_members mb
JOIN chunks c ON c.document_version_id = mb.document_version_id AND c.chunking_hash = :chash
WHERE mb.corpus_id = :corpus AND mb.version = :version
"""


@dataclass(frozen=True)
class VersionStats:
    chunks: int
    avg_length: float


class SqliteLexicalIndex:
    analyzer = ANALYZER

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    def _scope(self, corpus: Corpus, version: int) -> dict[str, object]:
        return {
            "chash": corpus.chunking.config_hash(),
            "corpus": corpus.id,
            "version": version,
            "analyzer": self.analyzer,
        }

    def ensure_indexed(self, corpus: Corpus, version: int) -> int:
        """Index any chunk of this corpus version not yet indexed. Returns how many were added.

        Idempotent and deterministic: a chunk's postings depend only on its text and the analyzer.
        """
        with self.store.transaction() as db:
            missing = db.execute(
                f"""SELECT c.id, json_extract(c.data, '$.text') FROM ({_MEMBERS}) m
                    JOIN chunks c ON c.id = m.chunk_id
                    LEFT JOIN lexical_docs d ON d.chunk_id = c.id AND d.analyzer = :analyzer
                    WHERE d.doc_key IS NULL ORDER BY c.id""",
                self._scope(corpus, version),
            ).fetchall()
            added = 0
            for chunk_id, text in missing:
                counts = Counter(analyze(text))
                cur = db.execute(
                    "INSERT OR IGNORE INTO lexical_docs (analyzer, chunk_id, length)"
                    " VALUES (?, ?, ?)",
                    (self.analyzer, chunk_id, sum(counts.values())),
                )
                if cur.rowcount == 0:  # indexed concurrently
                    continue
                db.executemany(
                    "INSERT OR IGNORE INTO lexical_terms (term) VALUES (?)", [(t,) for t in counts]
                )
                ids = dict(
                    db.execute(
                        "SELECT term, id FROM lexical_terms"
                        " WHERE term IN (SELECT value FROM json_each(?))",
                        (json.dumps(list(counts)),),
                    ).fetchall()
                )
                db.executemany(
                    "INSERT INTO lexical_postings (term_id, doc_key, tf) VALUES (?, ?, ?)",
                    [(ids[t], cur.lastrowid, tf) for t, tf in counts.items()],
                )
                added += 1
        return added

    def stats(self, corpus: Corpus, version: int) -> VersionStats:
        with self.store.transaction() as db:
            n, avg = db.execute(
                f"""SELECT COUNT(*), COALESCE(AVG(d.length), 0) FROM ({_MEMBERS}) m
                    JOIN lexical_docs d ON d.chunk_id = m.chunk_id AND d.analyzer = :analyzer""",
                self._scope(corpus, version),
            ).fetchone()
        return VersionStats(chunks=n, avg_length=avg)

    def postings(self, corpus: Corpus, version: int, terms: list[str]) -> list[Posting]:
        """Postings for `terms`, restricted to chunks in this corpus version."""
        with self.store.transaction() as db:
            rows = db.execute(
                f"""SELECT d.chunk_id, t.term, p.tf, d.length
                    FROM lexical_terms t
                    JOIN lexical_postings p ON p.term_id = t.id
                    JOIN lexical_docs d ON d.doc_key = p.doc_key AND d.analyzer = :analyzer
                    JOIN ({_MEMBERS}) m ON m.chunk_id = d.chunk_id
                    WHERE t.term IN (SELECT value FROM json_each(:terms))""",
                {**self._scope(corpus, version), "terms": json.dumps(terms)},
            ).fetchall()
        return [Posting(*r) for r in rows]
