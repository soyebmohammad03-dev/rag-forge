"""SQLite implementation of the metadata store.

Each table keeps the full model as JSON in `data` plus the columns needed for lookups, so the
schema stays small while models evolve. History tables are append-only, enforced by triggers:
document versions, extracted text, chunks, corpus versions, membership and ingestion records can
never be updated or deleted.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rag_forge.domain.models import (
    Chunk,
    Corpus,
    CorpusVersion,
    Document,
    DocumentVersion,
    FileOutcome,
    IngestionRecord,
)
from rag_forge.storage.base import ConcurrentModificationError, CorpusStats, VersionChange

_APPEND_ONLY = (
    "document_versions",
    "document_texts",
    "chunks",
    "corpus_versions",
    "corpus_version_members",
    "ingestions",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL REFERENCES corpora(id),
  filename TEXT NOT NULL, data TEXT NOT NULL, UNIQUE (corpus_id, filename));
CREATE TABLE IF NOT EXISTS document_versions (
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
  version INTEGER NOT NULL, content_sha256 TEXT NOT NULL, data TEXT NOT NULL,
  UNIQUE (document_id, version));
CREATE TABLE IF NOT EXISTS document_texts (
  document_version_id TEXT PRIMARY KEY REFERENCES document_versions(id), text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, document_version_id TEXT NOT NULL REFERENCES document_versions(id),
  chunking_hash TEXT NOT NULL, ordinal INTEGER NOT NULL, data TEXT NOT NULL,
  UNIQUE (document_version_id, chunking_hash, ordinal));
CREATE TABLE IF NOT EXISTS corpus_versions (
  corpus_id TEXT NOT NULL REFERENCES corpora(id), version INTEGER NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY (corpus_id, version));
CREATE TABLE IF NOT EXISTS corpus_version_members (
  corpus_id TEXT NOT NULL, version INTEGER NOT NULL,
  document_id TEXT NOT NULL REFERENCES documents(id),
  document_version_id TEXT NOT NULL REFERENCES document_versions(id),
  PRIMARY KEY (corpus_id, version, document_id),
  FOREIGN KEY (corpus_id, version) REFERENCES corpus_versions(corpus_id, version));
CREATE TABLE IF NOT EXISTS ingestions (
  id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL REFERENCES corpora(id),
  started_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created_at TEXT, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, experiment_id TEXT, data TEXT NOT NULL);
""" + "".join(
    f"""
CREATE TRIGGER IF NOT EXISTS {t}_no_update BEFORE UPDATE ON {t}
  BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
CREATE TRIGGER IF NOT EXISTS {t}_no_delete BEFORE DELETE ON {t}
  BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;"""
    for t in _APPEND_ONLY
)


# Forward-only migrations: MIGRATIONS[n] upgrades a database from schema n - 1 to n.
# Never edit a released migration; add a new one.
MIGRATIONS: dict[int, str] = {
    1: SCHEMA,
    2: """
-- Lexical retrieval index (derived from chunks; rebuildable, never the source of truth).
-- doc_key is an INTEGER PRIMARY KEY so it survives VACUUM, unlike implicit rowids.
CREATE TABLE lexical_terms (id INTEGER PRIMARY KEY, term TEXT NOT NULL UNIQUE);
CREATE TABLE lexical_docs (
  doc_key INTEGER PRIMARY KEY, analyzer TEXT NOT NULL,
  chunk_id TEXT NOT NULL REFERENCES chunks(id), length INTEGER NOT NULL,
  UNIQUE (analyzer, chunk_id));
CREATE TABLE lexical_postings (
  term_id INTEGER NOT NULL REFERENCES lexical_terms(id),
  doc_key INTEGER NOT NULL REFERENCES lexical_docs(doc_key), tf INTEGER NOT NULL,
  PRIMARY KEY (term_id, doc_key)) WITHOUT ROWID;
""",
    3: """
-- Dense retrieval. Vectors are keyed by embedder config and chunk, so each immutable chunk is
-- embedded once per model config and shared by every corpus version that contains it.
CREATE TABLE dense_vectors (
  embedder_hash TEXT NOT NULL, chunk_id TEXT NOT NULL REFERENCES chunks(id),
  vector BLOB NOT NULL, PRIMARY KEY (embedder_hash, chunk_id)) WITHOUT ROWID;
-- One row per build of (corpus version, embedder); status moves building -> ready | failed.
CREATE TABLE dense_indexes (
  id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL REFERENCES corpora(id),
  version INTEGER NOT NULL, embedder_hash TEXT NOT NULL, status TEXT NOT NULL,
  started_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX dense_indexes_lookup ON dense_indexes (corpus_id, embedder_hash, version);
CREATE TRIGGER dense_vectors_no_update BEFORE UPDATE ON dense_vectors
  BEGIN SELECT RAISE(ABORT, 'dense_vectors is append-only'); END;
""",
    4: """
-- Arena. Dataset versions, experiments, per-case results and artifacts are immutable; a run row
-- is updated only while it progresses (queued -> running -> completed | partial | failed).
-- The Phase 0 experiments/runs placeholder tables are left untouched and unused.
CREATE TABLE benchmark_datasets (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL, created_at TEXT NOT NULL,
  data TEXT NOT NULL, UNIQUE (name, version));
CREATE TABLE arena_experiments (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE arena_runs (
  id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES arena_experiments(id),
  created_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE arena_run_cases (
  run_id TEXT NOT NULL REFERENCES arena_runs(id), arm TEXT NOT NULL, case_id TEXT NOT NULL,
  data TEXT NOT NULL, PRIMARY KEY (run_id, arm, case_id));
CREATE TABLE arena_artifacts (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES arena_runs(id), data TEXT NOT NULL);
"""
    + "".join(
        f"""
CREATE TRIGGER {t}_no_update BEFORE UPDATE ON {t}
  BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
CREATE TRIGGER {t}_no_delete BEFORE DELETE ON {t}
  BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;"""
        for t in ("benchmark_datasets", "arena_experiments", "arena_run_cases", "arena_artifacts")
    ),
}
SCHEMA_VERSION = max(MIGRATIONS)


class SqliteStore:
    kind = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute("PRAGMA journal_mode=WAL")
            found = db.execute("PRAGMA user_version").fetchone()[0]
            if found > SCHEMA_VERSION:
                raise RuntimeError(f"{path}: schema v{found} is newer than v{SCHEMA_VERSION}")
        for version in range(found + 1, SCHEMA_VERSION + 1):
            with self.transaction() as db:
                # One explicit transaction per step: a failed migration leaves the prior schema.
                db.executescript(
                    f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version={version};\nCOMMIT;"
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        # ponytail: one connection per operation; pool connections if profiling says so.
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    # --- corpora ---------------------------------------------------------------

    def add_corpus(self, corpus: Corpus) -> Corpus:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO corpora (id, name, data) VALUES (?, ?, ?)",
                (corpus.id, corpus.name, corpus.model_dump_json()),
            )
            db.execute(
                "INSERT INTO corpus_versions (corpus_id, version, data) VALUES (?, 0, ?)",
                (
                    corpus.id,
                    CorpusVersion(
                        corpus_id=corpus.id,
                        version=0,
                        ingestion_id=None,
                        document_count=0,
                        created_at=corpus.created_at,
                    ).model_dump_json(),
                ),
            )
        return corpus

    def list_corpora(self) -> list[Corpus]:
        with self.transaction() as db:
            rows = db.execute("SELECT data FROM corpora ORDER BY rowid DESC").fetchall()
        return [Corpus.model_validate_json(r[0]) for r in rows]

    def get_corpus(self, corpus_id: str) -> Corpus | None:
        with self.transaction() as db:
            row = db.execute("SELECT data FROM corpora WHERE id = ?", (corpus_id,)).fetchone()
        return Corpus.model_validate_json(row[0]) if row else None

    def corpus_name_taken(self, name: str) -> bool:
        with self.transaction() as db:
            return (
                db.execute("SELECT 1 FROM corpora WHERE name = ?", (name,)).fetchone() is not None
            )

    def corpus_stats(self, corpus: Corpus) -> CorpusStats:
        with self.transaction() as db:
            docs, size, chars = db.execute(
                """SELECT COUNT(*), COALESCE(SUM(json_extract(dv.data, '$.byte_size')), 0),
                          COALESCE(SUM(json_extract(dv.data, '$.text_chars')), 0)
                   FROM corpus_version_members m
                   JOIN document_versions dv ON dv.id = m.document_version_id
                   WHERE m.corpus_id = ? AND m.version = ?""",
                (corpus.id, corpus.version),
            ).fetchone()
            (chunks,) = db.execute(
                """SELECT COUNT(*) FROM corpus_version_members m
                   JOIN chunks c ON c.document_version_id = m.document_version_id
                   WHERE m.corpus_id = ? AND m.version = ? AND c.chunking_hash = ?""",
                (corpus.id, corpus.version, corpus.chunking.config_hash()),
            ).fetchone()
            last = db.execute(
                "SELECT data FROM ingestions WHERE corpus_id = ? ORDER BY started_at DESC LIMIT 1",
                (corpus.id,),
            ).fetchone()
        return CorpusStats(
            document_count=docs,
            chunk_count=chunks,
            total_bytes=size,
            total_chars=chars,
            last_ingestion=IngestionRecord.model_validate_json(last[0]) if last else None,
        )

    # --- documents -------------------------------------------------------------

    def members(self, corpus_id: str, version: int) -> dict[str, DocumentVersion]:
        """document_id -> the document version contained in that corpus version."""
        with self.transaction() as db:
            rows = db.execute(
                """SELECT m.document_id, dv.data FROM corpus_version_members m
                   JOIN document_versions dv ON dv.id = m.document_version_id
                   WHERE m.corpus_id = ? AND m.version = ?
                   ORDER BY json_extract(dv.data, '$.filename')""",
                (corpus_id, version),
            ).fetchall()
        return {r[0]: DocumentVersion.model_validate_json(r[1]) for r in rows}

    def documents_by_filename(self, corpus_id: str) -> dict[str, Document]:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT data FROM documents WHERE corpus_id = ?", (corpus_id,)
            ).fetchall()
        docs = [Document.model_validate_json(r[0]) for r in rows]
        return {d.filename: d for d in docs}

    def get_document(self, document_id: str) -> Document | None:
        with self.transaction() as db:
            row = db.execute("SELECT data FROM documents WHERE id = ?", (document_id,)).fetchone()
        return Document.model_validate_json(row[0]) if row else None

    def document_versions(self, document_id: str) -> list[DocumentVersion]:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT data FROM document_versions WHERE document_id = ? ORDER BY version",
                (document_id,),
            ).fetchall()
        return [DocumentVersion.model_validate_json(r[0]) for r in rows]

    def get_document_version(self, version_id: str) -> DocumentVersion | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT data FROM document_versions WHERE id = ?", (version_id,)
            ).fetchone()
        return DocumentVersion.model_validate_json(row[0]) if row else None

    def document_text(self, version_id: str) -> str | None:
        with self.transaction() as db:
            row = db.execute(
                "SELECT text FROM document_texts WHERE document_version_id = ?", (version_id,)
            ).fetchone()
        return row[0] if row else None

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT data FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
                (json.dumps(chunk_ids),),
            ).fetchall()
        return {c.id: c for c in (Chunk.model_validate_json(r[0]) for r in rows)}

    def chunks(
        self, version_id: str, chunking_hash: str, offset: int = 0, limit: int = 100
    ) -> tuple[list[Chunk], int]:
        with self.transaction() as db:
            where = "document_version_id = ? AND chunking_hash = ?"
            (total,) = db.execute(
                f"SELECT COUNT(*) FROM chunks WHERE {where}", (version_id, chunking_hash)
            ).fetchone()
            rows = db.execute(
                f"SELECT data FROM chunks WHERE {where} ORDER BY ordinal LIMIT ? OFFSET ?",
                (version_id, chunking_hash, limit, offset),
            ).fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows], total

    # --- versions & ingestion ------------------------------------------------------

    def commit_version(
        self,
        corpus: Corpus,
        version: CorpusVersion,
        members: dict[str, str],
        new_documents: list[Document],
        new_versions: list[tuple[DocumentVersion, str, list[Chunk]]],
        record: IngestionRecord,
    ) -> Corpus:
        """Atomically write a new corpus version. `members` maps document_id -> version_id."""
        updated = corpus.model_copy(update={"version": version.version})
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO documents (id, corpus_id, filename, data) VALUES (?, ?, ?, ?)",
                [(d.id, d.corpus_id, d.filename, d.model_dump_json()) for d in new_documents],
            )
            for dv, text, chunks in new_versions:
                db.execute(
                    "INSERT INTO document_versions (id, document_id, version, content_sha256, data)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (dv.id, dv.document_id, dv.version, dv.content_sha256, dv.model_dump_json()),
                )
                db.execute("INSERT INTO document_texts VALUES (?, ?)", (dv.id, text))
                db.executemany(
                    "INSERT INTO chunks (id, document_version_id, chunking_hash, ordinal, data)"
                    " VALUES (?, ?, ?, ?, ?)",
                    [
                        (c.id, dv.id, c.chunking_hash, c.ordinal, c.model_dump_json())
                        for c in chunks
                    ],
                )
            db.execute(
                "INSERT INTO corpus_versions (corpus_id, version, data) VALUES (?, ?, ?)",
                (corpus.id, version.version, version.model_dump_json()),
            )
            # ponytail: full membership copied per version; switch to delta rows if corpora
            # reach ~10^5 documents x many versions.
            db.executemany(
                "INSERT INTO corpus_version_members VALUES (?, ?, ?, ?)",
                [(corpus.id, version.version, doc, dv) for doc, dv in members.items()],
            )
            self._insert_ingestion(db, record)
            bumped = db.execute(
                "UPDATE corpora SET data = ? WHERE id = ? AND json_extract(data, '$.version') = ?",
                (updated.model_dump_json(), corpus.id, corpus.version),
            )
            if bumped.rowcount != 1:  # someone committed a version since we read the corpus
                raise ConcurrentModificationError(corpus.id)
        return updated

    def add_ingestion(self, record: IngestionRecord) -> None:
        with self.transaction() as db:
            self._insert_ingestion(db, record)

    @staticmethod
    def _insert_ingestion(db: sqlite3.Connection, record: IngestionRecord) -> None:
        db.execute(
            "INSERT INTO ingestions (id, corpus_id, started_at, data) VALUES (?, ?, ?, ?)",
            (record.id, record.corpus_id, record.started_at.isoformat(), record.model_dump_json()),
        )

    def list_ingestions(self, corpus_id: str) -> list[IngestionRecord]:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT data FROM ingestions WHERE corpus_id = ? ORDER BY started_at DESC",
                (corpus_id,),
            ).fetchall()
        return [IngestionRecord.model_validate_json(r[0]) for r in rows]

    def list_versions(self, corpus_id: str) -> list[CorpusVersion]:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT data FROM corpus_versions WHERE corpus_id = ? ORDER BY version DESC",
                (corpus_id,),
            ).fetchall()
        return [CorpusVersion.model_validate_json(r[0]) for r in rows]

    def version_changes(self, corpus_id: str, version: int) -> list[VersionChange]:
        now, before = self.members(corpus_id, version), self.members(corpus_id, version - 1)
        changes = []
        for doc_id in sorted(now.keys() | before.keys()):
            cur, prev = now.get(doc_id), before.get(doc_id)
            if cur is None:
                change = FileOutcome.REMOVED
            elif prev is None:
                change = FileOutcome.ADDED
            elif prev.id != cur.id:
                change = FileOutcome.MODIFIED
            else:
                change = FileOutcome.UNCHANGED
            changes.append(
                VersionChange(
                    document_id=doc_id,
                    filename=(cur or prev).filename,  # type: ignore[union-attr]
                    change=change,
                    document_version_id=cur.id if cur else None,
                    previous_document_version_id=prev.id if prev else None,
                )
            )
        return changes
