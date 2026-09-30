"""SQLite-backed vector storage with exact cosine search.

Vectors live in `dense_vectors` keyed by (embedder hash, chunk id); index builds live in
`dense_indexes`. A search loads the matrix for exactly one ready index (one corpus version,
one embedder config) and scores it with a NumPy dot product. Swapping in FAISS or pgvector
means reimplementing `search` (and storage) behind the same methods.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from rag_forge.domain.models import Corpus, DenseIndex, DenseIndexStatus, utcnow
from rag_forge.storage.sqlite import SqliteStore

Matrix = NDArray[np.float32]

_MEMBERS = """
SELECT c.id AS chunk_id, json_extract(c.data, '$.text') AS text FROM corpus_version_members mb
JOIN chunks c ON c.document_version_id = mb.document_version_id AND c.chunking_hash = :chash
WHERE mb.corpus_id = :corpus AND mb.version = :version
"""


class IndexIntegrityError(RuntimeError):
    """Stored vectors do not match their index (missing rows or wrong dimension)."""


class SqliteVectorIndex:
    def __init__(self, store: SqliteStore, cache_size: int = 8) -> None:
        self.store = store
        # Ready indexes are immutable, so caching their matrices by id is always safe.
        # ponytail: whole matrix in memory per index; ANN/on-disk index past ~10^6 chunks.
        self._matrix = lru_cache(maxsize=cache_size)(self._load_matrix)

    # --- index records ------------------------------------------------------------

    def save(self, index: DenseIndex) -> None:
        with self.store.transaction() as db:
            db.execute(
                """INSERT INTO dense_indexes
                     (id, corpus_id, version, embedder_hash, status, started_at, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (id) DO UPDATE SET status = excluded.status, data = excluded.data""",
                (
                    index.id,
                    index.corpus_id,
                    index.corpus_version,
                    index.embedder.config_hash,
                    index.status,
                    index.started_at.isoformat(),
                    index.model_dump_json(),
                ),
            )

    def latest(self, corpus_id: str, version: int, embedder_hash: str) -> DenseIndex | None:
        with self.store.transaction() as db:
            row = db.execute(
                """SELECT data FROM dense_indexes WHERE corpus_id = ? AND version = ?
                   AND embedder_hash = ? ORDER BY started_at DESC LIMIT 1""",
                (corpus_id, version, embedder_hash),
            ).fetchone()
        return DenseIndex.model_validate_json(row[0]) if row else None

    def has_other_ready(self, corpus_id: str, version: int, embedder_hash: str) -> bool:
        """A ready index for this corpus under another version or embedder config exists."""
        with self.store.transaction() as db:
            return (
                db.execute(
                    """SELECT 1 FROM dense_indexes WHERE corpus_id = ? AND status = 'ready'
                       AND NOT (version = ? AND embedder_hash = ?) LIMIT 1""",
                    (corpus_id, version, embedder_hash),
                ).fetchone()
                is not None
            )

    def fail_interrupted(self) -> int:
        """Builds left 'building' by a previous process can never finish; mark them failed."""
        with self.store.transaction() as db:
            rows = db.execute("SELECT data FROM dense_indexes WHERE status = 'building'").fetchall()
        for (data,) in rows:
            index = DenseIndex.model_validate_json(data)
            self.save(
                index.model_copy(
                    update={
                        "status": DenseIndexStatus.FAILED,
                        "error": "interrupted: the API stopped during the build",
                        "finished_at": utcnow(),
                    }
                )
            )
        return len(rows)

    # --- vectors --------------------------------------------------------------------

    def member_chunks(self, corpus: Corpus, version: int) -> list[tuple[str, str]]:
        """(chunk id, text) for a corpus version, ordered by chunk id."""
        with self.store.transaction() as db:
            rows = db.execute(
                f"{_MEMBERS} ORDER BY c.id",
                {"chash": corpus.chunking.config_hash(), "corpus": corpus.id, "version": version},
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def existing(self, embedder_hash: str, chunk_ids: list[str]) -> set[str]:
        with self.store.transaction() as db:
            rows = db.execute(
                """SELECT chunk_id FROM dense_vectors WHERE embedder_hash = ?
                   AND chunk_id IN (SELECT value FROM json_each(?))""",
                (embedder_hash, json.dumps(chunk_ids)),
            ).fetchall()
        return {r[0] for r in rows}

    def put(self, embedder_hash: str, chunk_ids: list[str], vectors: Matrix) -> None:
        rows = [
            (embedder_hash, cid, vec.astype("<f4").tobytes())
            for cid, vec in zip(chunk_ids, vectors, strict=True)
        ]
        with self.store.transaction() as db:
            db.executemany(
                "INSERT OR IGNORE INTO dense_vectors (embedder_hash, chunk_id, vector)"
                " VALUES (?, ?, ?)",
                rows,
            )

    def _load_matrix(self, index: DenseIndex) -> tuple[tuple[str, ...], Matrix]:
        with self.store.transaction() as db:
            rows = db.execute(
                """SELECT c.id, v.vector FROM corpus_version_members mb
                   JOIN chunks c ON c.document_version_id = mb.document_version_id
                        AND c.chunking_hash = :chash
                   LEFT JOIN dense_vectors v ON v.chunk_id = c.id AND v.embedder_hash = :ehash
                   WHERE mb.corpus_id = :corpus AND mb.version = :version ORDER BY c.id""",
                {
                    "chash": index.chunking_hash,
                    "ehash": index.embedder.config_hash,
                    "corpus": index.corpus_id,
                    "version": index.corpus_version,
                },
            ).fetchall()
        dim = index.embedder.dimension
        missing = [cid for cid, blob in rows if blob is None]
        if missing:
            raise IndexIntegrityError(f"{len(missing)} chunks have no vector (e.g. {missing[0]})")
        bad = [cid for cid, blob in rows if len(blob) != dim * 4]
        if bad:
            raise IndexIntegrityError(
                f"vector for {bad[0]} is not {dim}-dimensional; the index is incompatible"
            )
        ids = tuple(r[0] for r in rows)
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype="<f4").reshape(len(rows), dim)
        return ids, matrix

    def content_hash(self, index: DenseIndex) -> str:
        """Identity of what an index searches: its chunk ids and their exact vector bytes."""
        ids, matrix = self._load_matrix(index)
        h = hashlib.sha256(index.embedder.config_hash.encode())
        h.update("\n".join(ids).encode())
        h.update(matrix.tobytes())
        return h.hexdigest()

    # --- search ---------------------------------------------------------------------

    def search(self, index: DenseIndex, query: Matrix, top_k: int) -> list[tuple[str, float]]:
        """Exact cosine similarity (vectors are unit-length). Best first, ties by chunk id."""
        if query.shape != (index.embedder.dimension,):
            raise IndexIntegrityError(
                f"query has shape {query.shape}, index expects ({index.embedder.dimension},)"
            )
        ids, matrix = self._matrix(index)
        # float64 accumulation, then rounding, keeps order stable against last-bit noise.
        scores = np.round(matrix.astype(np.float64) @ query.astype(np.float64), 6)
        order = np.argsort(-scores, kind="stable")[:top_k]  # rows are already in chunk-id order
        return [(ids[i], float(scores[i])) for i in order]
