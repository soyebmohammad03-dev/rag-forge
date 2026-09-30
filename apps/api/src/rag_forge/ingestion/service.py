"""Ingestion: turn an upload batch into (at most) one new corpus version, with provenance.

Rules, per file in a batch:
  - identity is the filename within the corpus (directories are stripped);
  - same filename + same SHA-256 as the current version  -> unchanged;
  - same SHA-256 as a *different* current document       -> duplicate, not ingested;
  - unsupported, empty, oversize, corrupt or unreadable  -> rejected, with the reason;
  - otherwise a new document version is extracted, chunked and stored (added / modified).
If anything was added or modified, a new corpus version is committed atomically. Every batch,
including ones that change nothing, leaves an immutable IngestionRecord.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from pathlib import PurePath

from rag_forge.domain.models import (
    Chunk,
    Corpus,
    CorpusVersion,
    Document,
    DocumentVersion,
    ExtractionStatus,
    FileOutcome,
    IngestionFileResult,
    IngestionRecord,
    IngestionStatus,
    utcnow,
)
from rag_forge.ingestion.chunking import chunk_text
from rag_forge.ingestion.extraction import ExtractionError, extract
from rag_forge.provenance.environment import capture_environment
from rag_forge.storage.base import CorpusStore
from rag_forge.storage.blobs import BlobStore, sha256_hex

MAX_FILE_BYTES = 25 * 1024 * 1024


class CorpusNotFoundError(LookupError):
    pass


class DocumentNotInCorpusError(LookupError):
    pass


class IngestionService:
    def __init__(self, store: CorpusStore, blobs: BlobStore, max_bytes: int = MAX_FILE_BYTES):
        self.store = store
        self.blobs = blobs
        self.max_bytes = max_bytes
        # ponytail: one lock per process serialises ingestion; the store's optimistic version
        # check still guards against other processes. Per-corpus locks if throughput matters.
        self._lock = threading.Lock()

    def ingest(self, corpus_id: str, files: Sequence[tuple[str, bytes]]) -> IngestionRecord:
        with self._lock:
            corpus = self._corpus(corpus_id)
            started = utcnow()
            members = self.store.members(corpus.id, corpus.version)
            docs_by_name = self.store.documents_by_filename(corpus.id)
            owner_of_hash = {
                dv.content_sha256: (doc_id, dv.filename) for doc_id, dv in members.items()
            }
            next_members = {doc_id: dv.id for doc_id, dv in members.items()}
            new_docs: list[Document] = []
            new_versions: list[tuple[DocumentVersion, str, list[Chunk]]] = []
            results: list[IngestionFileResult] = []
            seen: set[str] = set()

            for raw_name, data in files:
                name = PurePath(raw_name.replace("\\", "/")).name.strip()
                digest = sha256_hex(data)
                result = partial(
                    IngestionFileResult,
                    filename=name or raw_name,
                    byte_size=len(data),
                    content_sha256=digest,
                )

                def reject(reason: str, result: partial[IngestionFileResult] = result) -> None:
                    results.append(result(outcome=FileOutcome.REJECTED, error=reason))

                if not name:
                    reject("missing filename")
                    continue
                if name in seen:
                    reject("filename appears more than once in this upload")
                    continue
                seen.add(name)
                if not data:
                    reject("empty file")
                    continue
                if len(data) > self.max_bytes:
                    reject(f"file exceeds the {self.max_bytes // (1024 * 1024)} MiB limit")
                    continue

                doc = docs_by_name.get(name)
                current = members.get(doc.id) if doc else None
                if current and current.content_sha256 == digest:
                    results.append(
                        result(
                            outcome=FileOutcome.UNCHANGED,
                            media_type=current.media_type,
                            parser=current.parser,
                            document_id=current.document_id,
                            document_version_id=current.id,
                        )
                    )
                    continue
                owner = owner_of_hash.get(digest)
                if owner is not None and (doc is None or owner[0] != doc.id):
                    results.append(
                        result(
                            outcome=FileOutcome.DUPLICATE,
                            duplicate_of=owner[0],
                            duplicate_of_filename=owner[1],
                        )
                    )
                    continue

                try:
                    extractor, extraction = extract(name, data)
                except ExtractionError as exc:
                    reject(str(exc))
                    continue

                if doc is None:
                    doc = Document(corpus_id=corpus.id, filename=name)
                    new_docs.append(doc)
                    docs_by_name[name] = doc
                    number = 1
                else:
                    number = len(self.store.document_versions(doc.id)) + 1
                version = DocumentVersion(
                    document_id=doc.id,
                    version=number,
                    filename=name,
                    media_type=extractor.media_type,
                    content_sha256=digest,
                    byte_size=len(data),
                    parser=f"{extractor.name}@{extractor.version}",
                    extraction_status=(
                        ExtractionStatus.PARTIAL
                        if extraction.warnings
                        else ExtractionStatus.COMPLETE
                    ),
                    extraction_warnings=extraction.warnings,
                    text_chars=len(extraction.text),
                    metadata=extraction.metadata,
                )
                chunks = chunk_text(
                    extraction.text, corpus.chunking, version.id, extraction.page_offsets
                )
                self.blobs.put(data)  # content-addressed; an orphan blob on rollback is harmless
                new_versions.append((version, extraction.text, chunks))
                owner_of_hash[digest] = (doc.id, name)
                next_members[doc.id] = version.id
                results.append(
                    result(
                        outcome=FileOutcome.MODIFIED if current else FileOutcome.ADDED,
                        media_type=version.media_type,
                        parser=version.parser,
                        document_id=doc.id,
                        document_version_id=version.id,
                        chunk_count=len(chunks),
                        warnings=extraction.warnings,
                    )
                )

            changed = bool(new_versions)
            if changed:
                status = IngestionStatus.COMPLETED
            elif results and all(r.outcome is FileOutcome.REJECTED for r in results):
                status = IngestionStatus.FAILED
            else:
                status = IngestionStatus.NO_CHANGE
            return self._finish(
                corpus, started, results, status, next_members, new_docs, new_versions
            )

    def remove(self, corpus_id: str, document_id: str) -> IngestionRecord:
        """Remove a document from the *next* corpus version. Its history is untouched."""
        with self._lock:
            corpus = self._corpus(corpus_id)
            started = utcnow()
            members = self.store.members(corpus.id, corpus.version)
            current = members.get(document_id)
            if current is None:
                raise DocumentNotInCorpusError(document_id)
            result = IngestionFileResult(
                filename=current.filename,
                outcome=FileOutcome.REMOVED,
                byte_size=current.byte_size,
                content_sha256=current.content_sha256,
                media_type=current.media_type,
                document_id=document_id,
                document_version_id=current.id,
            )
            next_members = {d: dv.id for d, dv in members.items() if d != document_id}
            return self._finish(
                corpus,
                started,
                [result],
                IngestionStatus.COMPLETED,
                next_members,
                [],
                [],
                removed=1,
            )

    def _corpus(self, corpus_id: str) -> Corpus:
        corpus = self.store.get_corpus(corpus_id)
        if corpus is None:
            raise CorpusNotFoundError(corpus_id)
        return corpus

    def _finish(
        self,
        corpus: Corpus,
        started: datetime,
        results: list[IngestionFileResult],
        status: IngestionStatus,
        next_members: dict[str, str],
        new_docs: list[Document],
        new_versions: list[tuple[DocumentVersion, str, list[Chunk]]],
        removed: int = 0,
    ) -> IngestionRecord:
        changed = status is IngestionStatus.COMPLETED
        record = IngestionRecord(
            corpus_id=corpus.id,
            status=status,
            version_before=corpus.version,
            version_after=corpus.version + 1 if changed else corpus.version,
            files=results,
            chunking=corpus.chunking,
            chunking_hash=corpus.chunking.config_hash(),
            environment=capture_environment(),
            started_at=started,
            finished_at=utcnow(),
        )
        if not changed:
            self.store.add_ingestion(record)
            return record
        counts = Counter(r.outcome for r in results)
        touched = counts[FileOutcome.ADDED] + counts[FileOutcome.MODIFIED]
        version = CorpusVersion(
            corpus_id=corpus.id,
            version=record.version_after,
            ingestion_id=record.id,
            document_count=len(next_members),
            added=counts[FileOutcome.ADDED],
            modified=counts[FileOutcome.MODIFIED],
            removed=removed,
            unchanged=len(next_members) - touched,
            created_at=record.finished_at,
        )
        self.store.commit_version(corpus, version, next_members, new_docs, new_versions, record)
        return record
