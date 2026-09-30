"""The storage contract services depend on. `SqliteStore` implements it today; a Postgres store
implements the same methods. History is append-only: nothing here updates or deletes a
document version, chunk, corpus version or ingestion record.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from rag_forge.domain.models import (
    Chunk,
    Corpus,
    CorpusVersion,
    Document,
    DocumentVersion,
    Experiment,
    ExperimentRun,
    FileOutcome,
    IngestionRecord,
)


class ConcurrentModificationError(Exception):
    """The corpus gained a new version between read and commit."""


class CorpusStats(BaseModel):
    document_count: int
    chunk_count: int
    total_bytes: int
    total_chars: int
    last_ingestion: IngestionRecord | None


class VersionChange(BaseModel):
    document_id: str
    filename: str
    change: FileOutcome
    document_version_id: str | None
    previous_document_version_id: str | None


class CorpusStore(Protocol):
    kind: str

    def add_corpus(self, corpus: Corpus) -> Corpus: ...
    def list_corpora(self) -> list[Corpus]: ...
    def get_corpus(self, corpus_id: str) -> Corpus | None: ...
    def corpus_name_taken(self, name: str) -> bool: ...
    def corpus_stats(self, corpus: Corpus) -> CorpusStats: ...

    def members(self, corpus_id: str, version: int) -> dict[str, DocumentVersion]: ...
    def documents_by_filename(self, corpus_id: str) -> dict[str, Document]: ...
    def get_document(self, document_id: str) -> Document | None: ...
    def document_versions(self, document_id: str) -> list[DocumentVersion]: ...
    def get_document_version(self, version_id: str) -> DocumentVersion | None: ...
    def document_text(self, version_id: str) -> str | None: ...
    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...
    def chunks(
        self, version_id: str, chunking_hash: str, offset: int = 0, limit: int = 100
    ) -> tuple[list[Chunk], int]: ...

    def commit_version(
        self,
        corpus: Corpus,
        version: CorpusVersion,
        members: dict[str, str],
        new_documents: list[Document],
        new_versions: list[tuple[DocumentVersion, str, list[Chunk]]],
        record: IngestionRecord,
    ) -> Corpus: ...
    def add_ingestion(self, record: IngestionRecord) -> None: ...
    def list_ingestions(self, corpus_id: str) -> list[IngestionRecord]: ...
    def list_versions(self, corpus_id: str) -> list[CorpusVersion]: ...
    def version_changes(self, corpus_id: str, version: int) -> list[VersionChange]: ...

    def add_experiment(self, experiment: Experiment) -> Experiment: ...
    def list_experiments(self) -> list[Experiment]: ...
    def list_runs(self) -> list[ExperimentRun]: ...
