"""Corpus management, ingestion, inspection and retrieval routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from rag_forge.api.schemas import (
    ChunkPage,
    CorpusCreate,
    CorpusSummary,
    DocumentDetail,
    DocumentSummary,
    NotImplementedDetail,
    SupportedFormat,
)
from rag_forge.domain.models import (
    Corpus,
    CorpusVersion,
    IngestionRecord,
    RetrievalRequest,
    RetrievalResponse,
)
from rag_forge.ingestion.extraction import EXTRACTORS
from rag_forge.ingestion.service import DocumentNotInCorpusError, IngestionService
from rag_forge.retrieval.service import (
    CorpusVersionNotFoundError,
    RetrievalService,
    StrategyNotAvailableError,
)
from rag_forge.storage.base import CorpusStore, VersionChange

router = APIRouter(prefix="/api/v1", tags=["corpus"])


def _store(request: Request) -> CorpusStore:
    store: CorpusStore = request.app.state.store
    return store


def _service(request: Request) -> IngestionService:
    service: IngestionService = request.app.state.ingestion
    return service


def _corpus(request: Request, corpus_id: str) -> Corpus:
    corpus = _store(request).get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="corpus not found")
    return corpus


def _summary(store: CorpusStore, corpus: Corpus) -> CorpusSummary:
    return CorpusSummary(corpus=corpus, stats=store.corpus_stats(corpus))


@router.get("/ingestion/formats", response_model=list[SupportedFormat], tags=["ingestion"])
def supported_formats() -> list[SupportedFormat]:
    return [
        SupportedFormat(
            media_type=e.media_type, extensions=list(e.extensions), parser=f"{e.name}@{e.version}"
        )
        for e in EXTRACTORS
    ]


@router.get("/corpora", response_model=list[CorpusSummary])
def list_corpora(request: Request) -> list[CorpusSummary]:
    store = _store(request)
    return [_summary(store, c) for c in store.list_corpora()]


@router.post("/corpora", response_model=CorpusSummary, status_code=status.HTTP_201_CREATED)
def create_corpus(body: CorpusCreate, request: Request) -> CorpusSummary:
    store = _store(request)
    if store.corpus_name_taken(body.name):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="a corpus with this name exists")
    return _summary(store, store.add_corpus(Corpus(**body.model_dump())))


@router.get("/corpora/{corpus_id}", response_model=CorpusSummary)
def get_corpus(corpus_id: str, request: Request) -> CorpusSummary:
    return _summary(_store(request), _corpus(request, corpus_id))


@router.get("/corpora/{corpus_id}/documents", response_model=list[DocumentSummary])
def list_documents(
    corpus_id: str,
    request: Request,
    version: Annotated[int | None, Query(ge=0, description="Defaults to current")] = None,
) -> list[DocumentSummary]:
    corpus = _corpus(request, corpus_id)
    if version is not None and version > corpus.version:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="corpus version not found")
    store = _store(request)
    members = store.members(corpus.id, corpus.version if version is None else version)
    return [
        DocumentSummary(
            document_id=doc_id,
            filename=dv.filename,
            version_count=len(store.document_versions(doc_id)),
            current=dv,
        )
        for doc_id, dv in members.items()
    ]


@router.post("/corpora/{corpus_id}/documents", response_model=IngestionRecord, tags=["ingestion"])
async def ingest_documents(
    corpus_id: str,
    request: Request,
    files: Annotated[list[UploadFile], File(description="One or more .txt, .md or .pdf files")],
) -> IngestionRecord:
    """Ingest a batch. Returns the ingestion record: one outcome per file, and the new version."""
    corpus = _corpus(request, corpus_id)
    service = _service(request)
    # Read at most one byte past the limit so oversize uploads are rejected without buffering.
    batch = [(f.filename or "", await f.read(service.max_bytes + 1)) for f in files]
    return await run_in_threadpool(service.ingest, corpus.id, batch)


@router.get("/corpora/{corpus_id}/documents/{document_id}", response_model=DocumentDetail)
def get_document(corpus_id: str, document_id: str, request: Request) -> DocumentDetail:
    corpus = _corpus(request, corpus_id)
    store = _store(request)
    doc = store.get_document(document_id)
    if doc is None or doc.corpus_id != corpus.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="document not found")
    current = store.members(corpus.id, corpus.version).get(doc.id)
    return DocumentDetail(
        document=doc,
        versions=store.document_versions(doc.id),
        current_version_id=current.id if current else None,
    )


@router.delete(
    "/corpora/{corpus_id}/documents/{document_id}",
    response_model=IngestionRecord,
    tags=["ingestion"],
)
def remove_document(corpus_id: str, document_id: str, request: Request) -> IngestionRecord:
    """Remove from the next corpus version. Earlier versions keep the document."""
    corpus = _corpus(request, corpus_id)
    try:
        return _service(request).remove(corpus.id, document_id)
    except DocumentNotInCorpusError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="document is not in the current corpus version"
        ) from exc


@router.get("/document-versions/{version_id}/chunks", response_model=ChunkPage)
def list_chunks(
    version_id: str,
    request: Request,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> ChunkPage:
    store = _store(request)
    dv = store.get_document_version(version_id)
    doc = store.get_document(dv.document_id) if dv else None
    corpus = store.get_corpus(doc.corpus_id) if doc else None
    if corpus is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="document version not found")
    chash = corpus.chunking.config_hash()
    items, total = store.chunks(version_id, chash, offset, limit)
    return ChunkPage(
        document_version_id=version_id, chunking_hash=chash, total=total, offset=offset, items=items
    )


@router.get("/corpora/{corpus_id}/versions", response_model=list[CorpusVersion])
def list_versions(corpus_id: str, request: Request) -> list[CorpusVersion]:
    return _store(request).list_versions(_corpus(request, corpus_id).id)


@router.get("/corpora/{corpus_id}/versions/{version}/changes", response_model=list[VersionChange])
def version_changes(corpus_id: str, version: int, request: Request) -> list[VersionChange]:
    corpus = _corpus(request, corpus_id)
    if not 1 <= version <= corpus.version:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="corpus version not found")
    return _store(request).version_changes(corpus.id, version)


@router.get(
    "/corpora/{corpus_id}/ingestions", response_model=list[IngestionRecord], tags=["ingestion"]
)
def list_ingestions(corpus_id: str, request: Request) -> list[IngestionRecord]:
    return _store(request).list_ingestions(_corpus(request, corpus_id).id)


@router.post(
    "/corpora/{corpus_id}/retrieve",
    response_model=RetrievalResponse,
    tags=["retrieval"],
    responses={501: {"model": NotImplementedDetail}},
)
def retrieve(corpus_id: str, body: RetrievalRequest, request: Request) -> RetrievalResponse:
    """Rank chunks of one corpus version for a query. Defaults to the current version."""
    corpus = _corpus(request, corpus_id)
    service: RetrievalService = request.app.state.retrieval
    try:
        return service.retrieve(corpus.id, body)
    except CorpusVersionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except StrategyNotAvailableError as exc:
        detail = NotImplementedDetail(capability="Retrieval strategy", message=str(exc))
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=detail.model_dump()) from exc
