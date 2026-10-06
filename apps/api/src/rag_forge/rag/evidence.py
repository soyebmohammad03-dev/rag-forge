"""Evidence selection and deterministic context assembly.

Selection walks the final ranking (after reranking, if any) best first and keeps a passage
unless an explicit rule rejects it: score floor, near-duplicate of a passage already kept,
per-document cap, item budget or token budget. Every candidate's outcome is recorded, so no
passage is ever dropped or kept silently. Equal inputs and parameters give equal selections and
equal evidence ids.

The context holds only the selected passages, each under its own citation header, in selection
order. Its hash covers the corpus version, the prompt template, the evidence ids and the exact
evidence text, so two answers with equal context hashes were generated from identical evidence.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

from rag_forge.domain.models import (
    ChatMessage,
    ContextBlock,
    Evidence,
    EvidenceParams,
    EvidenceRetrieval,
    EvidenceSelection,
    GenerationContext,
    RetrievalHit,
    RetrievalResponse,
    SelectionDecision,
    SelectionOutcome,
    canonical_hash,
)
from rag_forge.rag.prompt import PromptTemplate
from rag_forge.rag.text import jaccard, terms

SELECTOR = "ranked-greedy-diverse@1"
BLOCK_SEPARATOR = "\n\n"

TokenCounter = Callable[[str], int]


class ContextBudgetError(ValueError):
    """The selected evidence or the full prompt does not fit the configured or model limits."""


def evidence_id(
    corpus_id: str, corpus_version: int, chunk_id: str, start: int, end: int, sha: str
) -> str:
    key = canonical_hash([corpus_id, corpus_version, chunk_id, start, end, sha])
    return f"evd_{key[:20]}"


def block_header(citation: str, filename: str) -> str:
    return f"[{citation}] ({filename})"


def _decision(
    hit: RetrievalHit,
    outcome: SelectionOutcome,
    detail: str,
    tokens: int | None = None,
    evidence: str | None = None,
    similar_to: str | None = None,
    similarity: float | None = None,
) -> SelectionDecision:
    return SelectionDecision(
        chunk_id=hit.chunk.id,
        document_id=hit.result.document_id,
        filename=hit.filename,
        rank=hit.result.rank,
        score=hit.result.score,
        outcome=outcome,
        detail=detail,
        token_count=tokens,
        evidence_id=evidence,
        similar_to=similar_to,
        similarity=similarity,
    )


def select_evidence(
    retrieval: RetrievalResponse,
    params: EvidenceParams,
    count_tokens: TokenCounter,
    tokenizer: str,
) -> EvidenceSelection:
    started = time.perf_counter()
    prov = retrieval.provenance
    selected: list[Evidence] = []
    selected_terms: list[tuple[str, set[str]]] = []
    per_document: dict[str, int] = {}
    decisions: list[SelectionDecision] = []
    used = 0
    for hit in retrieval.hits:
        chunk, result = hit.chunk, hit.result

        if len(selected) >= params.max_items:
            decisions.append(
                _decision(
                    hit, SelectionOutcome.OVER_ITEM_BUDGET, f"max_items={params.max_items} reached"
                )
            )
            continue
        if params.min_score is not None and result.score < params.min_score:
            decisions.append(
                _decision(
                    hit,
                    SelectionOutcome.BELOW_MIN_SCORE,
                    f"score {result.score:.4f} < min_score {params.min_score}",
                )
            )
            continue
        cap = params.max_per_document
        if cap is not None and per_document.get(result.document_version_id, 0) >= cap:
            decisions.append(
                _decision(
                    hit,
                    SelectionOutcome.DOCUMENT_CAP,
                    f"{hit.filename} already supplied {cap} passage(s)",
                )
            )
            continue
        chunk_terms = terms(chunk.text)
        threshold = params.near_duplicate_threshold
        if threshold is not None:
            nearest = max(
                ((eid, jaccard(chunk_terms, t)) for eid, t in selected_terms),
                key=lambda x: x[1],
                default=None,
            )
            if nearest is not None and nearest[1] >= threshold:
                decisions.append(
                    _decision(
                        hit,
                        SelectionOutcome.NEAR_DUPLICATE,
                        f"term Jaccard {nearest[1]:.3f} >= {threshold} with {nearest[0]}",
                        similar_to=nearest[0],
                        similarity=round(nearest[1], 4),
                    )
                )
                continue
        citation = f"E{len(selected) + 1}"
        block = block_header(citation, hit.filename) + "\n" + chunk.text
        tokens = count_tokens(block) + (count_tokens(BLOCK_SEPARATOR) if selected else 0)
        if used + tokens > params.max_context_tokens:
            decisions.append(
                _decision(
                    hit,
                    SelectionOutcome.OVER_TOKEN_BUDGET,
                    f"{tokens} tokens; {params.max_context_tokens - used} of "
                    f"{params.max_context_tokens} left",
                    tokens=tokens,
                )
            )
            continue
        sha = hashlib.sha256(chunk.text.encode()).hexdigest()
        eid = evidence_id(
            prov.corpus_id, prov.corpus_version, chunk.id, chunk.char_start, chunk.char_end, sha
        )
        rr = hit.rerank
        reason = f"final rank {result.rank}, score {result.score:.4f}"
        if rr is not None:
            reason += f" (reranked from rank {rr.original_rank})"
        selected.append(
            Evidence(
                id=eid,
                citation=citation,
                corpus_id=prov.corpus_id,
                corpus_version=prov.corpus_version,
                chunking_hash=chunk.chunking_hash,
                document_id=result.document_id,
                document_version_id=result.document_version_id,
                document_version=hit.document_version,
                filename=hit.filename,
                media_type=hit.media_type,
                chunk_id=chunk.id,
                chunk_ordinal=chunk.ordinal,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
                text_sha256=sha,
                token_count=tokens,
                retrieval=EvidenceRetrieval(
                    strategy=result.strategy,
                    retriever=result.retriever,
                    rank=result.rank,
                    score=result.score,
                    upstream_rank=rr.original_rank if rr else None,
                    upstream_score=rr.original_score if rr else None,
                    reranker_score=rr.reranker_score if rr else None,
                    configuration_hash=prov.configuration_hash,
                ),
                selection_rank=len(selected) + 1,
                selection_score=result.score,
                selection_reason=reason,
            )
        )
        selected_terms.append((eid, chunk_terms))
        per_document[result.document_version_id] = (
            per_document.get(result.document_version_id, 0) + 1
        )
        used += tokens
        decisions.append(
            _decision(hit, SelectionOutcome.SELECTED, reason, tokens=tokens, evidence=eid)
        )

    return EvidenceSelection(
        params=params,
        params_hash=params.config_hash(),
        selector=SELECTOR,
        candidates=len(retrieval.hits),
        selected=selected,
        decisions=decisions,
        tokens_used=used,
        tokenizer=tokenizer,
        selection_hash=canonical_hash(
            {
                "selector": SELECTOR,
                "params": params.model_dump(mode="json"),
                "tokenizer": tokenizer,
                "evidence": [e.id for e in selected],
            }
        ),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def build_context(
    question: str,
    evidence: list[Evidence],
    template: PromptTemplate,
    max_context_tokens: int,
    count_tokens: TokenCounter,
    tokenizer: str,
) -> GenerationContext:
    """Render selected evidence into the prompt. Raises if it exceeds the token budget."""
    if not evidence:
        raise ContextBudgetError("cannot build a context without evidence")
    corpora = {(e.corpus_id, e.corpus_version) for e in evidence}
    if len(corpora) != 1:  # evidence from different corpus versions must never be mixed
        raise ContextBudgetError(f"evidence spans several corpus versions: {sorted(corpora)}")
    (corpus_id, corpus_version) = corpora.pop()
    blocks = []
    for e in evidence:
        header = block_header(e.citation, e.filename)
        blocks.append(
            ContextBlock(
                citation=e.citation,
                evidence_id=e.id,
                header=header,
                text=e.text,
                token_count=count_tokens(header + "\n" + e.text),
            )
        )
    evidence_text = BLOCK_SEPARATOR.join(f"{b.header}\n{b.text}" for b in blocks)
    context_tokens = count_tokens(evidence_text)
    if context_tokens > max_context_tokens:
        raise ContextBudgetError(
            f"evidence context is {context_tokens} tokens, over max_context_tokens "
            f"{max_context_tokens}"
        )
    messages: list[ChatMessage] = template.render(question, evidence_text)
    return GenerationContext(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        prompt_template=template.id,
        blocks=blocks,
        evidence_text=evidence_text,
        messages=messages,
        context_tokens=context_tokens,
        max_context_tokens=max_context_tokens,
        tokenizer=tokenizer,
        context_hash=canonical_hash(
            {
                "corpus_id": corpus_id,
                "corpus_version": corpus_version,
                "template": template.id,
                "evidence": [[b.evidence_id, b.citation, b.header, b.text] for b in blocks],
            }
        ),
        prompt_hash=canonical_hash([m.model_dump(mode="json") for m in messages]),
    )
