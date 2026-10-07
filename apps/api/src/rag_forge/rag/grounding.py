"""Claim-level grounding: measure whether each generated claim is supported by the evidence.

`GroundingVerifier` is a contract (`name`, `version`, `detects_contradiction`, `config_hash()`,
`verify()`), so a stronger verifier (an NLI model, an LLM judge) can register beside the
baseline without touching generation or presentation.

`LexicalSemanticVerifier` (`lexical-semantic@1`) is a transparent, local baseline:

- lexical coverage: share of the claim's content terms (BM25 analyzer, minus function words)
  that occur in the passage;
- semantic similarity: best cosine between the claim and the passage or any of its sentences,
  with the platform's pinned embedding model;
- per passage: supported if both reach the supported thresholds, weakly supported if both reach
  the weak thresholds, otherwise unsupported. A number in the claim that is absent from the
  passage, or a negation present on only one side, lowers the status by one level;
- a claim takes its best passage. If that is not "supported", the passages relevant to the claim
  are pooled, and the pool's coverage is tested, so a sentence that combines two passages can
  still be supported (flagged multi_passage).

It measures overlap of terms and meaning, not entailment. It cannot establish contradiction,
so it never emits "contradicted": a claim no passage supports is "unsupported", which says
nothing about whether the claim is false. Thresholds were calibrated on a handful of
hand-built pairs (docs/grounding/README.md), not learned, and are recorded in every report.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag_forge.domain.models import (
    AnswerGrounding,
    Claim,
    ClaimKind,
    Evidence,
    EvidenceSupport,
    GroundingReport,
    SupportStatus,
    canonical_hash,
)
from rag_forge.rag.claims import DraftClaim
from rag_forge.rag.text import negated, numbers, sentence_spans, terms
from rag_forge.retrieval.embedding import Embedder, EmbedderUnavailableError


class GroundingUnavailableError(RuntimeError):
    """The verifier could not run, so no claim may be reported as grounded."""


class GroundingVerifier(Protocol):
    name: str
    version: int
    detects_contradiction: bool

    def thresholds(self) -> dict[str, float]: ...

    def config_hash(self) -> str: ...

    def prepare(self) -> float:
        """Load any model the verifier needs; return the milliseconds spent (0 if loaded)."""
        ...

    def verify(self, claims: Sequence[DraftClaim], evidence: Sequence[Evidence]) -> list[Claim]:
        """Raises GroundingUnavailableError rather than returning unverified claims."""
        ...


_RANK = {
    SupportStatus.SUPPORTED: 2,
    SupportStatus.WEAKLY_SUPPORTED: 1,
    SupportStatus.UNSUPPORTED: 0,
}
_BY_RANK = {v: k for k, v in _RANK.items()}


class LexicalSemanticVerifier:
    name = "lexical-semantic"
    version = 1
    detects_contradiction = False

    def __init__(
        self,
        embedder: Embedder,
        supported_lexical: float = 0.7,
        supported_semantic: float = 0.85,
        weak_lexical: float = 0.5,
        weak_semantic: float = 0.75,
        pool_semantic: float = 0.6,
    ) -> None:
        self.embedder = embedder
        self._thresholds = {
            "supported_lexical": supported_lexical,
            "supported_semantic": supported_semantic,
            "weak_lexical": weak_lexical,
            "weak_semantic": weak_semantic,
            "pool_semantic": pool_semantic,
        }

    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    def config_hash(self) -> str:
        return canonical_hash(
            {
                "verifier": f"{self.name}@{self.version}",
                "thresholds": self._thresholds,
                "embedder": self.embedder.spec.config_hash(),
            }
        )

    def prepare(self) -> float:
        started = time.perf_counter()
        try:
            self.embedder.info()  # loads once per process; later calls return immediately
        except EmbedderUnavailableError as exc:
            raise GroundingUnavailableError(f"grounding embedder unavailable: {exc}") from exc
        return round((time.perf_counter() - started) * 1000, 3)

    def _status(self, lexical: float, semantic: float) -> SupportStatus:
        t = self._thresholds
        if lexical >= t["supported_lexical"] and semantic >= t["supported_semantic"]:
            return SupportStatus.SUPPORTED
        if lexical >= t["weak_lexical"] and semantic >= t["weak_semantic"]:
            return SupportStatus.WEAKLY_SUPPORTED
        return SupportStatus.UNSUPPORTED

    def verify(self, claims: Sequence[DraftClaim], evidence: Sequence[Evidence]) -> list[Claim]:
        factual = [c for c in claims if c.kind is ClaimKind.FACTUAL]
        sims: dict[str, list[tuple[float, str]]] = {}  # claim id -> per evidence (cos, sentence)
        if factual and evidence:
            units: list[tuple[int, str]] = []  # (evidence index, passage or sentence)
            for i, e in enumerate(evidence):
                pieces = dict.fromkeys([e.text, *(e.text[s:t] for s, t in sentence_spans(e.text))])
                units += [(i, p) for p in pieces]
            texts = [u for _, u in units] + [c.text for c in factual]
            # Embed in length order: a batch is padded to its longest text, so mixing whole
            # passages with short sentences costs several times more. Order is restored after.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            try:
                embedded = self.embedder.embed_documents([texts[i] for i in order])
            except EmbedderUnavailableError as exc:
                raise GroundingUnavailableError(f"grounding embedder unavailable: {exc}") from exc
            vectors = np.empty_like(embedded)
            vectors[order] = embedded
            unit_vecs, claim_vecs = vectors[: len(units)], vectors[len(units) :]
            cos = np.asarray(claim_vecs @ unit_vecs.T, dtype=np.float64)
            owner = np.array([i for i, _ in units])
            for row, claim in zip(cos, factual, strict=True):
                best = []
                for i in range(len(evidence)):
                    idx = np.flatnonzero(owner == i)
                    j = int(idx[np.argmax(row[idx])])
                    best.append((round(float(row[j]), 4), units[j][1]))
                sims[claim.id] = best
        return [self._claim(c, evidence, sims.get(c.id, [])) for c in claims]

    def _claim(
        self, draft: DraftClaim, evidence: Sequence[Evidence], sims: list[tuple[float, str]]
    ) -> Claim:
        cited = {c.evidence_id for c in draft.citations if c.valid}
        flags = []
        if any(not c.valid for c in draft.citations):
            flags.append("invalid_citation")
        base = {
            "id": draft.id,
            "index": draft.index,
            "text": draft.text,
            "raw_text": draft.raw_text,
            "char_start": draft.char_start,
            "char_end": draft.char_end,
            "kind": draft.kind,
            "content_terms": draft.content_terms,
            "citations": draft.citations,
            "cited_evidence_ids": [c.evidence_id for c in draft.citations if c.evidence_id],
        }
        if draft.kind is not ClaimKind.FACTUAL:
            return Claim(
                **base,
                supporting_evidence_ids=[],
                support=SupportStatus.NOT_APPLICABLE,
                support_score=None,
                missing_terms=[],
                unmatched_numbers=[],
                evidence=[],
                flags=flags,
                rationale=f"{draft.kind.value}: not checked against evidence",
            )
        if not draft.citations:
            flags.append("uncited")
        claim_terms = draft.content_terms
        claim_numbers = numbers(claim_terms)
        claim_negated = negated(draft.text)
        t = self._thresholds
        scored: list[tuple[EvidenceSupport, set[str]]] = []
        for e, (semantic, sentence) in zip(evidence, sims, strict=True):
            e_terms = terms(e.text)
            lexical = round(sum(x in e_terms for x in claim_terms) / len(claim_terms), 4)
            status = self._status(lexical, semantic)
            level = _RANK[status]
            if any(n not in e_terms for n in claim_numbers):
                level = min(level, 1)
            if claim_negated != negated(sentence):
                level = max(level - 1, 0)
            scored.append(
                (
                    EvidenceSupport(
                        evidence_id=e.id,
                        citation=e.citation,
                        cited=e.id in cited,
                        lexical_coverage=lexical,
                        semantic_similarity=semantic,
                        best_sentence=sentence,
                        score=round((lexical + semantic) / 2, 4),
                        status=_BY_RANK[level],
                    ),
                    e_terms,
                )
            )
        order = sorted(
            range(len(scored)), key=lambda i: (-_RANK[scored[i][0].status], -scored[i][0].score, i)
        )
        ranked = [scored[i] for i in order]
        if not ranked:  # no evidence at all (not reachable through the service)
            return Claim(
                **base,
                supporting_evidence_ids=[],
                support=SupportStatus.UNSUPPORTED,
                support_score=0.0,
                missing_terms=claim_terms,
                unmatched_numbers=claim_numbers,
                evidence=[],
                flags=flags,
                rationale="no evidence to check against",
            )
        best, best_terms = ranked[0]
        support, score = best.status, best.score
        supporting = [es for es, _ in ranked if es.status is not SupportStatus.UNSUPPORTED]
        covered = set().union(*(et for es, et in ranked if es in supporting)) or best_terms
        rationale = (
            f"best passage {best.citation}: lexical {best.lexical_coverage:.2f}, "
            f"semantic {best.semantic_similarity:.2f} -> {best.status.value}"
        )
        if support is not SupportStatus.SUPPORTED:
            pool = [
                (es, et)
                for es, et in ranked
                if es.semantic_similarity >= t["pool_semantic"]
                and any(x in et for x in claim_terms)
            ]
            if len(pool) >= 2:
                pooled = set().union(*(et for _, et in pool))
                lexical = sum(x in pooled for x in claim_terms) / len(claim_terms)
                semantic = max(es.semantic_similarity for es, _ in pool)
                level = _RANK[self._status(lexical, semantic)]
                if any(n not in pooled for n in claim_numbers):
                    level = min(level, 1)
                if all(claim_negated != negated(es.best_sentence) for es, _ in pool):
                    level = max(level - 1, 0)
                if level > _RANK[support]:
                    support = _BY_RANK[level]
                    score = round((lexical + semantic) / 2, 4)
                    supporting = [es for es, _ in pool]
                    covered = pooled
                    flags.append("multi_passage")
                    rationale = (
                        f"passages {'+'.join(es.citation for es in supporting)} together: "
                        f"lexical {lexical:.2f}, semantic {semantic:.2f} -> {support.value}"
                    )
        missing = [x for x in claim_terms if x not in covered]
        unmatched = [n for n in claim_numbers if n not in covered]
        if unmatched:
            flags.append("number_mismatch")
        if claim_negated != negated(best.best_sentence):
            flags.append("negation_mismatch")
        supporting_ids = [es.evidence_id for es in supporting]
        if cited - set(supporting_ids):
            flags.append("cited_not_supporting")
        return Claim(
            **base,
            supporting_evidence_ids=supporting_ids,
            support=support,
            support_score=score,
            missing_terms=missing,
            unmatched_numbers=unmatched,
            evidence=[es for es, _ in ranked],
            flags=flags,
            rationale=rationale,
        )


def summarize(
    claims: list[Claim],
    evidence: Sequence[Evidence],
    verifier: GroundingVerifier,
    latency_ms: float,
    load_ms: float = 0.0,
) -> GroundingReport:
    factual = [c for c in claims if c.kind is ClaimKind.FACTUAL]
    count = {s: sum(c.support is s for c in factual) for s in SupportStatus}
    sup, weak = count[SupportStatus.SUPPORTED], count[SupportStatus.WEAKLY_SUPPORTED]
    abstentions = sum(c.kind is ClaimKind.ABSTENTION for c in claims)
    supporting = {eid for c in factual for eid in c.supporting_evidence_ids}
    valid = [(c, x) for c in factual for x in c.citations if x.valid]
    if not claims:
        status = AnswerGrounding.NO_CLAIMS
    elif not factual:
        status = AnswerGrounding.ABSTAINED if abstentions else AnswerGrounding.NO_CLAIMS
    elif sup == len(factual):
        status = AnswerGrounding.GROUNDED
    elif sup + weak == 0:
        status = AnswerGrounding.UNGROUNDED
    else:
        status = AnswerGrounding.PARTIALLY_GROUNDED
    return GroundingReport(
        verifier=verifier.name,
        verifier_version=str(verifier.version),
        config_hash=verifier.config_hash(),
        thresholds=verifier.thresholds(),
        detects_contradiction=verifier.detects_contradiction,
        status=status,
        claims=len(claims),
        factual_claims=len(factual),
        supported=sup,
        weakly_supported=weak,
        unsupported=count[SupportStatus.UNSUPPORTED],
        contradicted=count[SupportStatus.CONTRADICTED],
        abstentions=abstentions,
        non_assertive=sum(c.kind is ClaimKind.NON_ASSERTIVE for c in claims),
        grounding_score=round((sup + 0.5 * weak) / len(factual), 4) if factual else None,
        evidence_coverage=round(len(supporting) / len(evidence), 4) if evidence else None,
        citation_coverage=(
            round(sum(any(x.valid for x in c.citations) for c in factual) / len(factual), 4)
            if factual
            else None
        ),
        citation_precision=(
            round(sum(x.evidence_id in c.supporting_evidence_ids for c, x in valid) / len(valid), 4)
            if valid
            else None
        ),
        invalid_citations=sum(not x.valid for c in claims for x in c.citations),
        grounding_hash=canonical_hash(
            [
                [c.id, c.kind, c.support, c.support_score, c.supporting_evidence_ids, c.flags]
                for c in claims
            ]
        ),
        load_ms=load_ms,
        latency_ms=latency_ms,
    )
