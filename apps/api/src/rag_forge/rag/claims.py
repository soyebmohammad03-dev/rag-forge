"""Split a generated answer into claims and parse the citations the generator wrote.

A claim is one sentence of the raw answer (`text.sentence_spans`). Citation markers are
`[E<n>]`, or several ids in one bracket (`[E1, E3]`). Markers that open a sentence are attached
to the previous one, because the prompt asks for citations after each sentence and models often
write "fact. [E1]". A citation is valid only if its label names a supplied passage; an invalid
one is kept and counted, never dropped or remapped.

A sentence is an abstention if it says the evidence is insufficient (fixed patterns below),
non-assertive if it has no content terms or is a lead-in ending in ":", and factual otherwise.

Claims are extracted before any verification, and support is left unset here: a claim's
citations are what the generator wrote (generated), never what a verifier found (measured).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_forge.domain.models import Citation, ClaimKind, Evidence, canonical_hash
from rag_forge.rag.text import content_terms, sentence_spans

EXTRACTOR = "sentence-citation@1"

_MARKER = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]", re.IGNORECASE)
_LEADING_MARKERS = re.compile(r"^(?:\s*\[\s*E\d+(?:\s*[,;]\s*E\d+)*\s*\])+", re.IGNORECASE)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_ABSTENTION = re.compile(
    r"\b(?:"
    r"(?:the\s+)?evidence\s+(?:is|was|seems|appears)\s+(?:insufficient|not\s+sufficient)"
    r"|insufficient\s+(?:evidence|information)"
    r"|not\s+enough\s+(?:evidence|information)"
    r"|(?:passages?|evidence|context)\s+(?:does|do)\s+not\s+(?:contain|provide|mention|say|state"
    r"|answer|include|specify)"
    r"|cannot\s+be\s+(?:determined|answered)\s+from"
    r"|insufficient_evidence"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class DraftClaim:
    """A claim before verification."""

    index: int
    id: str
    raw_text: str
    text: str
    char_start: int
    char_end: int
    kind: ClaimKind
    content_terms: list[str]
    citations: list[Citation] = field(default_factory=list)


def _citations(text: str, offset: int, by_label: dict[str, Evidence]) -> list[Citation]:
    out = []
    for m in _MARKER.finditer(text):
        for label in re.split(r"\s*[,;]\s*", m.group(1)):
            label = label.upper()
            ev = by_label.get(label)
            out.append(
                Citation(
                    label=label,
                    evidence_id=ev.id if ev else None,
                    valid=ev is not None,
                    char_start=offset + m.start(),
                    char_end=offset + m.end(),
                )
            )
    return out


def strip_markers(text: str) -> str:
    stripped = _MARKER.sub("", text)
    stripped = re.sub(r"\s+([.,;:!?])", r"\1", stripped)
    return _LIST_PREFIX.sub("", re.sub(r"\s{2,}", " ", stripped)).strip()


def extract_claims(answer: str, evidence: list[Evidence], answer_hash: str) -> list[DraftClaim]:
    by_label = {e.citation.upper(): e for e in evidence}
    claims: list[DraftClaim] = []
    for start, end in sentence_spans(answer):
        raw = answer[start:end]
        lead = _LEADING_MARKERS.match(raw)
        if lead and claims:  # "fact. [E1] Next fact." -> [E1] belongs to "fact."
            prev = claims[-1]
            prev.citations += _citations(raw[: lead.end()], start, by_label)
            prev.char_end = start + lead.end()
            prev.raw_text = answer[prev.char_start : prev.char_end]
            rest = raw[lead.end() :]
            start += lead.end() + len(rest) - len(rest.lstrip())
            raw = answer[start:end]
            if not raw:
                continue
        text = strip_markers(raw)
        if not text:  # markers only, with no previous sentence to attach them to
            continue
        if _ABSTENTION.search(text):
            kind = ClaimKind.ABSTENTION
        else:
            lead_in = text.endswith(":")  # "Here is what the evidence says:"
            kind = (
                ClaimKind.FACTUAL
                if content_terms(text) and not lead_in
                else ClaimKind.NON_ASSERTIVE
            )
        index = len(claims)
        claims.append(
            DraftClaim(
                index=index,
                id=f"clm_{canonical_hash([answer_hash, index, start, end])[:20]}",
                raw_text=raw,
                text=text,
                char_start=start,
                char_end=end,
                kind=kind,
                content_terms=content_terms(text) if kind is ClaimKind.FACTUAL else [],
                citations=_citations(raw, start, by_label),
            )
        )
    return claims
