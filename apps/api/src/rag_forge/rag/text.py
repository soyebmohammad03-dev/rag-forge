"""Text helpers shared by evidence selection, claim extraction and grounding.

Terms come from the BM25 analyzer (`retrieval.analysis.analyze`), so "a term the claim uses" means
the same thing everywhere in the platform. Content terms additionally drop the router's function
words (interrogatives, auxiliaries, pronouns, prepositions), which carry no checkable content.
"""

from __future__ import annotations

import re

from rag_forge.retrieval.analysis import analyze
from rag_forge.router.analyzer import ABBREVIATIONS, FUNCTION_WORDS, NEGATION

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_LAST_WORD = re.compile(r"(\w+(?:\.\w+)*)\.$")
_NO_SPLIT_AFTER = ABBREVIATIONS | {"dr", "mr", "mrs", "ms", "prof", "st", "fig", "no", "approx"}
_NUMBER = re.compile(r"\d")
_NEGATION = re.compile(r"\b(?:" + "|".join(NEGATION) + r")\b|n't\b", re.IGNORECASE)


def terms(text: str) -> set[str]:
    return set(analyze(text))


def content_terms(text: str) -> list[str]:
    """Distinct analyzer terms that are not function words, in first-occurrence order."""
    return list(dict.fromkeys(t for t in analyze(text) if t not in FUNCTION_WORDS))


def numbers(items: list[str]) -> list[str]:
    return [t for t in items if _NUMBER.search(t)]


def negated(text: str) -> bool:
    return _NEGATION.search(text) is not None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence (start, end) offsets: split after . ! ? plus whitespace, and on newlines.

    A period after a known abbreviation (e.g., i.e., Dr., Fig. ...) or a single-letter initial
    does not end a sentence. Spans are trimmed of surrounding whitespace and empty spans are
    dropped, so `text[start:end]` is always a non-empty sentence exactly as written.
    """
    spans = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        if "\n" not in m.group(0):
            word = _LAST_WORD.search(text[start : m.start()])
            if word and (word.group(1).casefold() in _NO_SPLIT_AFTER or len(word.group(1)) == 1):
                continue
        spans.append((start, m.start() + len(m.group(0).rstrip())))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        segment = text[s:e]
        stripped = segment.strip()
        if stripped:
            lead = len(segment) - len(segment.lstrip())
            out.append((s + lead, s + lead + len(stripped)))
    return out
