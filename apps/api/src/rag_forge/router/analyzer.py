"""Query intelligence: measured, deterministic features of a query, behind a small contract.

`HeuristicQueryAnalyzer` is a transparent rule-and-weight baseline, not a trained model. Every
feature is a pattern match or a count over the normalised text; every score is the clipped sum of
listed `feature value x weight` contributions; every label is a threshold on those scores. A
learned classifier can replace it by implementing `QueryAnalyzer` and registering under a new
name. Corpus statistics are optional input (`CorpusQuerySignals`), computed by the caller.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any, Protocol

from rag_forge.domain.models import (
    Complexity,
    CorpusQuerySignals,
    EvidenceNeed,
    QueryAnalysis,
    QueryClass,
    QueryFeatures,
    QueryLabels,
    QuerySignal,
    QuestionType,
    SignalContribution,
    canonical_hash,
)
from rag_forge.retrieval.analysis import STOPWORDS, analyze


class QueryAnalyzer(Protocol):
    name: str
    version: str

    def config_hash(self) -> str: ...

    def analyze(self, query: str, corpus: CorpusQuerySignals | None = None) -> QueryAnalysis: ...


WH = ("what", "which", "who", "whom", "whose", "when", "where", "why", "how")
AUX = (
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "will",
    "has", "have", "had", "may", "might", "must",
)  # fmt: skip
REQUEST = ("explain", "describe", "define", "list", "compare", "name", "give", "show", "tell")
FUNCTION_WORDS = STOPWORDS | set(WH) | set(AUX) | {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "his", "her", "its", "them",
    "about", "above", "across", "after", "against", "among", "around", "before", "behind",
    "between", "during", "from", "how", "over", "per", "than", "through", "under", "up", "upon",
    "via", "within", "without", "so", "too", "very", "also", "just", "only", "any", "some",
    "each", "every", "all", "both", "more", "most", "other", "own", "same", "those", "which",
}  # fmt: skip

COMPARISON = (
    "compare", "compared", "comparison", "comparing", "versus", "vs", "difference between",
    "differences between", "better than", "worse than", "similarities", "contrast",
    "trade-off", "tradeoff", "trade off",
)  # fmt: skip
MULTI_HOP = (
    "whose", "which was", "who was", "that was", "and then", "both", "relationship between",
    "relation between", "in relation to", "depends on", "depend on", "caused by", "leads to",
    "lead to", "result in", "results in", "because of", "impact of", "effect of", "affects",
    "influence of",
)  # fmt: skip
TEMPORAL = (
    "before", "after", "since", "until", "during", "latest", "recent", "recently", "current",
    "currently", "today", "yesterday", "first", "last", "earliest", "newest", "oldest", "now",
    "ago",
)  # fmt: skip
NEGATION = ("not", "no", "never", "without", "except", "excluding", "nor", "none")
VAGUE = (
    "thing", "things", "stuff", "something", "anything", "everything", "etc", "whatever",
    "somehow", "it", "this", "these", "those", "they", "them",
)  # fmt: skip
ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "cf"}

# Weights of each signal's contributions (each signal's weights sum to 1) and label thresholds.
# Part of the config hash: change a number, and every analysis hash changes with it.
CONFIG: dict[str, Any] = {
    "signals": {
        "lexical": {
            "identifiers": 0.35,
            "quoted_phrases": 0.20,
            "numbers": 0.10,
            "keyword_form": 0.20,
            "term_density": 0.15,
        },
        "semantic": {
            "question_form": 0.25,
            "function_words": 0.20,
            "length": 0.20,
            "descriptive_need": 0.20,
            "no_exact_anchor": 0.15,
        },
        "complexity": {
            "key_terms": 0.30,
            "concepts": 0.25,
            "multi_hop": 0.25,
            "comparison": 0.10,
            "constraints": 0.10,
        },
    },
    "class_margin": 0.15,  # |lexical - semantic| at least this picks a side, else mixed
    "complexity": {"moderate": 0.25, "complex": 0.5},
    "definition_max_words": 4,  # "what is X" with X this short is a definition question
}

_QUOTED = re.compile(r'"([^"]+)"')
_WORD = re.compile(r"\w+")
_YEAR = re.compile(r"\b(?:1[89]|20)\d{2}\b")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?%?$")
_STRIP = "\"'()[]{}<>.,;:!?"
_SEGMENTS = re.compile(
    r"\s*(?:[,;]|\band\b|\bor\b|\bvs\.?|\bversus\b|\bcompared (?:to|with)\b|\bas well as\b)\s*"
)
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def normalize(query: str) -> str:
    """NFKC, straight quotes, single spaces, trimmed. Case is kept: capitalisation is a feature."""
    return " ".join(unicodedata.normalize("NFKC", query).translate(_QUOTES).split())


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            out.append(item)
    return out


def _phrases(lower: str, phrases: Iterable[str]) -> list[str]:
    """Phrases present as whole words, in the order they first occur in the text."""
    found = []
    for p in phrases:
        m = re.search(rf"(?<!\w){re.escape(p)}(?!\w)", lower)
        if m:
            found.append((m.start(), p))
    return [p for _, p in sorted(found)]


def is_identifier(token: str) -> bool:
    if token.casefold() in ABBREVIATIONS or _NUMBER.match(token):
        return False
    letters = any(c.isalpha() for c in token)
    return letters and (
        any(c.isdigit() for c in token)
        or "_" in token
        or re.search(r"[a-z][A-Z]", token) is not None
        or sum(c.isupper() for c in token) >= 2
        or re.search(r"\w[./]\w", token) is not None
    )


def _question_type(lower: str, words: list[str], comparison: list[str]) -> QuestionType:
    first = words[0] if words else ""
    if comparison:
        return QuestionType.COMPARISON
    if first == "define" or "meaning of" in lower:
        return QuestionType.DEFINITION
    m = re.match(r"(?:what is|what's|what are)\s+(.*)", lower)
    if (
        m
        and not m.group(1).startswith("the ")
        and len(_WORD.findall(m.group(1))) <= int(CONFIG["definition_max_words"])
    ):
        return QuestionType.DEFINITION
    if first in ("list", "enumerate", "name") or re.match(r"(?:what|which) are the\b", lower):
        return QuestionType.LIST
    if "examples of" in lower or lower.startswith("give examples"):
        return QuestionType.LIST
    if re.match(r"how (?:many|much|long|often|old|far|big|large)\b", lower):
        return QuestionType.FACTOID
    if re.search(r"\bhow to\b|\bhow (?:do|can|should) (?:i|we|you)\b|\bsteps (?:to|for)\b", lower):
        return QuestionType.PROCEDURAL
    if first in ("why", "how", "explain", "describe") or lower.startswith("what causes"):
        return QuestionType.EXPLANATORY
    if first in AUX:
        return QuestionType.BOOLEAN
    if first in WH:
        return QuestionType.FACTOID
    if first in REQUEST:
        return QuestionType.EXPLANATORY
    return QuestionType.KEYWORD


def extract_features(normalized: str) -> QueryFeatures:
    lower = normalized.casefold()
    words = _WORD.findall(normalized)
    lower_words = [w.casefold() for w in words]
    bm25_terms = _unique(analyze(normalized))
    key_terms = [t for t in bm25_terms if t not in FUNCTION_WORDS]
    first = lower_words[0] if lower_words else None
    question_word = first if first in (*WH, *AUX, *REQUEST) else None
    comparison = _phrases(lower, COMPARISON)

    tokens = [t.strip(_STRIP) for t in normalized.split()]
    identifiers = _unique(t for t in tokens if t and is_identifier(t))
    numbers = _unique(t for t in tokens if t and _NUMBER.match(t))
    ident_words = {w for t in identifiers for w in _WORD.findall(t)}

    capitalized: list[str] = []  # runs of capitalised words; a sentence's first word is skipped
    run: list[str] = []
    for m in _WORD.finditer(normalized):
        w = m.group()
        start = re.search(r"(?:^|[.!?])\W*$", normalized[: m.start()]) is not None
        cap = w[:1].isupper() and not w.isupper() and w not in ident_words and w != "I"
        if cap and not start:
            run.append(w)
            continue
        if run:
            capitalized.append(" ".join(run))
        run = []
    if run:
        capitalized.append(" ".join(run))

    quoted = _unique(q.strip() for q in _QUOTED.findall(normalized) if q.strip())
    segments = [
        s for s in _SEGMENTS.split(lower) if any(t not in FUNCTION_WORDS for t in analyze(s))
    ]
    return QueryFeatures(
        char_count=len(normalized),
        token_count=len(words),
        bm25_terms=bm25_terms,
        key_terms=key_terms,
        function_word_ratio=round(sum(w in FUNCTION_WORDS for w in lower_words) / len(words), 4)
        if words
        else 0.0,
        is_question=normalized.endswith("?") or question_word is not None,
        question_word=question_word,
        question_type=_question_type(lower, lower_words, comparison),
        quoted_phrases=quoted,
        identifiers=identifiers,
        capitalized_terms=_unique(capitalized),
        numbers=numbers,
        entities=_unique([*quoted, *identifiers, *capitalized]),
        concept_segments=segments,
        comparison_markers=comparison,
        multi_hop_markers=_phrases(lower, MULTI_HOP),
        temporal_markers=_unique([*_YEAR.findall(normalized), *_phrases(lower, TEMPORAL)]),
        negation_markers=_unique([*_phrases(lower, NEGATION), *re.findall(r"\b\w+n't\b", lower)]),
        ambiguity_markers=_phrases(lower, VAGUE),
    )


def _signal(name: str, values: dict[str, float]) -> QuerySignal:
    weights: dict[str, float] = CONFIG["signals"][name]
    parts = [
        SignalContribution(
            feature=f, value=round(values[f], 4), weight=w, contribution=round(values[f] * w, 4)
        )
        for f, w in weights.items()
    ]
    score = round(min(1.0, max(0.0, sum(p.contribution for p in parts))), 4)
    return QuerySignal(name=name, score=score, contributions=parts)


def compute_signals(f: QueryFeatures) -> list[QuerySignal]:
    anchored = bool(f.identifiers or f.quoted_phrases)
    descriptive = {
        QuestionType.DEFINITION,
        QuestionType.PROCEDURAL,
        QuestionType.EXPLANATORY,
        QuestionType.COMPARISON,
        QuestionType.LIST,
    }
    lexical = {
        "identifiers": min(1.0, len(f.identifiers)),
        "quoted_phrases": min(1.0, len(f.quoted_phrases)),
        "numbers": min(1.0, len(f.numbers)),
        "keyword_form": float(not f.is_question),
        "term_density": 1 - f.function_word_ratio if f.token_count else 0.0,
    }
    semantic = {
        "question_form": float(f.is_question),
        "function_words": min(1.0, f.function_word_ratio / 0.5),
        "length": min(1.0, max(0, f.token_count - 3) / 9),
        "descriptive_need": float(f.question_type in descriptive),
        "no_exact_anchor": float(not anchored and f.token_count > 0),
    }
    complexity = {
        "key_terms": min(1.0, max(0, len(f.key_terms) - 2) / 8),
        "concepts": min(1.0, max(0, len(f.concept_segments) - 1) / 2),
        "multi_hop": min(1.0, len(f.multi_hop_markers) / 2),
        "comparison": float(bool(f.comparison_markers)),
        "constraints": min(1.0, (len(f.temporal_markers) + len(f.negation_markers)) / 2),
    }
    return [
        _signal("lexical", lexical),
        _signal("semantic", semantic),
        _signal("complexity", complexity),
    ]


def label(f: QueryFeatures, signals: list[QuerySignal]) -> QueryLabels:
    score = {s.name: s.score for s in signals}
    margin = round(score["lexical"] - score["semantic"], 4)
    threshold = float(CONFIG["class_margin"])
    query_class = (
        QueryClass.LEXICAL
        if margin >= threshold
        else QueryClass.SEMANTIC
        if margin <= -threshold
        else QueryClass.MIXED
    )
    bands = CONFIG["complexity"]
    complexity = (
        Complexity.COMPLEX
        if score["complexity"] >= bands["complex"]
        else Complexity.MODERATE
        if score["complexity"] >= bands["moderate"]
        else Complexity.SIMPLE
    )
    hops = len(f.multi_hop_markers)
    multi_hop = hops >= 2 or (hops >= 1 and len(f.concept_segments) >= 2)
    multiple = (
        multi_hop
        or f.question_type in (QuestionType.COMPARISON, QuestionType.LIST)
        or complexity is Complexity.COMPLEX
    )
    return QueryLabels(
        query_class=query_class,
        class_margin=margin,
        complexity=complexity,
        multi_hop_likely=multi_hop,
        # too little to anchor on: one key term or vague referents, and no entity to pin it down
        ambiguous=not f.entities and (len(f.key_terms) <= 1 or bool(f.ambiguity_markers)),
        evidence_need=EvidenceNeed.MULTIPLE_PASSAGES if multiple else EvidenceNeed.SINGLE_PASSAGE,
    )


class HeuristicQueryAnalyzer:
    name = "heuristic"
    version = "heuristic-query-analyzer@1"

    def config_hash(self) -> str:
        return canonical_hash({"version": self.version, **CONFIG})

    def analyze(self, query: str, corpus: CorpusQuerySignals | None = None) -> QueryAnalysis:
        normalized = normalize(query)
        features = extract_features(normalized)
        signals = compute_signals(features)
        analysis = QueryAnalysis(
            analyzer=self.name,
            analyzer_version=self.version,
            config_hash=self.config_hash(),
            query=query,
            normalized_query=normalized,
            features=features,
            signals=signals,
            labels=label(features, signals),
            corpus=corpus,
            analysis_hash="",
        )
        # the raw query is left out: queries that normalise identically are the same analysis
        digest = canonical_hash(
            analysis.model_dump(mode="json", exclude={"query", "analysis_hash"})
        )
        return analysis.model_copy(update={"analysis_hash": digest})
