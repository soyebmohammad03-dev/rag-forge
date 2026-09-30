"""Text analysis shared by lexical indexing and querying. Both sides must use the same analyzer,
so its identity is part of every lexical index row and every retriever config hash.
"""

from __future__ import annotations

import re
import unicodedata

# Bump the suffix whenever tokens can change for the same text.
ANALYZER = "nfkc-casefold-word-lucene33@1"

# Lucene's English stop set (EnglishAnalyzer.ENGLISH_STOP_WORDS_SET): small and well known.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)

_WORD = re.compile(r"\w+")


def analyze(text: str) -> list[str]:
    """NFKC-normalise, casefold, split on Unicode word characters, drop stopwords.

    No stemming: the baseline stays transparent and language-neutral beyond the stop set.
    """
    words = _WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    return [w for w in words if w not in STOPWORDS]
