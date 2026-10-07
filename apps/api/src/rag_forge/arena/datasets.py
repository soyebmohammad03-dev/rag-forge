"""Benchmark datasets: validation against a corpus version, and the bundled development set.

Relevance is annotated by filename (document level) or chunk id (chunk level) and resolved
against the pinned corpus version, so an annotation can never silently point at a different
version of a document.

`DEV_DOCUMENTS` and `DEV_CASES` form the development benchmark. Each document was written for
this set, extending the passages the test suite already uses, so its relevance judgements hold
by construction. A case is annotated only with what is known: the comparison case has no
reference answer, and the unanswerable case has no relevant documents. With 14 cases it
validates the benchmarking pipeline. It is not evidence that one method beats another.
"""

from __future__ import annotations

from rag_forge.domain.arena import (
    Annotation,
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDatasetCreate,
    DatasetSource,
)
from rag_forge.domain.models import Corpus
from rag_forge.storage.base import CorpusStore

DEV_NAME = "rag-forge-dev"
DEV_CORPUS = "arena-dev-corpus"

DEV_DOCUMENTS: list[tuple[str, str]] = [
    (
        "photosynthesis.txt",
        "Through photosynthesis, plants use sunlight, water and carbon dioxide to produce glucose "
        "and release oxygen. Chlorophyll in the leaves absorbs mostly red and blue light.",
    ),
    (
        "food-trucks.txt",
        "Food trucks need a mobile vending permit, a food handler certificate and regular health "
        "inspections before they may sell food to the public.",
    ),
    (
        "bread.txt",
        "To bake bread you need flour, water, yeast and salt. The dough rises because yeast "
        "ferments sugars and releases carbon dioxide gas.",
    ),
    (
        "car-maintenance.txt",
        "Automobiles need regular oil changes and tyre rotations. Most manufacturers recommend "
        "changing the engine oil every 10,000 kilometres.",
    ),
    (
        "interest-rates.txt",
        "Equity prices fell sharply after the central bank raised its policy interest rate by "
        "0.75 percentage points in June 2022.",
    ),
    (
        "bm25.txt",
        "BM25 is a sparse lexical ranking function. It rewards exact term matches, saturates term "
        "frequency with the parameter k1 and normalises for document length with the parameter b.",
    ),
    (
        "dense-retrieval.txt",
        "Dense retrieval encodes queries and passages as embedding vectors and ranks passages by "
        "cosine similarity, so it can match paraphrases that share no words with the query.",
    ),
    (
        "rrf.txt",
        "Reciprocal rank fusion combines several ranked lists by summing 1 / (k + rank) for each "
        "document, with k usually set to 60. It uses ranks only, never raw scores.",
    ),
    (
        "cross-encoder.txt",
        "A cross-encoder reranker reads the query and a candidate passage together and outputs a "
        "relevance score. It is slower than a bi-encoder, so it only rescores a small candidate "
        "pool.",
    ),
    (
        "tomato-sauce.txt",
        "Simmer the tomatoes with garlic and basil for twenty minutes, then season the sauce with "
        "salt and olive oil.",
    ),
    (
        "error-codes.txt",
        "Error E4021 means the upload exceeded the 25 MB limit. Error E4033 means the file type "
        "is not supported.",
    ),
    (
        "volcano.txt",
        "Mount Vesuvius erupted in 79 AD and buried the Roman towns of Pompeii and Herculaneum "
        "under volcanic ash.",
    ),
]


def _case(
    id: str,
    query: str,
    relevant: dict[str, float],
    reference: str | None,
    tags: list[str],
    answerable: bool = True,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=id,
        query=query,
        relevant_documents=relevant,
        reference_answer=reference,
        answerable=answerable,
        expected_evidence=[f for f, g in relevant.items() if g > 0],
        tags=tags,
    )


DEV_CASES: list[BenchmarkCase] = [
    _case(
        "photosynthesis-inputs",
        "What do plants need to make food?",
        {"photosynthesis.txt": 2},
        "sunlight, water and carbon dioxide",
        ["semantic", "factoid"],
    ),
    _case(
        "food-truck-permits",
        "Which permits does a food truck need?",
        {"food-trucks.txt": 2},
        "a mobile vending permit and a food handler certificate",
        ["lexical", "list"],
    ),
    _case(
        "error-e4021",
        "What does error E4021 mean?",
        {"error-codes.txt": 2},
        "the upload exceeded the 25 MB limit",
        ["lexical", "identifier"],
    ),
    _case(
        "bread-rising",
        "Why does bread dough rise?",
        {"bread.txt": 2},
        "yeast ferments sugars and releases carbon dioxide gas",
        ["explanatory"],
    ),
    _case(
        "oil-interval",
        "How often should engine oil be changed?",
        {"car-maintenance.txt": 2},
        "every 10,000 kilometres",
        ["factoid", "number"],
    ),
    _case(
        "rate-hike-stocks",
        "What happened to stock prices when the central bank raised rates?",
        {"interest-rates.txt": 2},
        "equity prices fell sharply",
        ["semantic", "paraphrase"],
    ),
    _case(
        "bm25-length",
        "How does BM25 normalise for document length?",
        {"bm25.txt": 2},
        "with the parameter b",
        ["lexical", "technical"],
    ),
    _case(
        "paraphrase-method",
        "Which retrieval method can match paraphrases without shared words?",
        {"dense-retrieval.txt": 2},
        "dense retrieval",
        ["semantic", "technical"],
    ),
    _case(
        "rrf-constant",
        "What constant is usually used in reciprocal rank fusion?",
        {"rrf.txt": 2},
        "60",
        ["factoid", "number", "technical"],
    ),
    _case(
        "cross-encoder-pool",
        "Why does a cross-encoder only rescore a small candidate pool?",
        {"cross-encoder.txt": 2},
        "it is slower than a bi-encoder",
        ["explanatory", "technical"],
    ),
    _case(
        "sparse-vs-dense",
        "Compare sparse and dense retrieval.",
        {"bm25.txt": 1, "dense-retrieval.txt": 1},
        None,  # a comparison has no single reference answer
        ["comparison", "multi_passage", "technical"],
    ),
    _case(
        "vesuvius-towns",
        "Which towns were buried when Vesuvius erupted?",
        {"volcano.txt": 2},
        "Pompeii and Herculaneum",
        ["factoid", "entity"],
    ),
    _case(
        "yeast-gas",
        "What gas does yeast release?",
        {"bread.txt": 2},
        "carbon dioxide",
        ["factoid"],
    ),
    _case(
        "world-cup-2018",
        "Who won the 2018 football world cup?",
        {},
        None,
        ["unanswerable"],
        answerable=False,
    ),
]

DEV_DESCRIPTION = (
    "Development benchmark bundled with RAG FORGE: 12 short documents and 14 cases covering "
    "lexical, semantic, numeric, comparison and unanswerable queries. It validates the "
    "benchmarking pipeline; it is too small to support claims about methods."
)
DEV_NOTES = (
    "Every document was written for this set, so document-level relevance holds by construction "
    "(grade 2 = answers the query, 1 = needed for a comparison). Reference answers are short "
    "spans copied from the relevant document. The comparison case has no reference answer, and "
    "the unanswerable case has no relevant documents."
)


class DatasetValidationError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems[:10]) + (" …" if len(problems) > 10 else ""))


def validate_against_corpus(
    store: CorpusStore, corpus: Corpus, version: int, cases: list[BenchmarkCase]
) -> None:
    """Every annotated filename and chunk must exist in the pinned corpus version."""
    if version < 1 or version > corpus.version:
        raise DatasetValidationError([f"corpus {corpus.id} has no version {version}"])
    members = store.members(corpus.id, version)
    filenames = {dv.filename for dv in members.values()}
    member_ids = {dv.id for dv in members.values()}
    chunk_ids = {cid for c in cases for cid in c.relevant_chunks}
    chunks = store.get_chunks(sorted(chunk_ids)) if chunk_ids else {}
    chash = corpus.chunking.config_hash()
    problems = []
    for case in cases:
        for f in sorted({*case.relevant_documents, *case.expected_evidence}):
            if f not in filenames:
                problems.append(f"{case.id}: {f!r} is not a document in v{version}")
        for cid in sorted(case.relevant_chunks):
            chunk = chunks.get(cid)
            if chunk is None or chunk.document_version_id not in member_ids:
                problems.append(f"{case.id}: chunk {cid!r} is not in v{version}")
            elif chunk.chunking_hash != chash:
                problems.append(f"{case.id}: chunk {cid!r} uses another chunking configuration")
    if problems:
        raise DatasetValidationError(problems)


def build_dataset(
    spec: BenchmarkDatasetCreate,
    corpus: Corpus,
    version: int,
    source: DatasetSource,
    previous: list[BenchmarkDataset],
) -> BenchmarkDataset:
    """A new version of a dataset name, or the existing one if its content is unchanged."""
    content_hash = BenchmarkDataset.hash_content(corpus.id, version, spec.cases)
    for ds in previous:
        if ds.content_hash == content_hash and ds.source is source:
            return ds
    counts = {a: sum(a in c.annotations() for c in spec.cases) for a in Annotation}
    return BenchmarkDataset(
        name=spec.name,
        version=max((d.version for d in previous), default=0) + 1,
        source=source,
        description=spec.description,
        annotation_notes=spec.annotation_notes,
        corpus_id=corpus.id,
        corpus_version=version,
        chunking_hash=corpus.chunking.config_hash(),
        cases=spec.cases,
        annotation_counts=counts,
        content_hash=content_hash,
    )
