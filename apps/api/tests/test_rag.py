"""Evidence selection, context assembly, claims, grounding and generators.

Retrieval is real BM25 over an ingested corpus, grounding uses the real pinned embedder, and the
generators are the real extractive baseline and a real HTTP client against a local stub server.
The model-backed end-to-end pipeline is in test_rag_pipeline.py.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from rag_forge.domain.models import (
    AnswerGrounding,
    ChatRole,
    ChunkingConfig,
    ClaimKind,
    Corpus,
    Evidence,
    EvidenceParams,
    FinishReason,
    GenerationParams,
    GeneratorSpec,
    RagRequest,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    SelectionOutcome,
    SupportStatus,
)
from rag_forge.ingestion.service import IngestionService
from rag_forge.rag.claims import extract_claims, strip_markers
from rag_forge.rag.evidence import (
    ContextBudgetError,
    build_context,
    evidence_id,
    select_evidence,
)
from rag_forge.rag.generation import (
    ExtractiveGenerator,
    GenerationInput,
    GeneratorUnavailableError,
    OnnxCausalLM,
    OpenAICompatibleGenerator,
    _sample,
    render_chatml,
)
from rag_forge.rag.grounding import LexicalSemanticVerifier, summarize
from rag_forge.rag.prompt import GROUNDED_QA, INSUFFICIENT
from rag_forge.rag.text import content_terms, sentence_spans
from rag_forge.retrieval.embedding import OnnxSentenceEmbedder
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.service import RetrievalService
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SqliteStore

DOCS = [
    ("photo.txt", b"Through photosynthesis, plants use sunlight, water and CO2 to produce food."),
    (
        "photo-copy.txt",
        b"Through photosynthesis plants use sunlight, water and CO2 to produce food!",
    ),
    (
        "plants.txt",
        b"Plants absorb water through their roots. Plants release oxygen as a by-product.",
    ),
    ("plants-more.txt", b"Plants store food as starch. Most plants grow toward sunlight."),
    ("trucks.txt", b"Food trucks need permits to make and sell food, and food safety checks."),
    ("model.txt", b"The model was trained on 30 billion tokens of web text in 2021."),
]
EXTRACTIVE = ExtractiveGenerator()
COUNT = EXTRACTIVE.count_tokens
TOKENIZER = EXTRACTIVE.tokenizer_id


@pytest.fixture
def corpus(store: SqliteStore, service: IngestionService) -> Corpus:
    c = store.add_corpus(
        Corpus(name="rag", chunking=ChunkingConfig(chunk_size=400, chunk_overlap=0))
    )
    service.ingest(c.id, DOCS)
    found = store.get_corpus(c.id)
    assert found is not None
    return found


@pytest.fixture
def retrieval(store: SqliteStore) -> RetrievalService:
    index = SqliteLexicalIndex(store)
    return RetrievalService(
        store, {RetrievalStrategy.SPARSE: lambda r: Bm25Retriever(index, r.bm25)}
    )


def ask(svc: RetrievalService, corpus: Corpus, query: str, **kw: Any) -> RetrievalResponse:
    return svc.retrieve(corpus.id, RetrievalRequest.model_validate({"query": query, **kw}))


def select(r: RetrievalResponse, **params: Any) -> list[Evidence]:
    return select_evidence(r, EvidenceParams(**params), COUNT, TOKENIZER).selected


# --- evidence identity and selection ----------------------------------------------------


def test_evidence_keeps_exact_identity_and_provenance(
    retrieval: RetrievalService, corpus: Corpus, store: SqliteStore
) -> None:
    r = ask(retrieval, corpus, "plants sunlight food", top_k=6)
    sel = select_evidence(r, EvidenceParams(near_duplicate_threshold=None), COUNT, TOKENIZER)
    assert sel.selected and sel.candidates == len(r.hits) == len(sel.decisions)
    for i, (e, hit) in enumerate(zip(sel.selected, r.hits, strict=False), 1):
        assert e.citation == f"E{i}" and e.selection_rank == i
        assert (e.corpus_id, e.corpus_version) == (corpus.id, corpus.version)
        assert e.chunking_hash == corpus.chunking.config_hash()
        assert e.chunk_id == hit.chunk.id and e.text == hit.chunk.text  # exact text, no copies
        assert (e.char_start, e.char_end) == (hit.chunk.char_start, hit.chunk.char_end)
        assert e.document_version_id == hit.result.document_version_id
        assert e.retrieval.rank == hit.result.rank and e.retrieval.score == hit.result.score
        assert e.retrieval.configuration_hash == r.provenance.configuration_hash
        assert e.retrieval.upstream_rank is None  # not reranked
        assert e.origin == "retrieved" and e.id.startswith("evd_")
        stored = store.get_chunks([e.chunk_id])[e.chunk_id]
        assert stored.text == e.text
    assert sel.tokens_used == sum(e.token_count for e in sel.selected)


def test_evidence_ids_are_stable_and_version_scoped(
    retrieval: RetrievalService, corpus: Corpus, service: IngestionService
) -> None:
    v1 = select(ask(retrieval, corpus, "food trucks permits"))
    again = select(ask(retrieval, corpus, "food trucks permits"))
    assert [e.id for e in v1] == [e.id for e in again]  # same corpus version, same ids
    service.ingest(corpus.id, [("extra.txt", b"Unrelated text about rivers and mountains.")])
    v2 = select(ask(retrieval, corpus, "food trucks permits"))
    pinned = select(ask(retrieval, corpus, "food trucks permits", version=corpus.version))
    assert v2[0].corpus_version == corpus.version + 1
    assert v2[0].chunk_id == v1[0].chunk_id  # the same chunk...
    assert v2[0].id != v1[0].id  # ...is different evidence in a different corpus version
    assert [e.id for e in pinned] == [e.id for e in v1]
    with pytest.raises(ContextBudgetError, match="several corpus versions"):
        build_context("q", [v1[0], v2[0]], GROUNDED_QA, 1000, COUNT, TOKENIZER)
    assert evidence_id("c", 1, "k", 0, 5, "s") == evidence_id("c", 1, "k", 0, 5, "s")
    assert evidence_id("c", 1, "k", 0, 5, "s") != evidence_id("c", 1, "k", 0, 6, "s")


def test_selection_is_deterministic(retrieval: RetrievalService, corpus: Corpus) -> None:
    r = ask(retrieval, corpus, "plants water sunlight food", top_k=6)
    params = EvidenceParams(max_items=3, max_per_document=1)
    runs = [select_evidence(r, params, COUNT, TOKENIZER) for _ in range(3)]
    assert len({s.selection_hash for s in runs}) == 1
    assert all(s.selected == runs[0].selected and s.decisions == runs[0].decisions for s in runs)
    other = select_evidence(r, params.model_copy(update={"max_items": 2}), COUNT, TOKENIZER)
    assert other.selection_hash != runs[0].selection_hash


def test_near_duplicates_and_document_cap(retrieval: RetrievalService, corpus: Corpus) -> None:
    r = ask(retrieval, corpus, "photosynthesis plants sunlight water CO2 food", top_k=6)
    names = [h.filename for h in r.hits]
    assert names[:2] == ["photo.txt", "photo-copy.txt"] or names[:2] == [
        "photo-copy.txt",
        "photo.txt",
    ]
    sel = select_evidence(r, EvidenceParams(), COUNT, TOKENIZER)
    dup = next(d for d in sel.decisions if d.outcome is SelectionOutcome.NEAR_DUPLICATE)
    assert dup.rank == 2 and dup.similar_to == sel.selected[0].id
    assert dup.similarity is not None and dup.similarity >= 0.8
    assert dup.chunk_id not in {e.chunk_id for e in sel.selected}
    keep_all = select(r, near_duplicate_threshold=None, max_items=6)
    assert {e.filename for e in keep_all} >= {"photo.txt", "photo-copy.txt"}

    # chunk_size 400 puts each file in one chunk, so cap documents via a multi-chunk corpus
    capped = select_evidence(
        r, EvidenceParams(max_per_document=1, near_duplicate_threshold=None), COUNT, TOKENIZER
    )
    docs = [e.document_version_id for e in capped.selected]
    assert len(docs) == len(set(docs))


def test_document_cap_limits_chunks_per_document(
    store: SqliteStore, service: IngestionService, retrieval: RetrievalService
) -> None:
    c = store.add_corpus(
        Corpus(name="long", chunking=ChunkingConfig(chunk_size=60, chunk_overlap=0))
    )
    long = " ".join(f"Plants need water and sunlight in season {i}." for i in range(8))
    service.ingest(c.id, [("long.txt", long.encode()), ("short.txt", b"Plants need water too.")])
    corpus = store.get_corpus(c.id)
    assert corpus is not None
    r = ask(retrieval, corpus, "plants water sunlight", top_k=10)
    sel = select_evidence(
        r, EvidenceParams(max_per_document=2, near_duplicate_threshold=None), COUNT, TOKENIZER
    )
    per_doc = [e.filename for e in sel.selected]
    assert per_doc.count("long.txt") == 2 and "short.txt" in per_doc
    capped = [d for d in sel.decisions if d.outcome is SelectionOutcome.DOCUMENT_CAP]
    assert capped and all(d.filename == "long.txt" for d in capped)


def test_item_token_and_score_budgets(retrieval: RetrievalService, corpus: Corpus) -> None:
    r = ask(retrieval, corpus, "plants water sunlight food", top_k=6)
    base = {"near_duplicate_threshold": None, "max_per_document": None}
    sel = select_evidence(r, EvidenceParams(max_items=2, **base), COUNT, TOKENIZER)
    assert len(sel.selected) == 2
    assert {d.outcome for d in sel.decisions[2:]} == {SelectionOutcome.OVER_ITEM_BUDGET}

    first = COUNT(f"[E1] ({r.hits[0].filename})\n{r.hits[0].chunk.text}")
    tight = select_evidence(
        r, EvidenceParams(max_context_tokens=max(first, 64), **base), COUNT, TOKENIZER
    )
    assert tight.tokens_used <= max(first, 64)
    assert any(d.outcome is SelectionOutcome.OVER_TOKEN_BUDGET for d in tight.decisions)

    floor = r.hits[1].result.score
    scored = select_evidence(r, EvidenceParams(min_score=floor, **base), COUNT, TOKENIZER)
    assert all(e.retrieval.score >= floor for e in scored.selected)
    below = [d for d in scored.decisions if d.outcome is SelectionOutcome.BELOW_MIN_SCORE]
    assert below and all(d.score < floor for d in below)
    assert all(d.detail for d in scored.decisions)  # every decision states why


# --- context ---------------------------------------------------------------------------


def test_context_contains_only_selected_evidence_and_hashes_deterministically(
    retrieval: RetrievalService, corpus: Corpus
) -> None:
    r = ask(retrieval, corpus, "plants water sunlight food", top_k=6)
    ev = select(r, max_items=2)
    ctx = build_context("What do plants need?", ev, GROUNDED_QA, 1500, COUNT, TOKENIZER)
    assert [b.evidence_id for b in ctx.blocks] == [e.id for e in ev]
    assert ctx.evidence_text == "\n\n".join(f"[{e.citation}] ({e.filename})\n{e.text}" for e in ev)
    unselected = [h.chunk.text for h in r.hits if h.chunk.id not in {e.chunk_id for e in ev}]
    assert unselected and not any(t in ctx.evidence_text for t in unselected)
    assert (ctx.corpus_id, ctx.corpus_version) == (corpus.id, corpus.version)
    assert ctx.prompt_template == "grounded-qa@1" and ctx.context_tokens <= 1500
    assert [m.role for m in ctx.messages] == [ChatRole.SYSTEM, ChatRole.USER]
    assert ctx.evidence_text in ctx.messages[1].content
    assert "Question: What do plants need?" in ctx.messages[1].content
    assert "only cite ids that appear in the passages" in ctx.messages[0].content
    assert INSUFFICIENT in ctx.messages[0].content

    same = build_context("What do plants need?", ev, GROUNDED_QA, 1500, COUNT, TOKENIZER)
    assert (same.context_hash, same.prompt_hash) == (ctx.context_hash, ctx.prompt_hash)
    other_q = build_context("Why?", ev, GROUNDED_QA, 1500, COUNT, TOKENIZER)
    assert other_q.context_hash == ctx.context_hash  # same evidence...
    assert other_q.prompt_hash != ctx.prompt_hash  # ...different prompt
    edited = [ev[0].model_copy(update={"text": ev[0].text + " Extra."}), *ev[1:]]
    assert build_context("q", edited, GROUNDED_QA, 1500, COUNT, TOKENIZER).context_hash != (
        ctx.context_hash
    )
    with pytest.raises(ContextBudgetError, match="over max_context_tokens"):
        build_context("q", ev, GROUNDED_QA, 10, COUNT, TOKENIZER)
    with pytest.raises(ContextBudgetError, match="without evidence"):
        build_context("q", [], GROUNDED_QA, 1500, COUNT, TOKENIZER)


def test_chatml_rendering() -> None:
    msgs = GROUNDED_QA.render("Q?", "[E1] (a.txt)\ntext")
    out = render_chatml(msgs)
    assert out.startswith("<|im_start|>system\n") and out.endswith("<|im_start|>assistant\n")
    assert out.count("<|im_end|>") == 2


# --- claims ----------------------------------------------------------------------------


def _ev(retrieval: RetrievalService, corpus: Corpus, query: str, **p: Any) -> list[Evidence]:
    return select(ask(retrieval, corpus, query, top_k=6), **p)


def test_sentence_spans_are_exact() -> None:
    text = 'One fact. "Two" facts! Three?\n- Four [E1]\n\n  Five.'
    spans = sentence_spans(text)
    parts = [text[a:b] for a, b in spans]
    assert parts == ["One fact.", '"Two" facts!', "Three?", "- Four [E1]", "Five."]
    assert sentence_spans("3.5 billion is e.g. large") == [(0, 25)]


def test_claim_extraction_parses_citations(retrieval: RetrievalService, corpus: Corpus) -> None:
    ev = _ev(retrieval, corpus, "plants water sunlight food", max_items=2)
    answer = (
        "Plants use sunlight to produce food [E1]. They absorb water [E2, e1]. [E2] "
        "Food trucks need permits [E9].\n- The evidence is insufficient to say more.\nHere:"
    )
    claims = extract_claims(answer, ev, "h")
    assert [c.kind for c in claims] == [
        ClaimKind.FACTUAL,
        ClaimKind.FACTUAL,
        ClaimKind.FACTUAL,
        ClaimKind.ABSTENTION,
        ClaimKind.NON_ASSERTIVE,
    ]
    first, second, third = claims[:3]
    assert first.text == "Plants use sunlight to produce food." and first.raw_text.endswith("[E1].")
    assert [c.label for c in first.citations] == ["E1"]
    assert [c.label for c in second.citations] == ["E2", "E1", "E2"]  # trailing [E2] attaches
    assert all(c.valid and c.evidence_id for c in second.citations)
    assert third.text == "Food trucks need permits." and not third.citations[0].valid
    assert third.citations[0].evidence_id is None  # never remapped to a real passage
    assert claims[3].text == "The evidence is insufficient to say more."  # list marker stripped
    for c in claims:
        assert answer[c.char_start : c.char_end] == c.raw_text
        for cit in c.citations:
            assert answer[cit.char_start : cit.char_end].upper().startswith("[E")
    again = extract_claims(answer, ev, "h")
    assert [c.id for c in again] == [c.id for c in claims]
    assert {c.id for c in extract_claims(answer, ev, "other")}.isdisjoint({c.id for c in claims})
    assert strip_markers("A [E1] , b [E2].") == "A, b."
    assert extract_claims("", ev, "h") == []


@pytest.mark.parametrize(
    "text",
    [
        "The evidence is insufficient.",
        "INSUFFICIENT_EVIDENCE",
        "The passages do not mention the year.",
        "There is not enough information to answer.",
    ],
)
def test_abstentions_are_recognised(text: str) -> None:
    assert extract_claims(text, [], "h")[0].kind is ClaimKind.ABSTENTION


# --- grounding (real embedder) -----------------------------------------------------------


@pytest.fixture(scope="module")
def verifier(embedder: OnnxSentenceEmbedder) -> LexicalSemanticVerifier:
    return LexicalSemanticVerifier(embedder)


def ground(
    verifier: LexicalSemanticVerifier, answer: str, ev: list[Evidence]
) -> tuple[list[Any], Any]:
    claims = verifier.verify(extract_claims(answer, ev, "h"), ev)
    return claims, summarize(claims, ev, verifier, 0.0)


def test_grounding_classifies_claims(
    retrieval: RetrievalService, corpus: Corpus, verifier: LexicalSemanticVerifier
) -> None:
    ev = _ev(retrieval, corpus, "plants photosynthesis water sunlight trained", max_items=5)
    by_file = {e.filename: e for e in ev}
    photo = by_file.get("photo.txt") or by_file["photo-copy.txt"]
    answer = (
        f"Plants use sunlight, water and CO2 to produce food [{photo.citation}]. "
        "The capital of France is Paris. "
        "Plants absorb water and nutrients through their leaves."
    )
    claims, report = ground(verifier, answer, ev)
    verbatim, paris, partial = claims
    assert verbatim.support is SupportStatus.SUPPORTED
    assert verbatim.supporting_evidence_ids[0] == photo.id
    assert verbatim.cited_evidence_ids == [photo.id] and "uncited" not in verbatim.flags
    assert paris.support is SupportStatus.UNSUPPORTED and paris.supporting_evidence_ids == []
    assert set(paris.missing_terms) == {"capital", "france", "paris"} and "uncited" in paris.flags
    assert partial.support is SupportStatus.WEAKLY_SUPPORTED  # "leaves" and "nutrients" absent
    assert {"nutrients", "leaves"} <= set(partial.missing_terms)
    for c in claims:
        assert [es.evidence_id for es in c.evidence] and len(c.evidence) == len(ev)
        assert c.support_score is not None and 0 <= c.support_score <= 1
        assert c.origin == "generated" and c.rationale
    assert report.status is AnswerGrounding.PARTIALLY_GROUNDED
    assert (report.factual_claims, report.supported, report.weakly_supported) == (3, 1, 1)
    assert report.unsupported == 1 and report.contradicted == 0
    assert report.grounding_score == round(1.5 / 3, 4)
    assert report.citation_coverage == round(1 / 3, 4) and report.citation_precision == 1.0
    assert not report.detects_contradiction and report.origin == "measured"
    assert verifier.prepare() < 50  # already loaded: model load is never billed to scoring


def test_numbers_negation_and_never_contradicted(
    retrieval: RetrievalService, corpus: Corpus, verifier: LexicalSemanticVerifier
) -> None:
    ev = _ev(retrieval, corpus, "model trained billion tokens web text plants water")
    claims, report = ground(
        verifier,
        "The model was trained on 300 billion tokens of web text. "
        "The model was trained on 30 billion tokens of web text. "
        "Plants do not absorb water through their roots.",
        ev,
    )
    wrong_number, right_number, negated = claims
    assert wrong_number.unmatched_numbers == ["300"] and "number_mismatch" in wrong_number.flags
    assert wrong_number.support is not SupportStatus.SUPPORTED
    assert right_number.support is SupportStatus.SUPPORTED and not right_number.unmatched_numbers
    assert "negation_mismatch" in negated.flags
    assert negated.support is not SupportStatus.SUPPORTED
    assert all(c.support is not SupportStatus.CONTRADICTED for c in claims)
    assert report.contradicted == 0


def test_multi_passage_support_and_citation_metrics(
    retrieval: RetrievalService, corpus: Corpus, verifier: LexicalSemanticVerifier
) -> None:
    ev = _ev(retrieval, corpus, "plants food starch oxygen", max_items=5)
    by_file = {e.filename: e for e in ev}
    a, b, trucks = by_file["plants.txt"], by_file["plants-more.txt"], by_file["trucks.txt"]
    # no single passage states both facts; together they do
    answer = f"Plants release oxygen and store food as starch [{a.citation}][{b.citation}]."
    (claim,), report = ground(verifier, answer, ev)
    assert claim.support is SupportStatus.SUPPORTED and "multi_passage" in claim.flags
    assert {a.id, b.id} <= set(claim.supporting_evidence_ids)
    assert trucks.id not in claim.supporting_evidence_ids
    single = {es.evidence_id: es.status for es in claim.evidence}
    assert SupportStatus.SUPPORTED not in (single[a.id], single[b.id])
    assert report.citation_coverage == 1.0 and report.citation_precision == 1.0
    assert report.evidence_coverage == round(len(claim.supporting_evidence_ids) / len(ev), 4)

    # a true claim cited to the wrong passage: supported, but the citation is measured as wrong
    wrong = f"Plants release oxygen as a by-product [{trucks.citation}]."
    (w,), r2 = ground(verifier, wrong, ev)
    assert w.support is SupportStatus.SUPPORTED and w.supporting_evidence_ids == [a.id]
    assert "cited_not_supporting" in w.flags and r2.citation_precision == 0.0


def test_abstention_and_empty_answers_are_not_scored(
    retrieval: RetrievalService, corpus: Corpus, verifier: LexicalSemanticVerifier
) -> None:
    ev = _ev(retrieval, corpus, "plants")
    claims, report = ground(verifier, "The evidence is insufficient.", ev)
    assert claims[0].support is SupportStatus.NOT_APPLICABLE and claims[0].support_score is None
    assert report.status is AnswerGrounding.ABSTAINED and report.grounding_score is None
    assert report.citation_coverage is None and report.abstentions == 1
    _, empty = ground(verifier, "", ev)
    assert empty.status is AnswerGrounding.NO_CLAIMS and empty.claims == 0


def test_grounding_fails_loudly_without_its_model(
    retrieval: RetrievalService, corpus: Corpus
) -> None:
    from rag_forge.domain.models import EmbedderSpec
    from rag_forge.rag.grounding import GroundingUnavailableError

    broken = OnnxSentenceEmbedder(EmbedderSpec(model="rag-forge/missing", revision="0" * 40))
    ev = _ev(retrieval, corpus, "plants")
    with pytest.raises(GroundingUnavailableError, match="grounding embedder unavailable"):
        LexicalSemanticVerifier(broken).verify(extract_claims("Plants grow.", ev, "h"), ev)


# --- generators --------------------------------------------------------------------------


def test_generation_params_validation_and_effective_hashing() -> None:
    for bad in ({"temperature": -0.1}, {"temperature": 2.5}, {"max_new_tokens": 0}, {"top_p": 0}):
        with pytest.raises(ValidationError):
            GenerationParams.model_validate(bad)
    greedy = GenerationParams(temperature=0, top_p=0.5, seed=7).effective()
    assert (greedy.top_p, greedy.seed) == (1.0, 0)  # ignored by greedy decoding, so not hashed
    sampled = GenerationParams(temperature=0.7, top_p=0.5, seed=7).effective()
    assert (sampled.top_p, sampled.seed) == (0.5, 7)
    with pytest.raises(ValidationError):
        EvidenceParams(max_items=0)
    with pytest.raises(ValidationError):
        RagRequest.model_validate({"retrieval": {"query": " "}})
    with pytest.raises(ValueError, match="chat template"):
        OnnxCausalLM(GeneratorSpec(chat_template="llama3"))


def test_sampling_is_reproducible_with_a_seed() -> None:
    logits = np.log(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    p = GenerationParams(temperature=1.0, top_p=0.9, seed=3)
    draws = [_sample(logits, p, np.random.default_rng(3)) for _ in range(5)]
    assert len(set(draws)) == 1
    many = np.random.default_rng(0)
    seen = {_sample(logits, GenerationParams(temperature=1.0, top_p=0.5), many) for _ in range(50)}
    assert seen <= {2, 3}  # nucleus 0.5 keeps the two most likely tokens
    assert _sample(logits, GenerationParams(), np.random.default_rng(0)) == 3


def test_extractive_baseline_is_verbatim_cited_and_deterministic(
    retrieval: RetrievalService, corpus: Corpus
) -> None:
    ev = _ev(retrieval, corpus, "plants water sunlight", max_items=3)
    ctx = build_context("What do plants need?", ev, GROUNDED_QA, 1500, COUNT, TOKENIZER)
    inp = GenerationInput(ctx.messages, "What do plants need?", ctx.blocks, GenerationParams())
    out = EXTRACTIVE.generate(inp)
    assert out.finish_reason is FinishReason.STOP and out.text
    assert all(EXTRACTIVE.generate(inp).text == out.text for _ in range(3))
    for line in out.text.splitlines():
        sentence, citation = line.rsplit(" [", 1)
        block = next(b for b in ctx.blocks if f"{b.citation}]" == citation)
        assert sentence in block.text  # verbatim evidence
    none = GenerationInput(ctx.messages, "capital of France?", ctx.blocks, GenerationParams())
    assert EXTRACTIVE.generate(none).text == INSUFFICIENT
    cut = EXTRACTIVE.generate(
        GenerationInput(ctx.messages, "plants", ctx.blocks, GenerationParams(max_new_tokens=3))
    )
    assert cut.finish_reason is FinishReason.LENGTH and cut.completion_tokens == 3
    assert EXTRACTIVE.describe().loaded and EXTRACTIVE.info().deterministic_at_zero_temperature


@pytest.fixture
def chat_server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(
                {"path": self.path, "body": body, "auth": self.headers["Authorization"]}
            )
            reply = {
                "choices": [
                    {"message": {"content": " Plants need water [E1]. "}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7},
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", received
    server.shutdown()


def test_openai_compatible_generator(chat_server: tuple[str, list[dict[str, Any]]]) -> None:
    url, received = chat_server
    gen = OpenAICompatibleGenerator(url, "tiny", api_key="test-key")
    msgs = GROUNDED_QA.render("Q?", "[E1] (a.txt)\nPlants need water.")
    params = GenerationParams(temperature=0.2, seed=5, max_new_tokens=30)
    out = gen.generate(GenerationInput(msgs, "Q?", [], params))
    assert out.text == "Plants need water [E1]." and out.finish_reason is FinishReason.STOP
    assert (out.prompt_tokens, out.completion_tokens) == (42, 7)
    (call,) = received
    assert call["path"] == "/v1/chat/completions" and call["auth"] == "Bearer test-key"
    assert call["body"]["messages"][0]["role"] == "system"
    assert (call["body"]["max_tokens"], call["body"]["seed"]) == (30, 5)
    info = gen.info()
    assert not info.local and not info.deterministic_at_zero_temperature
    assert "test-key" not in info.model_dump_json()  # the key is never recorded
    down = OpenAICompatibleGenerator("http://127.0.0.1:9/v1", "tiny", timeout_s=2)
    with pytest.raises(GeneratorUnavailableError, match="openai-compatible"):
        down.generate(GenerationInput(msgs, "Q?", [], params))


def test_local_generator_unavailable_is_explicit() -> None:
    gen = OnnxCausalLM(GeneratorSpec(model="rag-forge/missing-model", revision="0" * 40))
    assert not gen.describe().loaded  # describing never loads
    with pytest.raises(GeneratorUnavailableError, match="unavailable"):
        gen.info()


def test_content_terms_drop_function_words() -> None:
    assert content_terms("What do the plants need to make food?") == [
        "plants",
        "need",
        "make",
        "food",
    ]
