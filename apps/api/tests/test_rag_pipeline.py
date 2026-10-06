"""The answer pipeline end to end, with the real pinned models and nothing mocked:

query -> query intelligence -> router -> retrieval -> cross-encoder reranking -> evidence
selection -> context -> local ONNX generation -> claims -> grounding (real embedder).
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_forge.domain.models import (
    AnswerStatus,
    ClaimKind,
    EmbedderInfo,
    EmbedderSpec,
    GeneratorSpec,
    RagResponse,
    SupportStatus,
)
from rag_forge.main import create_app
from rag_forge.rag.generation import (
    ExtractiveGenerator,
    GenerationInput,
    GenerationOutput,
    OnnxCausalLM,
)
from rag_forge.retrieval.embedding import EmbedderUnavailableError, OnnxSentenceEmbedder, Vectors
from rag_forge.retrieval.rerank import OnnxCrossEncoder

DOCS = [
    ("trucks.txt", b"Food trucks need permits to make and sell food, and food safety checks."),
    ("recipes.txt", b"To make bread you need flour, water, yeast and salt."),
    ("photo.txt", b"Through photosynthesis, plants use sunlight, water and CO2 to produce food."),
    ("cars.txt", b"Automobiles need regular oil changes and tyre rotations to stay reliable."),
    ("markets.txt", b"Equity prices fell sharply after the central bank raised interest rates."),
]
QUERY = "what do plants need to make food?"
CHAIN = [
    "query",
    "query_analysis",
    "router_decision",
    "retrieval",
    "reranking",
    "evidence_selection",
    "context",
    "generation",
    "claims",
    "grounding",
]


@pytest.fixture(scope="module")
def lab(
    tmp_path_factory: pytest.TempPathFactory,
    embedder: OnnxSentenceEmbedder,
    cross_encoder: OnnxCrossEncoder,
    generator: OnnxCausalLM,
) -> Iterator[tuple[TestClient, str]]:
    app = create_app(tmp_path_factory.mktemp("rag"), embedder, cross_encoder, generator)
    with TestClient(app) as c:
        cid = c.post("/api/v1/corpora", json={"name": "rag"}).json()["corpus"]["id"]
        c.post(f"/api/v1/corpora/{cid}/documents", files=[("files", d) for d in DOCS])
        assert c.post(f"/api/v1/corpora/{cid}/dense-index", json={}).status_code == 202
        yield c, cid


def answer(c: TestClient, cid: str, query: str = QUERY, **body: Any) -> RagResponse:
    body.setdefault("retrieval", {})
    body["retrieval"] = {"query": query, "top_k": 4, **body["retrieval"]}
    r = c.post(f"/api/v1/corpora/{cid}/answer", json=body)
    assert r.status_code == 200, r.text
    return RagResponse.model_validate(r.json())


def test_end_to_end_adaptive_grounded_answer(lab: tuple[TestClient, str]) -> None:
    c, cid = lab
    r = answer(c, cid, retrieval={"mode": "adaptive"})

    # query intelligence and routing ran, and chose a reranked configuration
    routing = r.retrieval.provenance.routing
    assert routing is not None and routing.decision.rerank.enabled
    assert r.retrieval.reranking is not None and r.retrieval.hits[0].rerank is not None

    # evidence: drawn from the final (reranked) ranking, pinned to the corpus version
    sel = r.evidence
    assert sel.selected and sel.candidates == len(r.retrieval.hits)
    assert [e.chunk_id for e in sel.selected] == [
        d.chunk_id for d in sel.decisions if d.outcome == "selected"
    ]
    top = sel.selected[0]
    assert top.filename == "photo.txt" and top.retrieval.reranker_score is not None
    assert top.corpus_version == r.retrieval.provenance.corpus_version

    # context: only the selected evidence, under the generator's own tokenizer
    ctx = r.context
    assert ctx is not None and [b.evidence_id for b in ctx.blocks] == [e.id for e in sel.selected]
    assert ctx.tokenizer == sel.tokenizer == "onnx-community/Qwen2.5-0.5B-Instruct@cc5cc01a"
    assert ctx.context_tokens <= ctx.max_context_tokens

    # generation: the real local model, greedy, fully identified
    assert r.answer is not None and r.status is AnswerStatus.ANSWERED
    g = r.answer.generation
    assert g.generator.model == GeneratorSpec().model and g.generator.revision is not None
    assert g.generator.weights_sha256 and g.generator.local and g.deterministic
    assert g.raw_text == r.answer.text and g.prompt_hash == ctx.prompt_hash
    assert g.prompt_tokens and g.completion_tokens and g.latency_ms > 0
    assert "sunlight" in r.answer.text.lower()

    # claims and grounding: measured against the supplied evidence only
    factual = [cl for cl in r.claims if cl.kind is ClaimKind.FACTUAL]
    assert factual and r.grounding is not None
    selected_ids = {e.id for e in sel.selected}
    for cl in r.claims:
        assert r.answer.text[cl.char_start : cl.char_end] == cl.raw_text
        assert set(cl.supporting_evidence_ids) <= selected_ids
        assert set(cl.cited_evidence_ids) <= selected_ids
    assert any(cl.support is SupportStatus.SUPPORTED for cl in factual)
    assert top.id in factual[0].supporting_evidence_ids
    gr = r.grounding
    assert gr.factual_claims == len(factual) and gr.load_ms >= 0 and gr.latency_ms > 0
    assert gr.supported + gr.weakly_supported + gr.unsupported == len(factual)
    assert gr.citation_coverage == round(
        sum(any(x.valid for x in cl.citations) for cl in factual) / len(factual), 4
    )  # whatever the model cited is measured, never assumed

    # provenance: the complete chain, in order, with identities and timings
    chain = r.provenance.chain
    assert [s.stage for s in chain] == CHAIN
    by = {s.stage: s for s in chain}
    assert by["query_analysis"].hash == routing.analysis.analysis_hash
    assert by["router_decision"].hash == routing.decision.decision_hash
    assert by["retrieval"].config_hash == r.retrieval.provenance.configuration_hash
    assert by["evidence_selection"].hash == sel.selection_hash
    assert by["context"].hash == ctx.context_hash
    assert by["generation"].hash == g.answer_hash and by["generation"].origin == "generated"
    assert by["grounding"].hash == gr.grounding_hash and by["grounding"].origin == "measured"
    assert all(s.latency_ms is not None for s in chain if s.stage not in ("query", "claims"))
    cfg = r.provenance.configuration
    assert cfg.retrieval == r.retrieval.provenance.configuration
    assert cfg.generator_config_hash == g.generator.config_hash
    assert r.provenance.configuration_hash == cfg.config_hash()


def test_same_inputs_reproduce_every_deterministic_stage(lab: tuple[TestClient, str]) -> None:
    c, cid = lab
    a, b = (answer(c, cid, retrieval={"mode": "adaptive"}) for _ in range(2))
    assert a.query.id != b.query.id  # ids and timestamps differ...
    for x, y in zip(a.provenance.chain, b.provenance.chain, strict=True):
        assert (x.stage, x.hash, x.config_hash) == (y.stage, y.hash, y.config_hash)
        if x.stage != "query_analysis":  # ...but every identity matches
            assert x.deterministic
    assert a.answer is not None and b.answer is not None and a.answer.text == b.answer.text
    assert [cl.id for cl in a.claims] == [cl.id for cl in b.claims]
    assert a.provenance.configuration_hash == b.provenance.configuration_hash


def test_unanswerable_question_is_not_reported_as_grounded(lab: tuple[TestClient, str]) -> None:
    c, cid = lab
    # the passages are retrieved (they share terms) but none says who invented bread
    r = answer(c, cid, "who invented bread?", retrieval={"strategy": "sparse"})
    assert r.evidence.selected
    assert r.status in (AnswerStatus.ABSTAINED, AnswerStatus.ANSWERED)
    assert r.grounding is not None
    if r.status is AnswerStatus.ABSTAINED:
        assert r.grounding.status == "abstained" and r.grounding.grounding_score is None
    else:  # an attempted answer must not be measured as supported
        assert all(
            cl.support is not SupportStatus.SUPPORTED
            for cl in r.claims
            if cl.kind is ClaimKind.FACTUAL and "invent" in cl.text.lower()
        )


def test_no_evidence_gives_an_explicit_insufficient_result(lab: tuple[TestClient, str]) -> None:
    c, cid = lab
    r = answer(c, cid, "capital of France", retrieval={"strategy": "sparse"})
    assert r.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert r.retrieval.hits == [] and r.evidence.selected == []
    assert r.context is None and r.answer is None and r.claims == [] and r.grounding is None
    assert any("generator was not called" in w for w in r.warnings)
    assert [s.stage for s in r.provenance.chain] == ["query", "retrieval", "evidence_selection"]

    floor = answer(c, cid, retrieval={"strategy": "sparse"}, evidence={"min_score": 1e9})
    assert floor.status is AnswerStatus.INSUFFICIENT_EVIDENCE and floor.retrieval.hits
    # the configuration identity does not depend on the outcome
    same = answer(c, cid, retrieval={"strategy": "sparse"}, evidence={"min_score": 1e9})
    assert same.provenance.configuration_hash == floor.provenance.configuration_hash
    assert floor.provenance.configuration.generator_config_hash == GeneratorSpec().config_hash()
    assert {d.outcome for d in floor.evidence.decisions} == {"below_min_score"}
    assert any("below_min_score" in w for w in floor.warnings)


def test_evidence_over_the_token_budget_is_never_truncated(tmp_path: Path) -> None:
    long = " ".join(f"Plants need water and light in stage {i} of growth." for i in range(40))
    with TestClient(create_app(tmp_path)) as c:
        cid = c.post("/api/v1/corpora", json={"name": "b"}).json()["corpus"]["id"]
        c.post(f"/api/v1/corpora/{cid}/documents", files=[("files", ("long.txt", long.encode()))])
        body = {
            "retrieval": {"query": "plants water light", "strategy": "sparse"},
            "evidence": {"max_context_tokens": 64},
            "generation": {"generator": "extractive-baseline"},
        }
        r = RagResponse.model_validate(c.post(f"/api/v1/corpora/{cid}/answer", json=body).json())
    assert r.status is AnswerStatus.INSUFFICIENT_EVIDENCE and r.retrieval.hits
    assert {d.outcome for d in r.evidence.decisions} == {"over_token_budget"}
    assert all(d.token_count and d.token_count > 64 for d in r.evidence.decisions)
    assert any("over_token_budget" in w for w in r.warnings) and r.answer is None


def test_context_only_mode_and_extractive_baseline(lab: tuple[TestClient, str]) -> None:
    c, cid = lab
    dry = answer(c, cid, retrieval={"strategy": "sparse"}, generation=None)
    assert dry.status is AnswerStatus.NOT_GENERATED and dry.context is not None
    assert dry.answer is None and dry.grounding is None
    assert dry.provenance.configuration.generator is None
    assert [s.stage for s in dry.provenance.chain][-1] == "context"

    ext = answer(
        c, cid, retrieval={"strategy": "sparse"}, generation={"generator": "extractive-baseline"}
    )
    assert ext.answer is not None and ext.grounding is not None
    assert ext.context is not None and ext.context.tokenizer == "regex-word-punct@1"
    assert ext.grounding.citation_coverage == 1.0 and ext.grounding.invalid_citations == 0
    assert ext.grounding.status == "grounded"  # verbatim evidence: grounded, not necessarily good
    assert ext.provenance.configuration_hash != dry.provenance.configuration_hash


def test_answer_retrieval_equals_plain_retrieval(lab: tuple[TestClient, str]) -> None:
    c, cid = lab
    req = {"query": QUERY, "top_k": 3, "strategy": "hybrid", "rerank": {"enabled": True}}
    plain = c.post(f"/api/v1/corpora/{cid}/retrieve", json=req)
    assert plain.status_code == 200
    r = answer(c, cid, retrieval=req, generation=None)
    p = plain.json()
    assert [h["result"]["chunk_id"] for h in p["hits"]] == [
        h.result.chunk_id for h in r.retrieval.hits
    ]
    assert p["provenance"]["configuration_hash"] == r.retrieval.provenance.configuration_hash
    assert "evidence" not in p and "answer" not in p  # the retrieval contract is unchanged


def test_components_and_health_never_load_models(tmp_path: Path) -> None:
    gen = OnnxCausalLM(GeneratorSpec())
    with TestClient(create_app(tmp_path, generator=gen)) as c:
        comp = c.get("/api/v1/rag/components").json()
        health = {x["name"]: x for x in c.get("/api/v1/health").json()["components"]}
    assert comp["default_generator"] == GeneratorSpec().model
    names = {g["name"]: g for g in comp["generators"]}
    assert set(names) == {GeneratorSpec().model, "extractive-baseline"}
    assert names[GeneratorSpec().model]["loaded"] is False and gen._loaded is None
    assert comp["prompt_template"] == "grounded-qa@1"
    (verifier,) = comp["verifiers"]
    assert verifier["name"] == "lexical-semantic" and verifier["detects_contradiction"] is False
    assert health["llm_provider"]["state"] == "ok"
    assert "loads on first answer" in health["llm_provider"]["detail"]


class _NoEmbedder:
    """Retrieval (BM25) works; the grounding model does not."""

    spec = EmbedderSpec()

    def info(self) -> EmbedderInfo:
        raise EmbedderUnavailableError("offline")

    def embed_documents(self, texts: Any) -> Vectors:
        raise EmbedderUnavailableError("offline")

    def embed_query(self, text: str) -> Vectors:
        raise EmbedderUnavailableError("offline")


class _CountingGenerator(ExtractiveGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.name = "counting"
        self.calls = 0

    def generate(self, inp: GenerationInput) -> GenerationOutput:
        self.calls += 1
        return super().generate(inp)


def test_failures_are_explicit(tmp_path: Path, cross_encoder: OnnxCrossEncoder) -> None:
    missing = OnnxCausalLM(GeneratorSpec(model="rag-forge/missing-model", revision="0" * 40))
    with TestClient(create_app(tmp_path / "a", _NoEmbedder(), cross_encoder, missing)) as c:
        cid = c.post("/api/v1/corpora", json={"name": "f"}).json()["corpus"]["id"]
        c.post(f"/api/v1/corpora/{cid}/documents", files=[("files", d) for d in DOCS])
        url = f"/api/v1/corpora/{cid}/answer"
        sparse = {"query": QUERY, "strategy": "sparse"}

        r = c.post(url, json={"retrieval": sparse})  # the default generator cannot load
        assert r.status_code == 503 and r.json()["detail"]["component"] == "generator"

        r = c.post(url, json={"retrieval": sparse, "generation": {"generator": "gpt-imaginary"}})
        assert r.status_code == 501 and r.json()["detail"]["capability"] == "Generator"
        assert "extractive-baseline" in r.json()["detail"]["message"]

        r = c.post(url, json={"retrieval": sparse, "grounding": {"verifier": "nli"}})
        assert r.status_code == 501 and r.json()["detail"]["capability"] == "Grounding verifier"

        # generation succeeded but grounding could not run: no answer is returned
        ext = {"generator": "extractive-baseline"}
        r = c.post(url, json={"retrieval": sparse, "generation": ext})
        assert r.status_code == 503 and r.json()["detail"]["component"] == "grounding"

        for bad in (
            {"retrieval": {"query": " "}},
            {"retrieval": sparse, "evidence": {"max_items": 0}},
            {"retrieval": sparse, "evidence": {"max_context_tokens": 10}},
            {"retrieval": sparse, "generation": {"temperature": 3}},
            {"retrieval": sparse, "generation": {"max_new_tokens": 0}},
            {"retrieval": sparse, "unknown": 1},
        ):
            assert c.post(url, json=bad).status_code == 422, bad
        assert c.post("/api/v1/corpora/nope/answer", json={"retrieval": sparse}).status_code == 404
        r = c.post(url, json={"retrieval": {**sparse, "version": 99}})
        assert r.status_code == 404

        # retrieval failures surface exactly as on /retrieve
        r = c.post(url, json={"retrieval": {"query": QUERY, "strategy": "dense"}})
        assert (
            r.status_code
            == c.post(
                f"/api/v1/corpora/{cid}/retrieve", json={"query": QUERY, "strategy": "dense"}
            ).status_code
        )


def test_generator_is_not_called_without_evidence(tmp_path: Path) -> None:
    from rag_forge.main import make_generators
    from rag_forge.rag.grounding import LexicalSemanticVerifier
    from rag_forge.rag.service import RagService

    counting = _CountingGenerator()
    with TestClient(create_app(tmp_path)) as c:
        app: Any = c.app
        app.state.rag = RagService(
            app.state.retrieval,
            {**make_generators(counting)},
            {"lexical-semantic": LexicalSemanticVerifier(_NoEmbedder())},
            default_generator="counting",
        )
        cid = c.post("/api/v1/corpora", json={"name": "g"}).json()["corpus"]["id"]
        c.post(f"/api/v1/corpora/{cid}/documents", files=[("files", d) for d in DOCS])
        body = {"retrieval": {"query": "zzzz qqqq", "strategy": "sparse"}}
        r = c.post(f"/api/v1/corpora/{cid}/answer", json=body)
    assert r.status_code == 200 and r.json()["status"] == "insufficient_evidence"
    assert counting.calls == 0
