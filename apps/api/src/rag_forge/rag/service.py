"""Query -> (router) -> retrieval -> (reranking) -> evidence -> context -> generation -> claims
-> grounding -> cited answer with a complete provenance chain.

Every component is resolved before anything runs, so a request never retrieves and then fails
for a missing generator or verifier. Retrieval is the unchanged `RetrievalService`; this stage
only consumes its response. Failures propagate: an answer is never returned without grounding.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping

from rag_forge.domain.models import (
    AnswerStatus,
    Claim,
    ClaimKind,
    ContentOrigin,
    EvidenceParams,
    EvidenceSelection,
    FinishReason,
    GeneratedAnswer,
    GenerationContext,
    GenerationRecord,
    GroundingReport,
    PipelineStage,
    RagConfiguration,
    RagProvenance,
    RagRequest,
    RagResponse,
    RetrievalResponse,
    canonical_hash,
)
from rag_forge.provenance.environment import capture_environment
from rag_forge.rag.claims import EXTRACTOR, extract_claims
from rag_forge.rag.evidence import build_context, select_evidence
from rag_forge.rag.generation import GenerationInput, Generator
from rag_forge.rag.grounding import GroundingVerifier, summarize
from rag_forge.rag.prompt import GROUNDED_QA, PromptTemplate
from rag_forge.retrieval.service import RetrievalService


class RagComponentNotAvailableError(LookupError):
    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self.kind = kind
        names = ", ".join(sorted(available)) or "none"
        super().__init__(f"{kind} '{name}' is not registered (available: {names})")


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


class RagService:
    def __init__(
        self,
        retrieval: RetrievalService,
        generators: Mapping[str, Generator],
        verifiers: Mapping[str, GroundingVerifier],
        default_generator: str,
        template: PromptTemplate = GROUNDED_QA,
    ) -> None:
        if default_generator not in generators:
            raise ValueError(f"default generator {default_generator!r} is not registered")
        self.retrieval = retrieval
        self.generators = dict(generators)  # instances hold loaded models
        self.verifiers = dict(verifiers)
        self.default_generator = default_generator
        self.template = template

    def answer(self, corpus_id: str, request: RagRequest) -> RagResponse:
        started = time.perf_counter()
        gen_params = request.generation.effective() if request.generation else None
        name = (gen_params.generator if gen_params else None) or self.default_generator
        if gen_params is not None:  # record the resolved name, never "the default"
            gen_params = gen_params.model_copy(update={"generator": name})
        generator = self.generators.get(name)
        if generator is None:
            raise RagComponentNotAvailableError("generator", name, list(self.generators))
        verifier = self.verifiers.get(request.grounding.verifier)
        if verifier is None:
            raise RagComponentNotAvailableError(
                "verifier", request.grounding.verifier, list(self.verifiers)
            )

        retrieval = self.retrieval.retrieve(corpus_id, request.retrieval)
        query = retrieval.query
        selection = select_evidence(
            retrieval, request.evidence, generator.count_tokens, generator.tokenizer_id
        )
        evidence = selection.selected
        warnings = list(retrieval.warnings)
        context: GenerationContext | None = None
        answer: GeneratedAnswer | None = None
        claims: list[Claim] = []
        grounding: GroundingReport | None = None
        context_ms = None
        if not evidence:
            status = AnswerStatus.INSUFFICIENT_EVIDENCE
            if retrieval.hits:
                outcomes = Counter(d.outcome.value for d in selection.decisions)
                why = ", ".join(f"{n} {o}" for o, n in sorted(outcomes.items()))
                warnings.append(f"no usable evidence ({why}): the generator was not called")
            else:
                warnings.append("retrieval returned no passages: the generator was not called")
        else:
            t = time.perf_counter()
            context = build_context(
                request.retrieval.query,
                evidence,
                self.template,
                request.evidence.max_context_tokens,
                generator.count_tokens,
                generator.tokenizer_id,
            )
            context_ms = _ms(t)
            status = AnswerStatus.NOT_GENERATED
        if context is not None and gen_params is not None:
            info = generator.info()  # loads the model once; its failure fails the request
            output = generator.generate(
                GenerationInput(
                    messages=context.messages,
                    question=request.retrieval.query,
                    blocks=context.blocks,
                    params=gen_params,
                )
            )
            answer_hash = canonical_hash(
                {"prompt": context.prompt_hash, "text": output.text, "generator": info.config_hash}
            )
            record = GenerationRecord(
                generator=info,
                params=gen_params,
                params_hash=canonical_hash(gen_params.model_dump(mode="json")),
                prompt_hash=context.prompt_hash,
                raw_text=output.text,
                answer_hash=answer_hash,
                finish_reason=output.finish_reason,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                deterministic=info.local
                and info.deterministic_at_zero_temperature
                and gen_params.temperature == 0,
                load_ms=output.load_ms,
                latency_ms=output.latency_ms,
            )
            answer = GeneratedAnswer(text=output.text, generation=record)
            if output.finish_reason is FinishReason.LENGTH:
                warnings.append("the answer reached max_new_tokens and may be cut off")
            verifier_load_ms = verifier.prepare()  # its failure fails the request
            t = time.perf_counter()
            drafts = extract_claims(output.text, evidence, answer_hash)
            claims = verifier.verify(drafts, evidence)
            grounding = summarize(claims, evidence, verifier, _ms(t), verifier_load_ms)
            factual = any(c.kind is ClaimKind.FACTUAL for c in claims)
            abstained = any(c.kind is ClaimKind.ABSTENTION for c in claims)
            status = AnswerStatus.ABSTAINED if abstained and not factual else AnswerStatus.ANSWERED
            if grounding.invalid_citations:
                warnings.append(
                    f"the answer cites {grounding.invalid_citations} passage id(s) that were "
                    "not supplied"
                )

        configuration = RagConfiguration(
            retrieval=retrieval.provenance.configuration,
            evidence=request.evidence,
            prompt_template=self.template.id if gen_params else None,
            generator=name if gen_params else None,
            generator_config_hash=generator.describe().config_hash if gen_params else None,
            generation=gen_params,
            verifier=f"{verifier.name}@{verifier.version}" if gen_params else None,
            verifier_config_hash=verifier.config_hash() if gen_params else None,
        )
        return RagResponse(
            query=query,
            status=status,
            retrieval=retrieval,
            evidence=selection,
            context=context,
            answer=answer,
            claims=claims,
            grounding=grounding,
            provenance=RagProvenance(
                configuration=configuration,
                configuration_hash=configuration.config_hash(),
                chain=_chain(retrieval, request.evidence, selection, context, context_ms, answer)
                + _grounding_stages(answer, claims, grounding, verifier),
                environment=capture_environment(),
                elapsed_ms=_ms(started),
            ),
            warnings=warnings,
        )


def _chain(
    retrieval: RetrievalResponse,
    params: EvidenceParams,
    selection: EvidenceSelection,
    context: GenerationContext | None,
    context_ms: float | None,
    answer: GeneratedAnswer | None,
) -> list[PipelineStage]:
    prov = retrieval.provenance
    stages = [
        PipelineStage(
            stage="query",
            hash=canonical_hash(retrieval.query.text),
            config_hash=None,
            latency_ms=None,
            origin=ContentOrigin.RETRIEVED,
            deterministic=True,
            detail=f"query {retrieval.query.id}",
        )
    ]
    if prov.routing is not None:
        r = prov.routing
        stages += [
            PipelineStage(
                stage="query_analysis",
                hash=r.analysis.analysis_hash,
                config_hash=r.analysis.config_hash,
                latency_ms=r.analysis_ms,
                origin=ContentOrigin.INFERRED,
                deterministic=True,
                detail=f"{r.analysis.analyzer}@{r.analysis.analyzer_version}: "
                f"{r.analysis.labels.query_class.value}, {r.analysis.labels.complexity.value}",
            ),
            PipelineStage(
                stage="router_decision",
                hash=r.decision.decision_hash,
                config_hash=r.routing_hash,
                latency_ms=r.decision_ms,
                origin=ContentOrigin.INFERRED,
                deterministic=True,
                detail=f"{r.decision.policy}@{r.decision.policy_version} -> "
                f"{r.decision.option.value}",
            ),
        ]
    upstream = prov.elapsed_ms - (prov.reranking.latency_ms if prov.reranking else 0)
    if prov.routing is not None:
        upstream -= prov.routing.analysis_ms + prov.routing.decision_ms
    stages.append(
        PipelineStage(
            stage="retrieval",
            hash=canonical_hash([h.result.chunk_id for h in retrieval.hits]),
            config_hash=prov.configuration_hash,
            latency_ms=round(max(upstream, 0.0), 3),
            origin=ContentOrigin.RETRIEVED,
            deterministic=True,
            detail=f"{prov.retriever} on {prov.corpus_id} v{prov.corpus_version}: "
            f"{len(retrieval.hits)} hits",
        )
    )
    if prov.reranking is not None:
        rr = prov.reranking
        stages.append(
            PipelineStage(
                stage="reranking",
                hash=canonical_hash([[h.result.chunk_id, h.result.score] for h in retrieval.hits]),
                config_hash=rr.reranker_config_hash,
                latency_ms=rr.latency_ms,
                origin=ContentOrigin.MEASURED,
                deterministic=True,
                detail=f"{rr.info.spec.model}: {rr.candidates_scored} candidates scored",
            )
        )
    stages.append(
        PipelineStage(
            stage="evidence_selection",
            hash=selection.selection_hash,
            config_hash=params.config_hash(),
            latency_ms=selection.latency_ms,
            origin=ContentOrigin.RETRIEVED,
            deterministic=True,
            detail=f"{selection.selector}: {len(selection.selected)} of {selection.candidates} "
            f"passages, {selection.tokens_used} tokens",
        )
    )
    if context is not None:
        stages.append(
            PipelineStage(
                stage="context",
                hash=context.context_hash,
                config_hash=canonical_hash(context.prompt_template),
                latency_ms=context_ms,
                origin=ContentOrigin.RETRIEVED,
                deterministic=True,
                detail=f"{context.prompt_template}: {context.context_tokens} of "
                f"{context.max_context_tokens} context tokens",
            )
        )
    if answer is not None:
        g = answer.generation
        stages.append(
            PipelineStage(
                stage="generation",
                hash=g.answer_hash,
                config_hash=canonical_hash([g.generator.config_hash, g.params_hash]),
                latency_ms=g.latency_ms,
                origin=ContentOrigin.GENERATED,
                deterministic=g.deterministic,
                detail=f"{g.generator.model} ({g.generator.provider}): "
                f"{g.completion_tokens} tokens, {g.finish_reason.value}",
            )
        )
    return stages


def _grounding_stages(
    answer: GeneratedAnswer | None,
    claims: list[Claim],
    grounding: GroundingReport | None,
    verifier: GroundingVerifier,
) -> list[PipelineStage]:
    if answer is None or grounding is None:
        return []
    return [
        PipelineStage(
            stage="claims",
            hash=canonical_hash([[c.id, c.text, c.kind] for c in claims]),
            config_hash=canonical_hash(EXTRACTOR),
            latency_ms=None,
            origin=ContentOrigin.INFERRED,
            deterministic=True,
            detail=f"{EXTRACTOR}: {len(claims)} claims",
        ),
        PipelineStage(
            stage="grounding",
            hash=grounding.grounding_hash,
            config_hash=grounding.config_hash,
            latency_ms=grounding.latency_ms,
            origin=ContentOrigin.MEASURED,
            deterministic=True,
            detail=f"{verifier.name}@{verifier.version}: {grounding.status.value}",
        ),
    ]
