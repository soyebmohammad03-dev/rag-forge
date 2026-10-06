"use client";

import type { CorpusSummary, RagComponents, RagRequest, RagResponse, RetrievalMode, RetrievalRequest } from "@rag-forge/shared";
import { Database, Microscope } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, useState } from "react";
import { Button, Kbd } from "@/components/ui/button";
import { FormField, Input, Segmented } from "@/components/ui/input";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { fetchCorpora, fetchRagComponents, useApi } from "@/lib/use-api";
import { DENSE_STATE, DenseStateBadge, useDenseIndex } from "@/features/retrieval/dense-index";
import { errorMessage } from "@/features/retrieval/errors";
import { RerankAnalysis } from "@/features/retrieval/rerank-analysis";
import { QueryIntelligencePanel, RouterDecisionPanel } from "@/features/retrieval/routing-panels";
import { AnswerPanel, ClaimList, type Focus, GroundingPanel } from "./answer-panels";
import { EvidencePanel, PipelineTrace, RagProvenancePanel, SelectionTable } from "./evidence-panels";

type Strategy = "sparse" | "dense" | "rrf";
type Run = { kind: "idle" } | { kind: "busy" } | { kind: "done"; response: RagResponse } | { kind: "error"; message: string };

const CONTEXT_ONLY = "__context_only__";
/** The server's default reranker, as in the Retrieval Lab; provenance shows its exact revision. */
const RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2";
const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: "sparse", label: "BM25" },
  { value: "dense", label: "Dense" },
  { value: "rrf", label: "Hybrid RRF" },
];
const select =
  "h-9 w-full rounded-md border border-line-strong bg-surface px-2.5 font-mono text-[12px] text-fg focus:border-trace focus:outline-none";

export function EvidencePage() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const corpora = useApi(fetchCorpora);
  const components = useApi(fetchRagComponents);
  const selectedId = params.get("corpus") ?? corpora.data?.[0]?.corpus.id ?? "";
  const summary = corpora.data?.find((c) => c.corpus.id === selectedId);

  if ((corpora.loading && !corpora.data) || (components.loading && !components.data))
    return <Panel className="mx-auto max-w-[1440px]"><LoadingState rows={5} label="Loading" /></Panel>;
  if (corpora.error || components.error || !components.data)
    return <Panel className="mx-auto max-w-3xl"><ErrorState title="Could not reach the API">{corpora.error ?? components.error}</ErrorState></Panel>;

  return (
    <div className="mx-auto max-w-[1600px] space-y-5">
      <header>
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Evidence</p>
        <h1 className="mt-1.5 text-2xl font-medium tracking-tight">Evidence Lab</h1>
        <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-fg-muted">
          Run the whole pipeline on one corpus version: route, retrieve, rerank, select evidence, generate from that
          evidence only, then check every claim against it. Grounding is measured overlap with the supplied passages; it
          is not a judgement of answer quality or truth.
        </p>
      </header>
      {!corpora.data?.length || !summary ? (
        <Panel>
          <EmptyState icon={Database} title="No corpora to answer from" action={<Link href="/corpus" className="text-xs text-trace underline-offset-4 hover:underline">Create a corpus</Link>}>
            Answers are generated only from ingested chunks. Create a corpus and upload documents first.
          </EmptyState>
        </Panel>
      ) : (
        <Workspace
          key={summary.corpus.id}
          corpora={corpora.data}
          summary={summary}
          components={components.data}
          onPickCorpus={(id) => router.replace(`${pathname}?corpus=${encodeURIComponent(id)}`)}
        />
      )}
    </div>
  );
}

function num(v: string, fallback: number) {
  const n = Number(v);
  return Number.isFinite(n) && v !== "" ? n : fallback;
}

function Workspace({
  corpora,
  summary,
  components,
  onPickCorpus,
}: {
  corpora: CorpusSummary[];
  summary: CorpusSummary;
  components: RagComponents;
  onPickCorpus: (id: string) => void;
}) {
  const ev = components.evidence_defaults;
  const gen = components.generation_defaults;
  const [query, setQuery] = useState("");
  const [version, setVersion] = useState<number | null>(null);
  const [routing, setRouting] = useState<RetrievalMode>("adaptive");
  const [strategy, setStrategy] = useState<Strategy>("sparse");
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(10);
  const [maxItems, setMaxItems] = useState(ev.max_items ?? 5);
  const [maxTokens, setMaxTokens] = useState(ev.max_context_tokens ?? 1500);
  const [perDoc, setPerDoc] = useState<string>(String(ev.max_per_document ?? ""));
  const [dupe, setDupe] = useState<string>(String(ev.near_duplicate_threshold ?? ""));
  const [minScore, setMinScore] = useState("");
  const [generator, setGenerator] = useState(components.default_generator);
  const [maxNew, setMaxNew] = useState(gen.max_new_tokens ?? 200);
  const [temperature, setTemperature] = useState(gen.temperature ?? 0);
  const [seed, setSeed] = useState(gen.seed ?? 0);
  const [run, setRun] = useState<Run>({ kind: "idle" });
  const [focus, setFocus] = useState<Focus>({ claim: null, evidence: null });
  const dense = useDenseIndex(summary.corpus.id, version);
  const denseState = dense.data?.state;
  const adaptive = routing === "adaptive";
  const denseBlocked = !adaptive && strategy !== "sparse" && denseState !== "ready";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || denseBlocked) return;
    setRun({ kind: "busy" });
    setFocus({ claim: null, evidence: null });
    const retrieval: RetrievalRequest = {
      query,
      top_k: topK,
      version,
      mode: routing,
      strategy: strategy === "rrf" ? "hybrid" : strategy,
      ...(strategy === "rrf" ? { hybrid: { fusion: "rrf", rrf_k: 60, candidate_k: Math.max(50, topK), weights: { sparse: 0.5, dense: 0.5 } } } : {}),
      rerank: { enabled: rerank && !adaptive, model: RERANK_MODEL, candidate_k: Math.max(50, topK) },
    };
    const body: RagRequest = {
      retrieval,
      evidence: {
        max_items: maxItems,
        max_context_tokens: maxTokens,
        max_per_document: perDoc === "" ? null : num(perDoc, 2),
        near_duplicate_threshold: dupe === "" ? null : num(dupe, 0.8),
        min_score: minScore === "" ? null : num(minScore, 0),
      },
      generation: generator === CONTEXT_ONLY ? null : { generator, max_new_tokens: maxNew, temperature, top_p: gen.top_p ?? 1, seed },
    };
    try {
      const { data, error, response } = await api.POST("/api/v1/corpora/{corpus_id}/answer", {
        params: { path: { corpus_id: summary.corpus.id } },
        body,
      });
      if (!data) throw new Error(errorMessage(error, response.status));
      setRun({ kind: "done", response: data });
    } catch (err) {
      setRun({ kind: "error", message: err instanceof TypeError ? "API unreachable" : (err as Error).message });
    }
  };

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[340px_1fr]">
      <div className="space-y-5">
        <Panel>
          <PanelHeader eyebrow="Query" title="Answer request" actions={denseState && <DenseStateBadge state={denseState} />} />
          <form onSubmit={submit} className="space-y-4 p-4">
            <div className="grid grid-cols-[1fr_96px] gap-3">
              <FormField label="Corpus" htmlFor="el-corpus">
                <select id="el-corpus" className={select} value={summary.corpus.id} onChange={(e) => onPickCorpus(e.target.value)}>
                  {corpora.map((c) => <option key={c.corpus.id} value={c.corpus.id}>{c.corpus.name}</option>)}
                </select>
              </FormField>
              <FormField label="Version" htmlFor="el-version">
                <select id="el-version" className={select} value={version ?? "current"} onChange={(e) => setVersion(e.target.value === "current" ? null : Number(e.target.value))}>
                  <option value="current">v{summary.corpus.version} ●</option>
                  {Array.from({ length: summary.corpus.version }, (_, i) => summary.corpus.version - 1 - i).map((v) => <option key={v} value={v}>v{v}</option>)}
                </select>
              </FormField>
            </div>
            <FormField label="Question" htmlFor="el-query">
              <Input id="el-query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. what do plants need to make food?" maxLength={2000} autoFocus />
            </FormField>

            <fieldset className="space-y-2 rounded-md border border-line p-3">
              <legend className="px-1 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">Retrieval</legend>
              <Segmented name="routing" value={routing} options={[{ value: "adaptive", label: "Router" }, { value: "manual", label: "Fixed" }]} onChange={setRouting} />
              {adaptive ? (
                <p className="text-[11px] leading-relaxed text-fg-subtle">The router chooses strategy, fusion and reranking from the query analysis.</p>
              ) : (
                <>
                  <Segmented name="strategy" value={strategy} options={STRATEGIES} onChange={setStrategy} />
                  <label className="flex items-center gap-2 text-[11px] text-fg-muted">
                    <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} className="accent-[var(--color-trace)]" />
                    Rerank with the cross-encoder
                  </label>
                </>
              )}
              <FormField label="Top k" htmlFor="el-topk" hint="ranked passages offered to evidence selection">
                <Input id="el-topk" type="number" min={1} max={100} value={topK} onChange={(e) => setTopK(e.target.valueAsNumber || 1)} />
              </FormField>
            </fieldset>

            <fieldset className="grid grid-cols-2 gap-3 rounded-md border border-line p-3">
              <legend className="px-1 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">Evidence budget</legend>
              <FormField label="Max passages" htmlFor="el-items"><Input id="el-items" type="number" min={1} max={20} value={maxItems} onChange={(e) => setMaxItems(e.target.valueAsNumber || 1)} /></FormField>
              <FormField label="Max tokens" htmlFor="el-tokens"><Input id="el-tokens" type="number" min={64} max={16000} value={maxTokens} onChange={(e) => setMaxTokens(e.target.valueAsNumber || 64)} /></FormField>
              <FormField label="Per document" htmlFor="el-perdoc" hint="blank = no cap"><Input id="el-perdoc" type="number" min={1} max={20} value={perDoc} onChange={(e) => setPerDoc(e.target.value)} /></FormField>
              <FormField label="Near-dup ≥" htmlFor="el-dupe" hint="term Jaccard; blank = off"><Input id="el-dupe" type="number" min={0.05} max={1} step={0.05} value={dupe} onChange={(e) => setDupe(e.target.value)} /></FormField>
              <div className="col-span-2">
                <FormField label="Min score" htmlFor="el-min" hint="final ranking score floor; blank = none"><Input id="el-min" type="number" step="any" value={minScore} onChange={(e) => setMinScore(e.target.value)} /></FormField>
              </div>
            </fieldset>

            <fieldset className="space-y-3 rounded-md border border-line p-3">
              <legend className="px-1 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">Generation</legend>
              <FormField label="Generator" htmlFor="el-gen">
                <select id="el-gen" className={select} value={generator} onChange={(e) => setGenerator(e.target.value)}>
                  {components.generators.map((g) => (
                    <option key={g.name} value={g.name}>{g.name}{g.local ? "" : " (remote)"}{g.name === components.default_generator ? " ●" : ""}</option>
                  ))}
                  <option value={CONTEXT_ONLY}>none: stop after context</option>
                </select>
              </FormField>
              {generator !== CONTEXT_ONLY && (
                <div className="grid grid-cols-3 gap-3">
                  <FormField label="Max new" htmlFor="el-maxnew"><Input id="el-maxnew" type="number" min={1} max={1024} value={maxNew} onChange={(e) => setMaxNew(e.target.valueAsNumber || 1)} /></FormField>
                  <FormField label="Temp" htmlFor="el-temp"><Input id="el-temp" type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(e.target.valueAsNumber || 0)} /></FormField>
                  <FormField label="Seed" htmlFor="el-seed"><Input id="el-seed" type="number" min={0} value={seed} disabled={temperature === 0} onChange={(e) => setSeed(e.target.valueAsNumber || 0)} /></FormField>
                </div>
              )}
              <p className="text-[11px] leading-relaxed text-fg-subtle">
                {temperature === 0 ? "Greedy decoding: a local model reproduces the same answer." : "Sampling: reproducible only with the same seed and a local model."}
                {" "}Prompt contract {components.prompt_template}; grounding by {components.verifiers.map((v) => `${v.name}@${v.version}`).join(", ")}.
              </p>
            </fieldset>

            {denseBlocked && (
              <p role="status" className="rounded-md border border-line-strong bg-surface-2 px-3 py-2 text-[11px] leading-relaxed text-fg-muted">
                Dense retrieval is unavailable for this version: {denseState ? DENSE_STATE[denseState].help : "checking index status…"}
              </p>
            )}
            <div className="flex justify-end">
              <Button type="submit" variant="primary" disabled={!query.trim() || run.kind === "busy" || denseBlocked}>
                <Microscope className="size-4" /> {run.kind === "busy" ? "Answering…" : "Answer"} <Kbd className="border-black/20 bg-black/10 text-black/70">↵</Kbd>
              </Button>
            </div>
          </form>
        </Panel>
        {run.kind === "done" && <RagProvenancePanel response={run.response} />}
      </div>
      <div className="min-w-0">
        <Results run={run} corpusId={summary.corpus.id} focus={focus} onFocus={setFocus} />
      </div>
    </div>
  );
}

function Results({ run, corpusId, focus, onFocus }: { run: Run; corpusId: string; focus: Focus; onFocus: (f: Focus) => void }) {
  if (run.kind === "idle")
    return (
      <Panel>
        <EmptyState icon={Microscope} title="Ask a question">
          The answer appears with its claims underlined by measured support, the evidence it was generated from, every
          selection decision, and the full provenance chain from query to grounding.
        </EmptyState>
      </Panel>
    );
  if (run.kind === "busy") return <Panel><LoadingState rows={6} label="Running the pipeline (the first answer also loads the local model)" /></Panel>;
  if (run.kind === "error") return <Panel><ErrorState title="The pipeline failed">{run.message}</ErrorState></Panel>;
  const r = run.response;
  const routing = r.retrieval.provenance.routing;
  return (
    <div className="space-y-5">
      <PipelineTrace chain={r.provenance.chain} total={r.provenance.elapsed_ms} />
      <div className="grid items-start gap-5 2xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-5">
          <AnswerPanel response={r} focus={focus} onFocus={onFocus} />
          {r.grounding && <GroundingPanel report={r.grounding} />}
          {r.answer && <ClaimList claims={r.claims} evidence={r.evidence.selected} focus={focus} onFocus={onFocus} />}
        </div>
        <div className="min-w-0 space-y-5">
          <EvidencePanel response={r} corpusId={corpusId} focus={focus} onFocus={onFocus} />
          <SelectionTable selection={r.evidence} />
        </div>
      </div>
      {routing && (
        <section aria-label="Query intelligence and routing" className="grid items-start gap-5 2xl:grid-cols-2">
          <QueryIntelligencePanel analysis={routing.analysis} />
          <RouterDecisionPanel decision={routing.decision} />
        </section>
      )}
      {r.retrieval.provenance.reranking && r.retrieval.hits.length > 0 && <RerankAnalysis response={r.retrieval} />}
    </div>
  );
}
