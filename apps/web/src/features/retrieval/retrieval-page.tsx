"use client";

import type { CorpusSummary, RetrievalProvenance, RetrievalResponse, RetrievalStrategy } from "@rag-forge/shared";
import { Database, Layers, ScanSearch, SearchX } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { type FormEvent, useState } from "react";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { Button, Kbd } from "@/components/ui/button";
import { FormField, Input, Segmented } from "@/components/ui/input";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { shortHash } from "@/lib/format";
import { fetchCorpora, useApi } from "@/lib/use-api";
import { Field } from "@/features/corpus/bits";
import { RUN_LABEL, type RunKey, type Runs } from "./compare";
import { CompareView } from "./compare-view";
import { DENSE_STATE, DenseIndexPanel, DenseStateBadge, useDenseIndex } from "./dense-index";
import { EvidenceCard } from "./evidence-card";
import { errorMessage } from "./errors";
import { RerankAnalysis } from "./rerank-analysis";

/** The server's default reranker; the provenance panel shows the exact revision and weights. */
const RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2";

type Mode = RunKey | "compare";
type Run =
  | { kind: "idle" }
  | { kind: "busy" }
  | { kind: "single"; response: RetrievalResponse }
  | { kind: "compare"; runs: Required<Runs> }
  | { kind: "error"; message: string };

const MODES: { value: Mode; label: string }[] = [
  { value: "sparse", label: "BM25" },
  { value: "dense", label: "Dense" },
  { value: "rrf", label: "RRF" },
  { value: "weighted", label: "Weighted" },
  { value: "compare", label: "Compare" },
];

const MODE_HELP: Record<Mode, string> = {
  sparse: "BM25: exact term overlap.",
  dense: "Dense: cosine similarity of embeddings.",
  rrf: "Hybrid RRF: fuse BM25 and dense by rank, 1 / (k + rank). The primary hybrid baseline.",
  weighted: "Hybrid weighted: fuse min-max normalised BM25 and dense scores with weights.",
  compare: "Run all four on the same query and compare their rankings.",
};

const select =
  "h-9 w-full rounded-md border border-line-strong bg-surface px-2.5 font-mono text-[12px] text-fg focus:border-trace focus:outline-none";

export function RetrievalPage() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const corpora = useApi(fetchCorpora);
  const selectedId = params.get("corpus") ?? corpora.data?.[0]?.corpus.id ?? "";
  const summary = corpora.data?.find((c) => c.corpus.id === selectedId);

  if (corpora.loading && !corpora.data) return <Panel className="mx-auto max-w-[1440px]"><LoadingState rows={5} label="Loading corpora" /></Panel>;
  if (corpora.error && !corpora.data) return <Panel className="mx-auto max-w-3xl"><ErrorState title="Could not load corpora">{corpora.error}</ErrorState></Panel>;

  return (
    <div className="mx-auto max-w-[1440px] space-y-5">
      <header>
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Retrieval</p>
        <h1 className="mt-1.5 text-2xl font-medium tracking-tight">Retrieval Lab</h1>
        <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-fg-muted">
          Run a query against one corpus version with BM25 (exact terms), dense retrieval (semantic similarity) or a
          hybrid of both, optionally rerank the candidates with a local cross-encoder, or compare strategies side by
          side. This is inspection, not evaluation: no quality metrics are computed here.
        </p>
      </header>
      {!corpora.data?.length || !summary ? (
        <Panel>
          <EmptyState icon={Database} title="No corpora to search" action={<Link href="/corpus" className="text-xs text-trace underline-offset-4 hover:underline">Create a corpus</Link>}>
            Retrieval runs over ingested chunks. Create a corpus and upload documents first.
          </EmptyState>
        </Panel>
      ) : (
        <Workspace
          key={summary.corpus.id}
          corpora={corpora.data}
          summary={summary}
          onPickCorpus={(id) => router.replace(`${pathname}?corpus=${encodeURIComponent(id)}`)}
        />
      )}
    </div>
  );
}

function Workspace({ corpora, summary, onPickCorpus }: { corpora: CorpusSummary[]; summary: CorpusSummary; onPickCorpus: (id: string) => void }) {
  const [mode, setMode] = useState<Mode>("sparse");
  const [query, setQuery] = useState("");
  const [version, setVersion] = useState<number | null>(null);
  const [topK, setTopK] = useState(10);
  const [k1, setK1] = useState(1.2);
  const [b, setB] = useState(0.75);
  const [rrfK, setRrfK] = useState(60);
  const [candidateK, setCandidateK] = useState(50);
  const [denseWeight, setDenseWeight] = useState(0.5);
  const [rerankOn, setRerankOn] = useState(false);
  const [poolK, setPoolK] = useState(50);
  const [run, setRun] = useState<Run>({ kind: "idle" });
  const dense = useDenseIndex(summary.corpus.id, version);
  const denseState = dense.data?.state;
  const needsDense = mode !== "sparse";
  const usesHybrid = mode === "rrf" || mode === "weighted" || mode === "compare";
  const rerank = rerankOn && mode !== "compare"; // reranking analysis runs on one upstream strategy
  const rerankError = rerank && poolK < topK ? "The candidate pool must be at least the final top k." : null;
  const hybridError =
    usesHybrid && candidateK < (rerank ? poolK : topK)
      ? rerank
        ? "Candidates per retriever must be at least the rerank pool."
        : "Candidates per retriever must be at least top k."
      : null;
  const denseBlocked = needsDense && denseState !== "ready";

  const retrieve = async (key: RunKey) => {
    const strategy: RetrievalStrategy = key === "rrf" || key === "weighted" ? "hybrid" : key;
    const w = Math.round(denseWeight * 100) / 100;
    const { data, error, response } = await api.POST("/api/v1/corpora/{corpus_id}/retrieve", {
      params: { path: { corpus_id: summary.corpus.id } },
      body: {
        query,
        top_k: topK,
        version,
        strategy,
        bm25: { k1, b },
        hybrid: {
          fusion: key === "weighted" ? "weighted" : "rrf",
          rrf_k: rrfK,
          candidate_k: candidateK,
          weights: { sparse: Math.round((1 - w) * 100) / 100, dense: w },
        },
        ...(rerank && key === mode ? { rerank: { enabled: true, model: RERANK_MODEL, candidate_k: poolK } } : {}),
      },
    });
    if (!data) throw new Error(errorMessage(error, response.status));
    return data;
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || denseBlocked || hybridError || rerankError) return;
    setRun({ kind: "busy" });
    try {
      if (mode === "compare") {
        const [sparse, denseRun, rrf, weighted] = await Promise.all(
          (["sparse", "dense", "rrf", "weighted"] as const).map(retrieve),
        );
        setRun({ kind: "compare", runs: { sparse, dense: denseRun, rrf, weighted } });
      } else {
        setRun({ kind: "single", response: await retrieve(mode) });
      }
    } catch (err) {
      setRun({ kind: "error", message: err instanceof TypeError ? "API unreachable" : (err as Error).message });
    }
  };

  const provenance = run.kind === "single" ? [run.response.provenance] : [];

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[360px_1fr]">
      <div className="space-y-5">
        <Panel>
          <PanelHeader eyebrow="Query" title="Retrieval request" actions={denseState && <DenseStateBadge state={denseState} />} />
          <form onSubmit={submit} className="space-y-4 p-4">
            <div className="grid grid-cols-[1fr_96px] gap-3">
              <FormField label="Corpus" htmlFor="rl-corpus">
                <select id="rl-corpus" className={select} value={summary.corpus.id} onChange={(e) => onPickCorpus(e.target.value)}>
                  {corpora.map((c) => (
                    <option key={c.corpus.id} value={c.corpus.id}>{c.corpus.name}</option>
                  ))}
                </select>
              </FormField>
              <FormField label="Version" htmlFor="rl-version">
                <select id="rl-version" className={select} value={version ?? "current"} onChange={(e) => setVersion(e.target.value === "current" ? null : Number(e.target.value))}>
                  <option value="current">v{summary.corpus.version} ●</option>
                  {Array.from({ length: summary.corpus.version }, (_, i) => summary.corpus.version - 1 - i).map((v) => (
                    <option key={v} value={v}>v{v}</option>
                  ))}
                </select>
              </FormField>
            </div>
            <p className="-mt-2 font-mono text-[10px] text-fg-subtle">
              {summary.stats.document_count} docs · {summary.stats.chunk_count} chunks in current version
            </p>
            <div className="space-y-1.5">
              <span className="block font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">Strategy</span>
              <Segmented name="strategy" value={mode} options={MODES} onChange={setMode} />
              <p className="text-[11px] leading-relaxed text-fg-subtle">{MODE_HELP[mode]}</p>
            </div>
            <FormField label="Query" htmlFor="rl-query">
              <Input id="rl-query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. how does hybrid retrieval fuse rankings?" maxLength={2000} autoFocus />
            </FormField>
            <details className="group rounded-md border border-line">
              <summary className="cursor-pointer select-none px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted marker:text-fg-subtle">
                Parameters · top {topK}
                {mode !== "dense" && ` · k1 ${k1} · b ${b}`}
                {usesHybrid && ` · rrf k ${rrfK} · cand ${candidateK}`}
              </summary>
              <div className="grid grid-cols-3 gap-3 border-t border-line p-3">
                <FormField label="Top k" htmlFor="rl-k">
                  <Input id="rl-k" type="number" min={1} max={100} value={topK} onChange={(e) => setTopK(e.target.valueAsNumber || 1)} />
                </FormField>
                <FormField label="k1" htmlFor="rl-k1" hint="BM25">
                  <Input id="rl-k1" type="number" min={0} max={3} step={0.1} value={k1} onChange={(e) => setK1(e.target.valueAsNumber || 0)} />
                </FormField>
                <FormField label="b" htmlFor="rl-b" hint="BM25">
                  <Input id="rl-b" type="number" min={0} max={1} step={0.05} value={b} onChange={(e) => setB(e.target.valueAsNumber || 0)} />
                </FormField>
              </div>
              {usesHybrid && (
                <div className="grid grid-cols-2 gap-3 border-t border-line p-3">
                  <FormField label="RRF k" htmlFor="rl-rrfk" hint="rank smoothing">
                    <Input id="rl-rrfk" type="number" min={1} max={1000} value={rrfK} onChange={(e) => setRrfK(e.target.valueAsNumber || 1)} />
                  </FormField>
                  <FormField label="Candidates" htmlFor="rl-cand" hint="per retriever" error={hybridError}>
                    <Input id="rl-cand" type="number" min={1} max={200} value={candidateK} aria-invalid={!!hybridError} onChange={(e) => setCandidateK(e.target.valueAsNumber || 1)} />
                  </FormField>
                </div>
              )}
            </details>
            <RerankControls
              on={rerankOn}
              onToggle={setRerankOn}
              poolK={poolK}
              onPoolK={setPoolK}
              topK={topK}
              disabled={mode === "compare"}
              error={rerankError}
            />
            {(mode === "weighted" || mode === "compare") && (
              <div className="space-y-1.5">
                <label htmlFor="rl-weight" className="flex justify-between font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">
                  <span>Weights (weighted fusion)</span>
                  <span className="num normal-case tracking-normal">
                    <span className="text-s-sparse">BM25 {(1 - denseWeight).toFixed(2)}</span> · <span className="text-s-dense">dense {denseWeight.toFixed(2)}</span>
                  </span>
                </label>
                <input
                  id="rl-weight"
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={denseWeight}
                  onChange={(e) => setDenseWeight(e.target.valueAsNumber)}
                  className="w-full accent-[var(--color-s-dense)]"
                  aria-valuetext={`BM25 ${(1 - denseWeight).toFixed(2)}, dense ${denseWeight.toFixed(2)}`}
                />
              </div>
            )}
            {denseBlocked && (
              <p role="status" className="rounded-md border border-line-strong bg-surface-2 px-3 py-2 text-[11px] leading-relaxed text-fg-muted">
                Dense retrieval is unavailable for this version:{" "}
                {denseState ? DENSE_STATE[denseState].help : "checking index status…"}
              </p>
            )}
            <div className="flex items-center justify-end">
              <Button type="submit" variant="primary" disabled={!query.trim() || run.kind === "busy" || denseBlocked || !!hybridError || !!rerankError}>
                <ScanSearch className="size-4" /> {run.kind === "busy" ? "Retrieving…" : mode === "compare" ? "Compare" : "Retrieve"}{" "}
                <Kbd className="border-black/20 bg-black/10 text-black/70">↵</Kbd>
              </Button>
            </div>
          </form>
        </Panel>
        {(needsDense || denseState === "building") && denseState !== "ready" && <DenseIndexPanel handle={dense} />}
        {provenance.map((p) => <ProvenancePanel key={p.strategy} p={p} />)}
        {run.kind === "compare" && <RunConfigurations runs={run.runs} />}
      </div>

      <div className="min-w-0">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={run.kind === "single" ? run.response.query.id : run.kind === "compare" ? run.runs.rrf.query.id : run.kind} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
            {run.kind === "compare" ? <CompareView runs={run.runs} /> : <Results run={run} corpusId={summary.corpus.id} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function Results({ run, corpusId }: { run: Exclude<Run, { kind: "compare" }>; corpusId: string }) {
  if (run.kind === "idle")
    return (
      <Panel>
        <EmptyState icon={ScanSearch} title="Run a query">
          Retrieved chunks appear here, ranked by the selected strategy, with their document, version and chunk identity.
          Compare runs BM25 and dense on the same query and shows where they agree.
        </EmptyState>
      </Panel>
    );
  if (run.kind === "busy") return <Panel><LoadingState rows={6} label="Retrieving" /></Panel>;
  if (run.kind === "error") return <Panel><ErrorState title="Retrieval failed">{run.message}</ErrorState></Panel>;

  const { hits, provenance, warnings, reranking } = run.response;
  const top = hits[0]?.result.score ?? 0;
  const pool = reranking?.candidates;
  const scores = pool?.map((c) => c.rerank.reranker_score) ?? [];
  const range: [number, number] | undefined = pool ? [Math.min(...scores), Math.max(...scores)] : undefined;
  const dense = provenance.strategy === "dense";
  const hybrid = provenance.configuration.hybrid;
  return (
    <section aria-label="Retrieved evidence" className="min-w-0 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-[13px] font-medium">
          <span className="num">{hits.length}</span> retrieved chunk{hits.length === 1 ? "" : "s"}
          <span className="ml-2 font-mono text-[11px] font-normal text-fg-subtle">
            {hybrid ? `fused from ${provenance.statistics.fused_candidates} candidates (${hybrid.fusion})` : dense ? "by cosine similarity" : `of ${provenance.statistics.matched_chunks ?? 0} matching`} ·{" "}
            {provenance.statistics.candidate_chunks} searched ·{" "}
            {provenance.reranking && `reranked ${provenance.reranking.candidates_scored} · `}
            {provenance.elapsed_ms.toFixed(1)} ms
          </span>
        </h2>
        <OriginBadge origin="retrieved" />
      </div>
      {warnings.map((w) => (
        <p key={w} role="status" className="rounded-md border border-signal/30 bg-signal-dim/40 px-3 py-2 text-xs text-signal">{w}</p>
      ))}
      {provenance.reranking && hits.length > 0 && <RerankAnalysis response={run.response} />}
      {hits.length === 0 ? (
        <Panel>
          <EmptyState icon={SearchX} title="No matching chunks">
            {provenance.corpus_version === 0
              ? "Version 0 is the empty corpus created before any ingestion."
              : "No chunk in this corpus version contains any query term. BM25 matches exact terms (no stemming or synonyms); try Dense for semantic matches."}
          </EmptyState>
        </Panel>
      ) : (
        hits.map((h, i) => <EvidenceCard key={h.result.chunk_id} hit={h} topScore={top} corpusId={corpusId} index={i} rrfK={hybrid?.rrf_k} pool={pool} scoreRange={range} />)
      )}
    </section>
  );
}

function ProvenancePanel({ p }: { p: RetrievalProvenance }) {
  const s = p.statistics;
  const c = p.retriever_config;
  const dense = p.strategy === "dense";
  const hybrid = p.configuration.hybrid;
  const fusion = c.fusion as Record<string, unknown> | undefined;
  return (
    <Panel>
      <PanelHeader eyebrow={`Provenance · ${hybrid ? `hybrid ${hybrid.fusion}` : dense ? "dense" : "BM25"}${p.reranking ? " + rerank" : ""}`} title="How this ranking was produced" actions={<Badge tone="trace">v{p.corpus_version}</Badge>} />
      {!dense && p.query_terms.length > 0 && (
        <div className="border-b border-line px-3 py-2.5">
          <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">Query terms</div>
          <div className="flex flex-wrap gap-1">
            {p.query_terms.length ? p.query_terms.map((t) => <Badge key={t}>{t}</Badge>) : <span className="text-[11px] text-fg-subtle">none</span>}
          </div>
        </div>
      )}
      <dl className="divide-y divide-line">
        <Field label="Retriever"><span className="font-mono">{p.strategy} · {p.retriever}</span></Field>
        {hybrid && fusion ? (
          <>
            <Field label="Fusion"><span className="font-mono">{String(fusion.method)}{fusion.k !== undefined && ` · k ${String(fusion.k)}`}</span></Field>
            {hybrid.fusion === "weighted" && (
              <Field label="Weights"><span className="num font-mono">{Object.entries(hybrid.weights ?? {}).map(([s, w]) => `${s === "sparse" ? "BM25" : "dense"} ${w}`).join(" · ")}</span></Field>
            )}
            {fusion.normalization !== undefined && <Field label="Normalisation"><span className="font-mono">{String(fusion.normalization)}</span></Field>}
            <Field label="Components"><span className="font-mono">{hybrid.retrievers.join(" + ")} · {hybrid.candidate_k} each</span></Field>
            <Field label="Fused candidates"><span className="num">{s.fused_candidates}</span></Field>
            <Field label="Dense index"><span className="font-mono" title={p.index_id ?? ""}>{p.index_id}</span></Field>
            <Field label="Model"><span className="font-mono">{p.configuration.embedder?.model}</span></Field>
          </>
        ) : dense ? (
          <>
            <Field label="Model"><span className="font-mono" title={String(c.model)}>{String(c.model)}</span></Field>
            <Field label="Revision"><span className="font-mono" title={String(c.revision)}>{shortHash(String(c.revision), 10)}</span></Field>
            <Field label="Vector"><span className="font-mono">{String(c.dimension)}d · {String(c.similarity)} · {String(c.pooling)}</span></Field>
            <Field label="Index"><span className="font-mono" title={p.index_id ?? ""}>{p.index_id}</span></Field>
            <Field label="Query embedding"><span className="num">{s.query_embedding_ms} ms</span></Field>
          </>
        ) : (
          <>
            <Field label="k1 · b"><span className="num font-mono">{String(c.k1)} · {String(c.b)}</span></Field>
            <Field label="Analyzer"><span className="font-mono" title={String(c.analyzer)}>{String(c.analyzer)}</span></Field>
            <Field label="Avg chunk length"><span className="num">{s.avg_chunk_length} tokens</span></Field>
            <Field label="Indexed now"><span className="num">{s.indexed_now}</span></Field>
          </>
        )}
        {p.reranking && <RerankProvenanceFields r={p.reranking} />}
        <Field label="Configuration"><span className="font-mono text-trace" title={p.configuration_hash}>{shortHash(p.configuration_hash, 12)}</span></Field>
        <Field label="Chunking hash"><span className="font-mono" title={p.chunking_hash}>{shortHash(p.chunking_hash, 12)}</span></Field>
        <Field label="Chunks searched"><span className="num">{s.candidate_chunks}</span></Field>
        <Field label="Elapsed"><span className="num">{p.elapsed_ms.toFixed(1)} ms</span></Field>
        <Field label="Git commit"><span className="font-mono">{p.environment.git_commit ? shortHash(p.environment.git_commit, 12) : "unavailable"}</span></Field>
      </dl>
    </Panel>
  );
}

/** In compare mode: the exact configuration behind each of the four runs. */
function RunConfigurations({ runs }: { runs: Required<Runs> }) {
  return (
    <Panel>
      <PanelHeader eyebrow="Provenance" title="Run configurations" />
      <dl className="divide-y divide-line">
        {(["sparse", "dense", "rrf", "weighted"] as const).map((key) => {
          const p = runs[key].provenance;
          return (
            <Field key={key} label={RUN_LABEL[key]}>
              <span className="font-mono text-trace" title={`configuration ${p.configuration_hash}`}>{shortHash(p.configuration_hash, 12)}</span>
            </Field>
          );
        })}
      </dl>
      <p className="border-t border-line px-3 py-2 text-[10px] leading-relaxed text-fg-subtle">
        Same corpus version and query for all four; each hash identifies one complete retrieval configuration.
      </p>
    </Panel>
  );
}

function RerankControls({
  on,
  onToggle,
  poolK,
  onPoolK,
  topK,
  disabled,
  error,
}: {
  on: boolean;
  onToggle: (on: boolean) => void;
  poolK: number;
  onPoolK: (k: number) => void;
  topK: number;
  disabled: boolean;
  error: string | null;
}) {
  const active = on && !disabled;
  return (
    <div className="space-y-2 rounded-md border border-line p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">
          <Layers className="size-3.5 text-s-rerank" /> Rerank
        </span>
        <Segmented
          name="rerank"
          value={active ? "on" : "off"}
          options={[
            { value: "off", label: "Off" },
            { value: "on", label: "Cross-encoder" },
          ]}
          onChange={(v) => !disabled && onToggle(v === "on")}
        />
      </div>
      {disabled ? (
        <p className="text-[11px] leading-relaxed text-fg-subtle">Reranking analysis runs on one strategy at a time; pick BM25, Dense, RRF or Weighted.</p>
      ) : active ? (
        <>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Candidate pool" htmlFor="rl-pool" hint="upstream results scored" error={error}>
              <Input id="rl-pool" type="number" min={1} max={200} value={poolK} aria-invalid={!!error} onChange={(e) => onPoolK(e.target.valueAsNumber || 1)} />
            </FormField>
            <div className="space-y-1.5">
              <span className="block font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">Final top k</span>
              <div className="num flex h-9 items-center rounded-md border border-line bg-surface-2 px-3 font-mono text-[13px] text-fg-muted">{topK}</div>
              <p className="text-[11px] text-fg-subtle">set by Top k</p>
            </div>
          </div>
          <p className="font-mono text-[10px] leading-relaxed text-fg-subtle">
            retrieve {poolK} → score {poolK} (query, chunk) pairs with {RERANK_MODEL.split("/")[1]} → keep {topK}
          </p>
        </>
      ) : (
        <p className="text-[11px] leading-relaxed text-fg-subtle">Re-score the retrieved candidates with a local cross-encoder and inspect how the ranking moves.</p>
      )}
    </div>
  );
}

function RerankProvenanceFields({ r }: { r: NonNullable<RetrievalProvenance["reranking"]> }) {
  return (
    <>
      <Field label="Reranker"><span className="font-mono" title={r.info.spec.model}>{r.reranker} · {r.info.spec.model}</span></Field>
      <Field label="Reranker revision"><span className="font-mono" title={r.info.spec.revision}>{shortHash(r.info.spec.revision, 10)}</span></Field>
      <Field label="Weights sha256"><span className="font-mono" title={r.info.weights_sha256}>{shortHash(r.info.weights_sha256, 12)}</span></Field>
      <Field label="Scoring"><span className="font-mono">{r.info.scoring} · max {r.info.max_seq_length} tok</span></Field>
      <Field label="Pool → top k"><span className="num font-mono">{r.candidates_scored} scored of {r.candidate_k} → {r.final_top_k}</span></Field>
      <Field label="Rerank latency"><span className="num">{r.latency_ms.toFixed(1)} ms</span></Field>
      <Field label="Upstream config"><span className="font-mono" title={r.upstream_configuration_hash}>{shortHash(r.upstream_configuration_hash, 12)}</span></Field>
      <Field label="Reranker config"><span className="font-mono" title={r.reranker_config_hash}>{shortHash(r.reranker_config_hash, 12)}</span></Field>
    </>
  );
}
