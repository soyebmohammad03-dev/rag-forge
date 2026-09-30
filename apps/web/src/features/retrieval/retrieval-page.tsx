"use client";

import type { CorpusSummary, RetrievalProvenance, RetrievalResponse, RetrievalStrategy } from "@rag-forge/shared";
import { Database, ScanSearch, SearchX } from "lucide-react";
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
import { CompareView } from "./compare-view";
import { DENSE_STATE, DenseIndexPanel, DenseStateBadge, useDenseIndex } from "./dense-index";
import { EvidenceCard } from "./evidence-card";
import { errorMessage } from "./errors";

type Mode = "sparse" | "dense" | "compare";
type Run =
  | { kind: "idle" }
  | { kind: "busy" }
  | { kind: "single"; response: RetrievalResponse }
  | { kind: "compare"; sparse: RetrievalResponse; dense: RetrievalResponse }
  | { kind: "error"; message: string };

const MODES: { value: Mode; label: string }[] = [
  { value: "sparse", label: "BM25" },
  { value: "dense", label: "Dense" },
  { value: "compare", label: "Compare" },
];

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
          Run a query against one corpus version with BM25 (exact terms) or dense retrieval (semantic similarity), or
          both side by side. This is inspection, not evaluation: no quality metrics are computed here.
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
  const [run, setRun] = useState<Run>({ kind: "idle" });
  const dense = useDenseIndex(summary.corpus.id, version);
  const denseState = dense.data?.state;
  const needsDense = mode !== "sparse";
  const denseBlocked = needsDense && denseState !== "ready";

  const retrieve = async (strategy: RetrievalStrategy) => {
    const { data, error, response } = await api.POST("/api/v1/corpora/{corpus_id}/retrieve", {
      params: { path: { corpus_id: summary.corpus.id } },
      body: { query, top_k: topK, version, strategy, bm25: { k1, b } },
    });
    if (!data) throw new Error(errorMessage(error, response.status));
    return data;
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || denseBlocked) return;
    setRun({ kind: "busy" });
    try {
      if (mode === "compare") {
        const [sparse, denseRun] = await Promise.all([retrieve("sparse"), retrieve("dense")]);
        setRun({ kind: "compare", sparse, dense: denseRun });
      } else {
        setRun({ kind: "single", response: await retrieve(mode) });
      }
    } catch (err) {
      setRun({ kind: "error", message: err instanceof TypeError ? "API unreachable" : (err as Error).message });
    }
  };

  const provenance = run.kind === "single" ? [run.response.provenance] : run.kind === "compare" ? [run.sparse.provenance, run.dense.provenance] : [];

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
            </div>
            <FormField label="Query" htmlFor="rl-query">
              <Input id="rl-query" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. how does hybrid retrieval fuse rankings?" maxLength={2000} autoFocus />
            </FormField>
            <details className="group rounded-md border border-line">
              <summary className="cursor-pointer select-none px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted marker:text-fg-subtle">
                Parameters · top {topK} · k1 {k1} · b {b}
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
            </details>
            {denseBlocked && (
              <p role="status" className="rounded-md border border-line-strong bg-surface-2 px-3 py-2 text-[11px] leading-relaxed text-fg-muted">
                Dense retrieval is unavailable for this version:{" "}
                {denseState ? DENSE_STATE[denseState].help : "checking index status…"}
              </p>
            )}
            <div className="flex items-center justify-end">
              <Button type="submit" variant="primary" disabled={!query.trim() || run.kind === "busy" || denseBlocked}>
                <ScanSearch className="size-4" /> {run.kind === "busy" ? "Retrieving…" : mode === "compare" ? "Compare" : "Retrieve"}{" "}
                <Kbd className="border-black/20 bg-black/10 text-black/70">↵</Kbd>
              </Button>
            </div>
          </form>
        </Panel>
        {(needsDense || denseState === "building") && denseState !== "ready" && <DenseIndexPanel handle={dense} />}
        {provenance.map((p) => <ProvenancePanel key={p.strategy} p={p} />)}
      </div>

      <div className="min-w-0">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={run.kind === "single" ? run.response.query.id : run.kind === "compare" ? run.dense.query.id : run.kind} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
            {run.kind === "compare" ? <CompareView sparse={run.sparse} dense={run.dense} /> : <Results run={run} corpusId={summary.corpus.id} />}
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

  const { hits, provenance, warnings } = run.response;
  const top = hits[0]?.result.score ?? 0;
  const dense = provenance.strategy === "dense";
  return (
    <section aria-label="Retrieved evidence" className="min-w-0 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-[13px] font-medium">
          <span className="num">{hits.length}</span> retrieved chunk{hits.length === 1 ? "" : "s"}
          <span className="ml-2 font-mono text-[11px] font-normal text-fg-subtle">
            {dense ? "by cosine similarity" : `of ${provenance.statistics.matched_chunks ?? 0} matching`} · {provenance.statistics.candidate_chunks} searched ·{" "}
            {provenance.elapsed_ms.toFixed(1)} ms
          </span>
        </h2>
        <OriginBadge origin="retrieved" />
      </div>
      {warnings.map((w) => (
        <p key={w} role="status" className="rounded-md border border-signal/30 bg-signal-dim/40 px-3 py-2 text-xs text-signal">{w}</p>
      ))}
      {hits.length === 0 ? (
        <Panel>
          <EmptyState icon={SearchX} title="No matching chunks">
            {provenance.corpus_version === 0
              ? "Version 0 is the empty corpus created before any ingestion."
              : "No chunk in this corpus version contains any query term. BM25 matches exact terms (no stemming or synonyms); try Dense for semantic matches."}
          </EmptyState>
        </Panel>
      ) : (
        hits.map((h, i) => <EvidenceCard key={h.result.chunk_id} hit={h} topScore={top} corpusId={corpusId} index={i} />)
      )}
    </section>
  );
}

function ProvenancePanel({ p }: { p: RetrievalProvenance }) {
  const s = p.statistics;
  const c = p.retriever_config;
  const dense = p.strategy === "dense";
  return (
    <Panel>
      <PanelHeader eyebrow={`Provenance · ${dense ? "dense" : "BM25"}`} title="How this ranking was produced" actions={<Badge tone="trace">v{p.corpus_version}</Badge>} />
      {!dense && (
        <div className="border-b border-line px-3 py-2.5">
          <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">Query terms</div>
          <div className="flex flex-wrap gap-1">
            {p.query_terms.length ? p.query_terms.map((t) => <Badge key={t}>{t}</Badge>) : <span className="text-[11px] text-fg-subtle">none</span>}
          </div>
        </div>
      )}
      <dl className="divide-y divide-line">
        <Field label="Retriever"><span className="font-mono">{p.strategy} · {p.retriever}</span></Field>
        {dense ? (
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
        <Field label="Config hash"><span className="font-mono text-trace" title={p.retriever_config_hash}>{shortHash(p.retriever_config_hash, 12)}</span></Field>
        <Field label="Chunking hash"><span className="font-mono" title={p.chunking_hash}>{shortHash(p.chunking_hash, 12)}</span></Field>
        <Field label="Chunks searched"><span className="num">{s.candidate_chunks}</span></Field>
        <Field label="Elapsed"><span className="num">{p.elapsed_ms.toFixed(1)} ms</span></Field>
        <Field label="Git commit"><span className="font-mono">{p.environment.git_commit ? shortHash(p.environment.git_commit, 12) : "unavailable"}</span></Field>
      </dl>
    </Panel>
  );
}
