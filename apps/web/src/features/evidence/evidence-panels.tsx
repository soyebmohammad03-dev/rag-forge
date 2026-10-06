"use client";

import type { Evidence, EvidenceSelection, GenerationContext, PipelineStage, RagResponse } from "@rag-forge/shared";
import { ArrowRight, FileText } from "lucide-react";
import Link from "next/link";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { cn } from "@/lib/cn";
import { shortHash } from "@/lib/format";
import { Field } from "@/features/corpus/bits";
import type { Focus } from "./answer-panels";
import { OUTCOME_LABEL } from "./grounding";

const STAGE_LABEL: Record<string, string> = {
  query: "Query",
  query_analysis: "Query intelligence",
  router_decision: "Router",
  retrieval: "Retrieval",
  reranking: "Reranking",
  evidence_selection: "Evidence",
  context: "Context",
  generation: "Generation",
  claims: "Claims",
  grounding: "Grounding",
};

const ORIGIN_COLOR: Record<string, string> = {
  retrieved: "var(--color-s-sparse)",
  inferred: "var(--color-s-metadata)",
  generated: "var(--color-s-graph)",
  measured: "var(--color-ok)",
};

/** The provenance chain as a data-flow strip: every stage, its output identity and its cost. */
export function PipelineTrace({ chain, total }: { chain: PipelineStage[]; total: number }) {
  const timed = chain.reduce((s, c) => s + (c.latency_ms ?? 0), 0);
  return (
    <Panel>
      <PanelHeader eyebrow="Pipeline" title="Provenance chain" actions={<span className="num font-mono text-[11px] text-fg-muted">{total.toFixed(0)} ms total</span>} />
      <ol className="flex flex-wrap items-stretch gap-y-2 p-3" aria-label="Pipeline stages">
        {chain.map((s, i) => (
          <li key={s.stage} className="flex items-center">
            <div
              className="min-w-[104px] rounded-md bg-surface-2 px-2.5 py-1.5 ring-1 ring-line"
              style={{ boxShadow: `inset 2px 0 0 ${ORIGIN_COLOR[s.origin] ?? "var(--color-line-strong)"}` }}
              title={`${s.detail}\noutput ${s.hash ?? "—"}\nconfig ${s.config_hash ?? "—"}\n${s.deterministic ? "deterministic" : "not deterministic"}`}
            >
              <div className="text-[12px] font-medium">{STAGE_LABEL[s.stage] ?? s.stage}</div>
              <div className="num font-mono text-[10px] text-fg-subtle">
                {s.latency_ms !== null ? `${s.latency_ms.toFixed(1)} ms` : "—"} · {s.hash ? shortHash(s.hash, 6) : "—"}
                {!s.deterministic && <span className="ml-1 text-signal">≈</span>}
              </div>
            </div>
            {i < chain.length - 1 && <ArrowRight className="mx-1 size-3 shrink-0 text-fg-subtle" />}
          </li>
        ))}
      </ol>
      {timed > 0 && (
        <div className="px-3 pb-3">
          <div className="flex h-2 overflow-hidden rounded-full bg-surface-3" aria-label="Latency by stage">
            {chain.filter((s) => s.latency_ms).map((s) => (
              <span key={s.stage} title={`${STAGE_LABEL[s.stage]}: ${s.latency_ms?.toFixed(1)} ms`} style={{ width: `${((s.latency_ms ?? 0) / timed) * 100}%`, background: ORIGIN_COLOR[s.origin] }} />
            ))}
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-3 font-mono text-[10px] text-fg-subtle">
            {Object.entries(ORIGIN_COLOR).map(([o, c]) => (
              <span key={o} className="flex items-center gap-1"><span className="size-2 rounded-sm" style={{ background: c }} />{o}</span>
            ))}
            <span>≈ not deterministic</span>
          </div>
        </div>
      )}
    </Panel>
  );
}

export function ContextBudget({ selection, context }: { selection: EvidenceSelection; context: GenerationContext | null }) {
  const max = selection.params.max_context_tokens;
  const used = context?.context_tokens ?? selection.tokens_used;
  return (
    <div className="space-y-1.5 border-b border-line px-4 py-3">
      <div className="flex justify-between font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">
        <span>Context budget</span>
        <span className="num normal-case tracking-normal">{used} / {max} tokens · {selection.selected.length} / {selection.params.max_items} passages</span>
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-surface-3">
        {selection.selected.map((e, i) => (
          <span key={e.id} title={`${e.citation}: ${e.token_count} tokens`} className={cn("h-full border-r border-bg", i % 2 ? "bg-trace/70" : "bg-trace")} style={{ width: `${(e.token_count / max) * 100}%` }} />
        ))}
      </div>
      <p className="font-mono text-[10px] text-fg-subtle">
        tokenizer {selection.tokenizer} · {selection.selector} · per-document cap {selection.params.max_per_document ?? "none"} · near-duplicate ≥ {selection.params.near_duplicate_threshold ?? "off"}
        {selection.params.min_score !== null && selection.params.min_score !== undefined && ` · min score ${selection.params.min_score}`}
      </p>
    </div>
  );
}

export function EvidencePanel({
  response,
  corpusId,
  focus,
  onFocus,
}: {
  response: RagResponse;
  corpusId: string;
  focus: Focus;
  onFocus: (f: Focus) => void;
}) {
  const { evidence, context, claims } = response;
  const supports = (id: string) => claims.filter((c) => c.supporting_evidence_ids.includes(id)).map((c) => c.index + 1);
  const cites = (id: string) => claims.filter((c) => c.cited_evidence_ids.includes(id)).map((c) => c.index + 1);
  return (
    <Panel>
      <PanelHeader eyebrow="Evidence" title={`${evidence.selected.length} passage${evidence.selected.length === 1 ? "" : "s"} selected from ${evidence.candidates} ranked`} actions={<OriginBadge origin="retrieved" />} />
      <ContextBudget selection={evidence} context={context} />
      {evidence.selected.length === 0 ? (
        <p className="px-4 py-6 text-center text-xs text-fg-subtle">No passage was selected. See the selection decisions below.</p>
      ) : (
        <ul className="divide-y divide-line">
          {evidence.selected.map((e) => (
            <EvidenceRow key={e.id} e={e} corpusId={corpusId} active={focus.evidence === e.id} supports={supports(e.id)} cites={cites(e.id)} onClick={() => onFocus({ claim: focus.claim, evidence: e.id })} />
          ))}
        </ul>
      )}
    </Panel>
  );
}

function EvidenceRow({ e, corpusId, active, supports, cites, onClick }: { e: Evidence; corpusId: string; active: boolean; supports: number[]; cites: number[]; onClick: () => void }) {
  const r = e.retrieval;
  return (
    <li data-evidence={e.id} onClick={onClick} className={cn("cursor-pointer px-4 py-3 transition-colors hover:bg-surface-2/60", active && "bg-trace-dim/30 shadow-[inset_2px_0_0_var(--color-trace)]")}>
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="trace">{e.citation}</Badge>
        <Link href={`/corpus/${corpusId}`} onClick={(ev) => ev.stopPropagation()} className="flex items-center gap-1 text-[13px] font-medium hover:text-trace">
          <FileText className="size-3.5 text-fg-subtle" />{e.filename}
        </Link>
        <span className="font-mono text-[10px] text-fg-subtle">doc v{e.document_version} · chunk {e.chunk_ordinal} · chars {e.char_start}–{e.char_end} · {e.token_count} tok</span>
        <span className="ml-auto flex gap-1">
          {supports.length > 0 && <Badge tone="ok" title="Claims the verifier measured this passage as supporting">supports {supports.join(", ")}</Badge>}
          {cites.length > 0 && <Badge title="Claims whose generated citation names this passage">cited by {cites.join(", ")}</Badge>}
        </span>
      </div>
      <p className={cn("mt-1.5 text-[12.5px] leading-relaxed text-fg-muted", !active && "line-clamp-3")}>{e.text}</p>
      <div className="mt-1.5 flex flex-wrap gap-x-3 font-mono text-[10px] text-fg-subtle">
        <span>{r.retriever} · final rank {r.rank} · score {r.score.toFixed(4)}</span>
        {r.upstream_rank !== null && <span>reranked from #{r.upstream_rank} ({r.upstream_score?.toFixed(4)})</span>}
        <span title={e.id}>{shortHash(e.id, 16)}</span>
        <span title={e.text_sha256}>sha {shortHash(e.text_sha256, 8)}</span>
      </div>
    </li>
  );
}

export function SelectionTable({ selection }: { selection: EvidenceSelection }) {
  return (
    <Panel>
      <PanelHeader eyebrow="Evidence selection" title="Every ranked candidate and why it was kept or skipped" />
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[12px]">
          <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
            <tr className="border-b border-line">
              <th className="px-4 py-2 font-normal">rank</th>
              <th className="font-normal">passage</th>
              <th className="font-normal">score</th>
              <th className="font-normal">outcome</th>
              <th className="pr-4 font-normal">detail</th>
            </tr>
          </thead>
          <tbody>
            {selection.decisions.map((d) => (
              <tr key={d.chunk_id} className="border-b border-line last:border-0">
                <td className="num px-4 py-1.5 font-mono">{d.rank}</td>
                <td className="max-w-[200px] truncate" title={d.chunk_id}>{d.filename}</td>
                <td className="num font-mono">{d.score.toFixed(4)}</td>
                <td><Badge tone={d.outcome === "selected" ? "ok" : "neutral"}>{OUTCOME_LABEL[d.outcome]}</Badge></td>
                <td className="pr-4 font-mono text-[11px] text-fg-muted">{d.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

export function RagProvenancePanel({ response }: { response: RagResponse }) {
  const p = response.provenance;
  const g = response.answer?.generation;
  const ctx = response.context;
  return (
    <Panel>
      <PanelHeader eyebrow="Provenance" title="What produced this answer" actions={<Badge tone="trace">v{response.retrieval.provenance.corpus_version}</Badge>} />
      <dl className="divide-y divide-line">
        <Field label="Pipeline config"><span className="font-mono text-trace" title={p.configuration_hash}>{shortHash(p.configuration_hash, 12)}</span></Field>
        <Field label="Retrieval config"><span className="font-mono" title={p.configuration.retrieval.strategy}>{response.retrieval.provenance.strategy} · {shortHash(response.retrieval.provenance.configuration_hash, 12)}</span></Field>
        <Field label="Evidence selection"><span className="font-mono" title={response.evidence.selection_hash}>{shortHash(response.evidence.selection_hash, 12)}</span></Field>
        {ctx && (
          <>
            <Field label="Context hash"><span className="font-mono text-trace" title={ctx.context_hash}>{shortHash(ctx.context_hash, 12)}</span></Field>
            <Field label="Prompt"><span className="font-mono" title={ctx.prompt_hash}>{ctx.prompt_template} · {shortHash(ctx.prompt_hash, 8)}</span></Field>
          </>
        )}
        {g && (
          <>
            <Field label="Generator"><span className="font-mono" title={g.generator.model}>{g.generator.model}</span></Field>
            <Field label="Provider"><span className="font-mono">{g.generator.provider}{g.generator.local ? " · local" : " · remote"}</span></Field>
            {g.generator.revision && <Field label="Revision"><span className="font-mono" title={g.generator.revision}>{shortHash(g.generator.revision, 10)}</span></Field>}
            {g.generator.weights_sha256 && <Field label="Weights sha256"><span className="font-mono" title={g.generator.weights_sha256}>{shortHash(g.generator.weights_sha256, 12)}</span></Field>}
            <Field label="Decoding"><span className="num font-mono">{g.params.temperature === 0 ? "greedy" : `T ${g.params.temperature} · top-p ${g.params.top_p} · seed ${g.params.seed}`} · max {g.params.max_new_tokens}</span></Field>
            <Field label="Tokens"><span className="num font-mono">{g.prompt_tokens ?? "?"} prompt · {g.completion_tokens ?? "?"} completion</span></Field>
            <Field label="Generation"><span className="num">{g.latency_ms.toFixed(0)} ms{g.load_ms > 0 && ` + ${g.load_ms.toFixed(0)} ms load`}</span></Field>
            <Field label="Reproducible"><span className={g.deterministic ? "text-ok" : "text-signal"}>{g.deterministic ? "yes (greedy, local)" : "no"}</span></Field>
            <Field label="Answer hash"><span className="font-mono" title={g.answer_hash}>{shortHash(g.answer_hash, 12)}</span></Field>
          </>
        )}
        {response.grounding && <Field label="Grounding config"><span className="font-mono" title={response.grounding.config_hash}>{shortHash(response.grounding.config_hash, 12)}</span></Field>}
        <Field label="Elapsed"><span className="num">{p.elapsed_ms.toFixed(0)} ms</span></Field>
        <Field label="Git commit"><span className="font-mono">{p.environment.git_commit ? shortHash(p.environment.git_commit, 12) : "unavailable"}</span></Field>
      </dl>
    </Panel>
  );
}
