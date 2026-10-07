"use client";

import type { ArtifactTrace, ExperimentRun, PipelineStage, RunCase } from "@rag-forge/shared";
import { CheckCircle2, CircleDashed, ExternalLink } from "lucide-react";
import { type ReactNode, useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { ErrorState, LoadingState } from "@/components/ui/states";
import { API_URL, api } from "@/lib/api";
import { shortHash } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { fmt } from "@/features/arena/format";

const STAGE_LABEL: Record<string, string> = {
  query: "Query",
  query_analysis: "Query analysis",
  router_decision: "Router decision",
  retrieval: "Retrieval",
  reranking: "Reranking",
  evidence_selection: "Evidence selection",
  context: "Context",
  generation: "Generation",
  claims: "Claims",
  grounding: "Grounding",
};

export function PipelineView({ run }: { run: ExperimentRun }) {
  const [arm, setArm] = useState(run.arms.find((a) => a.startsWith("rag")) ?? run.arms[0]);
  const [caseId, setCaseId] = useState(run.case_ids[0]);
  const fetchCases = useCallback(() => api.GET("/api/v1/runs/{run_id}/cases", { params: { path: { run_id: run.id }, query: { arm } } }), [run.id, arm]);
  const cases = useApi(fetchCases);
  const current = cases.data?.find((c) => c.case_id === caseId);
  return (
    <div className="space-y-5">
      <Panel>
        <PanelHeader
          eyebrow="Trace"
          title="One case through one configuration, stage by stage"
          actions={
            <span className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
              <select aria-label="Arm" value={arm} onChange={(e) => setArm(e.target.value)} className="h-7 rounded-md border border-line-strong bg-surface px-2">
                {run.arms.map((a) => <option key={a}>{a}</option>)}
              </select>
              <select aria-label="Case" value={caseId} onChange={(e) => setCaseId(e.target.value)} className="h-7 max-w-56 rounded-md border border-line-strong bg-surface px-2">
                {run.case_ids.map((c) => <option key={c}>{c}</option>)}
              </select>
            </span>
          }
        />
        {!cases.data ? (
          cases.error ? <ErrorState title="Could not load cases">{cases.error}</ErrorState> : <LoadingState rows={2} label="Loading cases" />
        ) : !current ? (
          <p className="p-4 text-[12px] text-fg-subtle">This case was not evaluated for this arm.</p>
        ) : current.status === "failed" ? (
          <p role="alert" className="p-4 text-[12px] text-err"><span className="font-mono">{current.error_type}</span>: {current.error}</p>
        ) : current.artifact_id ? (
          <Trace key={current.artifact_id} result={current} artifactId={current.artifact_id} />
        ) : (
          <p className="p-4 text-[12px] text-fg-subtle">No trace was recorded for this case.</p>
        )}
      </Panel>
    </div>
  );
}

function Trace({ result, artifactId }: { result: RunCase; artifactId: string }) {
  const fetcher = useCallback(() => api.GET("/api/v1/artifacts/{artifact_id}", { params: { path: { artifact_id: artifactId } } }), [artifactId]);
  const trace = useApi(fetcher);
  if (!trace.data) return trace.error ? <ErrorState title="Could not load the trace">{trace.error}</ErrorState> : <LoadingState rows={6} label="Loading the trace" />;
  const t = trace.data;
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-line px-4 py-2.5 font-mono text-[11px] text-fg-subtle">
        <span>config {shortHash(result.config_hash, 12)}</span>
        {result.retrieval_configuration_hash && <span>retrieval {shortHash(result.retrieval_configuration_hash, 12)}</span>}
        <span>artifact {t.artifact_id} · sha256 {shortHash(t.sha256, 12)}</span>
        <a href={`${API_URL}/api/v1/artifacts/${t.artifact_id}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-trace hover:underline">raw trace <ExternalLink className="size-3" /></a>
      </div>
      <ol className="relative space-y-0 p-4">
        {t.stages.map((s, i) => (
          <StageRow key={s.stage} stage={s} last={i === t.stages.length - 1}>
            <StageDetail stage={s.stage} trace={t} />
          </StageRow>
        ))}
        <StageRow stage={{ stage: "metrics", hash: null, config_hash: null, latency_ms: null, origin: "measured", deterministic: true, detail: `${result.metrics.filter((m) => m.value !== null).length} measured, ${result.metrics.filter((m) => m.skipped).length} skipped` }} last>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-0.5 font-mono text-[11px] sm:grid-cols-3 lg:grid-cols-4">
            {result.metrics.filter((m) => m.value !== null).map((m) => (
              <div key={m.metric} className="flex justify-between gap-2"><dt className="truncate text-fg-subtle">{m.metric}</dt><dd className="num">{fmt(m.value, m.metric)}</dd></div>
            ))}
          </dl>
        </StageRow>
      </ol>
    </div>
  );
}

function StageRow({ stage, last, children }: { stage: PipelineStage; last?: boolean; children?: ReactNode }) {
  return (
    <li className="relative grid grid-cols-[20px_1fr] gap-3 pb-4">
      {!last && <span className="absolute left-[9px] top-5 bottom-0 w-px bg-line" aria-hidden />}
      <span className="mt-0.5">{stage.deterministic ? <CheckCircle2 className="size-[18px] text-ok" aria-label="deterministic" /> : <CircleDashed className="size-[18px] text-signal" aria-label="not deterministic" />}</span>
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] font-medium text-fg">{STAGE_LABEL[stage.stage] ?? stage.stage}</span>
          <Badge>{stage.origin}</Badge>
          {!stage.deterministic && <Badge tone="signal">non-deterministic</Badge>}
          {stage.latency_ms != null && <span className="font-mono text-[10px] text-fg-subtle">{fmt(stage.latency_ms, "x_ms")}</span>}
          {stage.hash && <span className="ml-auto font-mono text-[10px] text-fg-subtle" title={`output ${stage.hash}\nconfig ${stage.config_hash ?? "—"}`}>out {shortHash(stage.hash, 10)}{stage.config_hash ? ` · cfg ${shortHash(stage.config_hash, 8)}` : ""}</span>}
        </div>
        <p className="text-[11px] text-fg-muted">{stage.detail}</p>
        {children && <div className="mt-2">{children}</div>}
      </div>
    </li>
  );
}

function StageDetail({ stage, trace }: { stage: string; trace: ArtifactTrace }) {
  const retrieval = trace.rag?.retrieval ?? trace.retrieval;
  const routing = retrieval?.provenance.routing;
  const rag = trace.rag;
  switch (stage) {
    case "query":
      return <p className="text-[13px] text-fg">&ldquo;{retrieval?.query.text}&rdquo;</p>;
    case "query_analysis":
      return routing ? (
        <div className="flex flex-wrap gap-1">
          {Object.entries(routing.analysis.labels).map(([k, v]) => <Badge key={k} tone="trace">{k.replace(/_/g, " ")}: {String(v)}</Badge>)}
        </div>
      ) : null;
    case "router_decision":
      return routing ? (
        <ul className="space-y-0.5 text-[11px] text-fg-muted">
          <li className="text-fg">chose <span className="font-mono text-signal">{routing.decision.option}</span>{routing.decision.margin != null && ` (margin ${routing.decision.margin.toFixed(2)})`}</li>
          {routing.decision.rationale.slice(0, 3).map((r) => <li key={r}>{r}</li>)}
        </ul>
      ) : null;
    case "retrieval":
    case "reranking":
      return retrieval ? (
        <ol className="space-y-0.5 font-mono text-[11px]">
          {retrieval.hits.slice(0, stage === "retrieval" ? 5 : 5).map((h) => (
            <li key={h.result.chunk_id} className="flex gap-2">
              <span className="w-5 text-right text-fg-subtle">{h.result.rank}</span>
              <span className="min-w-0 flex-1 truncate text-fg-muted">{h.filename}</span>
              {stage === "reranking" && h.rerank && h.rerank.original_rank !== h.result.rank && <span className={h.rerank.original_rank > h.result.rank ? "text-ok" : "text-err"}>from {h.rerank.original_rank}</span>}
              <span className="num text-fg-subtle">{h.result.score.toFixed(3)}</span>
            </li>
          ))}
        </ol>
      ) : null;
    case "evidence_selection":
      return rag ? (
        <ul className="space-y-0.5 text-[11px]">
          {rag.evidence.selected.map((e) => <li key={e.id} className="flex gap-2"><span className="font-mono text-trace">{e.citation}</span><span className="truncate text-fg-muted">{e.filename}</span><span className="text-fg-subtle">· {e.selection_reason}</span></li>)}
        </ul>
      ) : null;
    case "context":
      return rag?.context ? <p className="font-mono text-[11px] text-fg-subtle">{rag.context.blocks.length} blocks · {rag.context.context_tokens}/{rag.context.max_context_tokens} tokens · tokenizer {rag.context.tokenizer}</p> : null;
    case "generation":
      return rag?.answer ? (
        <div className="rounded-md bg-surface-2 p-2.5">
          <p className="whitespace-pre-wrap text-[12px] leading-relaxed text-fg">{rag.answer.text}</p>
          <p className="mt-1 font-mono text-[10px] text-fg-subtle">
            {rag.answer.generation.generator.model} · T={rag.answer.generation.params.temperature} · {rag.answer.generation.completion_tokens ?? "?"} tokens · {rag.answer.generation.deterministic ? "deterministic" : "not deterministic"}
          </p>
        </div>
      ) : null;
    case "claims":
      return rag ? (
        <ul className="space-y-0.5 text-[11px]">
          {rag.claims.slice(0, 6).map((c) => <li key={c.id} className="flex gap-2"><Badge tone={c.support === "supported" ? "ok" : c.support === "unsupported" ? "err" : "neutral"}>{c.support.replace("_", " ")}</Badge><span className="text-fg-muted">{c.text}</span></li>)}
        </ul>
      ) : null;
    case "grounding":
      return rag?.grounding ? (
        <p className="font-mono text-[11px] text-fg-muted">
          {rag.grounding.status} · score {fmt(rag.grounding.grounding_score, "x")} · {rag.grounding.supported} supported / {rag.grounding.weakly_supported} weak / {rag.grounding.unsupported} unsupported of {rag.grounding.factual_claims} factual
        </p>
      ) : null;
    default:
      return null;
  }
}
