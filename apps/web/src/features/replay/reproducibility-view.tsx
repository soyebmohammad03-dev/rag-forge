"use client";

import type { ExperimentRun, ReproducibilityManifest } from "@rag-forge/shared";
import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { useCallback } from "react";
import { Badge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { formatBytes, shortHash } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { armColor } from "@/features/arena/format";

/** Which stages are deterministic, as the pipeline records them. Generation depends on the arm. */
export const STAGE_DETERMINISM: { stage: string; deterministic: boolean | "arm"; note: string }[] = [
  { stage: "query", deterministic: true, note: "hash of the query text" },
  { stage: "query_analysis", deterministic: true, note: "heuristic analyzer; corpus term statistics of the pinned version" },
  { stage: "router_decision", deterministic: true, note: "rule policy over the analysis" },
  { stage: "retrieval", deterministic: true, note: "BM25 / exact cosine / fusion over the pinned version" },
  { stage: "reranking", deterministic: true, note: "cross-encoder scores; may differ across CPUs or onnxruntime builds" },
  { stage: "evidence_selection", deterministic: true, note: "greedy selection under recorded budgets" },
  { stage: "context", deterministic: true, note: "versioned prompt template over the selected evidence" },
  { stage: "generation", deterministic: "arm", note: "greedy local decoding is deterministic; sampling and remote endpoints are not" },
  { stage: "claims", deterministic: true, note: "rule-based extraction from the generated text" },
  { stage: "grounding", deterministic: true, note: "lexical-semantic verifier with recorded thresholds" },
];

export function useManifest(runId: string) {
  const fetcher = useCallback(() => api.GET("/api/v1/runs/{run_id}/manifest", { params: { path: { run_id: runId } } }), [runId]);
  return useApi(fetcher);
}

export function ReproducibilityView({ run, part }: { run: ExperimentRun; part: "run" | "reproducibility" }) {
  const m = useManifest(run.id);
  if (!m.data) return <Panel>{m.error ? <ErrorState title="Could not load the manifest">{m.error}</ErrorState> : <LoadingState rows={6} label="Building the manifest" />}</Panel>;
  return part === "run" ? <RunFacts run={run} m={m.data} /> : <Reproducibility run={run} m={m.data} />;
}

function Facts({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="divide-y divide-line text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[140px_1fr] gap-3 px-4 py-2">
          <dt className="text-fg-subtle">{k}</dt>
          <dd className="min-w-0 break-words font-mono text-fg">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function RunFacts({ run, m }: { run: ExperimentRun; m: ReproducibilityManifest }) {
  const r = m.run as Record<string, unknown>;
  const d = m.dataset as Record<string, unknown>;
  const env = m.environment;
  return (
    <div className="grid items-start gap-5 xl:grid-cols-2">
      <Panel>
        <PanelHeader eyebrow="Experiment" title={String(r.experiment)} />
        {r.hypothesis ? <p className="border-b border-line px-4 py-3 text-[12px] text-fg-muted"><span className="text-fg">Hypothesis:</span> {String(r.hypothesis)}</p> : null}
        <Facts
          rows={[
            ["Status", `${String(r.status)} · ${String(r.completed)}/${String(r.evaluations)} evaluated · ${String(r.failed)} failed`],
            ["Recorded", `${String(r.created_at)} → ${String(r.finished_at ?? "—")}`],
            ["Dataset", <span key="d">{String(d.name)} v{String(d.version)} <Badge tone={d.source === "development" ? "signal" : "neutral"}>{String(d.source)}</Badge></span>],
            ["Dataset hash", String(d.content_hash)],
            ["Corpus", `${String(d.corpus_id)} v${String(d.corpus_version)}`],
            ["Chunking hash", String(d.chunking_hash)],
            ["Cases", `${run.case_ids.length} of ${String(d.cases)}`],
          ]}
        />
      </Panel>
      <Panel>
        <PanelHeader eyebrow="Environment" title="Recorded when the run was created" />
        <Facts
          rows={[
            ["Git commit", (m.runtime?.git_commit ?? env.git_commit ?? "not recorded") + (m.runtime?.git_dirty ? " (uncommitted changes)" : "")],
            ["RAG FORGE", env.rag_forge_version],
            ["Python", env.python_version],
            ["Platform", env.platform],
            ["Node / uv", m.runtime ? `${m.runtime.node_version ?? "not found"} / ${m.runtime.uv_version ?? "not found"}` : "not recorded for this run"],
            ["Packages", Object.entries(m.runtime?.packages ?? env.packages).map(([k, v]) => `${k} ${v}`).join(", ")],
          ]}
        />
      </Panel>
      <Panel className="xl:col-span-2">
        <PanelHeader eyebrow="Configurations" title="One immutable snapshot per arm" />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[11px]">
            <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
              <tr className="border-b border-line">
                <th className="px-4 py-2 font-normal">arm</th><th className="font-normal">pipeline</th><th className="font-normal">config hash</th><th className="font-normal">retrieval / routing hash</th><th className="font-normal">prompt</th><th className="font-normal">generator</th><th className="px-4 font-normal">verifier</th>
              </tr>
            </thead>
            <tbody>
              {m.arms.map((a, i) => (
                <tr key={a.arm} className="border-b border-line/60 font-mono last:border-0">
                  <td className="px-4 py-2"><span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full" style={{ background: armColor(i) }} />{a.arm}</span></td>
                  <td>{a.pipeline}{a.retrieval_mode === "adaptive" && " · adaptive"}</td>
                  <td title={a.config_hash}>{shortHash(a.config_hash, 16)}</td>
                  <td className="text-fg-muted" title={a.retrieval_configuration_hash ?? a.routing_hash ?? ""}>{shortHash(a.retrieval_configuration_hash ?? a.routing_hash ?? "—", 16)}</td>
                  <td className="text-fg-muted">{a.prompt_template ?? "—"}</td>
                  <td className="text-fg-muted">{a.generator ? `${a.generator} · T=${a.temperature} · seed ${a.seed}` : "—"}</td>
                  <td className="px-4 text-fg-muted">{a.verifier ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function Reproducibility({ run, m }: { run: ExperimentRun; m: ReproducibilityManifest }) {
  const fetchChecks = useCallback(() => api.GET("/api/v1/runs/{run_id}/replayability", { params: { path: { run_id: run.id } } }), [run.id]);
  const checks = useApi(fetchChecks);
  const differences = (m.current as { differences?: string[] }).differences ?? [];
  const generative = m.arms.filter((a) => a.generator);
  return (
    <div className="grid items-start gap-5 xl:grid-cols-2">
      <Panel>
        <PanelHeader eyebrow="Determinism" title="Which stages must reproduce exactly" />
        <ul className="divide-y divide-line/60 text-[12px]">
          {STAGE_DETERMINISM.map((s) => (
            <li key={s.stage} className="flex items-start gap-2.5 px-4 py-2">
              {s.deterministic === true ? <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-ok" /> : <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-signal" />}
              <span className="min-w-0">
                <span className="font-mono text-fg">{s.stage}</span>
                <span className="block text-[11px] text-fg-subtle">{s.note}</span>
                {s.deterministic === "arm" && generative.length > 0 && (
                  <span className="mt-1 flex flex-wrap gap-1">
                    {generative.map((a) => (
                      <Badge key={a.arm} tone={a.generation_deterministic ? "ok" : "signal"} title={`temperature ${a.temperature}, seed ${a.seed}`}>
                        {a.arm}: {a.generation_deterministic === null ? "not recorded" : a.generation_deterministic ? "deterministic" : "non-deterministic"}
                      </Badge>
                    ))}
                  </span>
                )}
              </span>
            </li>
          ))}
        </ul>
      </Panel>
      <Panel>
        <PanelHeader eyebrow="Replay availability" title="Each arm resolved again in this environment" />
        {!checks.data ? (
          checks.error ? <ErrorState title="Could not check">{checks.error}</ErrorState> : <LoadingState rows={4} label="Resolving arms" />
        ) : (
          <ul className="divide-y divide-line/60 text-[12px]">
            {checks.data.map((c) => (
              <li key={c.arm} className="px-4 py-2.5">
                <div className="flex items-center gap-2">
                  {c.replayable ? <CheckCircle2 className="size-3.5 text-ok" /> : <XCircle className="size-3.5 text-err" />}
                  <span className="font-mono text-fg">{c.arm}</span>
                  <Badge tone={c.replayable ? "ok" : "err"}>{c.replayable ? "replayable" : "not replayable"}</Badge>
                  <span className="ml-auto font-mono text-[10px] text-fg-subtle" title={c.current_hash ?? ""}>{c.current_hash === c.recorded_hash ? "hash matches" : c.current_hash ? "hash differs" : "cannot resolve"}</span>
                </div>
                {c.reason && <p className="mt-1 text-[11px] text-err">{c.reason}</p>}
                {c.changes.length > 0 && <p className="mt-1 font-mono text-[10px] text-fg-subtle">{c.changes.slice(0, 4).map((x) => x.path).join(", ")}{c.changes.length > 4 ? ` +${c.changes.length - 4}` : ""}</p>}
                {c.generation && <p className="mt-1 text-[11px] text-fg-subtle">{c.generation}</p>}
              </li>
            ))}
          </ul>
        )}
      </Panel>
      <Panel className="xl:col-span-2">
        <PanelHeader eyebrow="Models" title="Exact model files, from the Hugging Face cache" />
        {m.models.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[11px]">
              <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
                <tr className="border-b border-line"><th className="px-4 py-2 font-normal">role</th><th className="font-normal">model @ revision</th><th className="font-normal">file</th><th className="font-normal">sha256</th><th className="px-4 text-right font-normal">size</th></tr>
              </thead>
              <tbody>
                {m.models.flatMap((mr) =>
                  (mr.files.length ? mr.files : [null]).map((f, i) => (
                    <tr key={`${mr.role}-${mr.model}-${f?.path ?? "none"}`} className="border-b border-line/60 font-mono last:border-0">
                      <td className="px-4 py-1.5 text-fg-muted">{i === 0 ? mr.role : ""}</td>
                      <td className="text-fg">{i === 0 ? `${mr.model}@${(mr.revision ?? "—").slice(0, 10)}` : ""}</td>
                      <td className="text-fg-muted">{f?.path ?? "no model files"}</td>
                      <td className="text-fg-muted" title={f?.sha256 ?? ""}>{f?.sha256 ? shortHash(f.sha256, 16) : f?.source ?? "—"}</td>
                      <td className="px-4 text-right text-fg-subtle">{f?.bytes != null ? formatBytes(f.bytes) : "—"}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="p-4 text-[12px] text-fg-subtle">This run was recorded before model file capture existed. Its snapshots still pin every model by revision.</p>
        )}
      </Panel>
      <Panel>
        <PanelHeader eyebrow="Runtime" title="Toolchain, lockfiles and settings" />
        {m.runtime ? (
          <Facts
            rows={[
              ["Lockfiles", Object.entries(m.runtime.lockfiles).map(([k, v]) => `${k} ${shortHash(v, 12)}`).join(" · ")],
              ["Settings", Object.entries(m.runtime.settings).map(([k, v]) => `${k}=${v}`).join(" · ") || "defaults"],
              ["Seeds", `bootstrap ${String((m.seeds as Record<string, unknown>).bootstrap)}; generation ${generative.map((a) => `${a.arm} ${a.seed}`).join(", ") || "none"}`],
              ["Captured", m.runtime.captured_at],
            ]}
          />
        ) : (
          <p className="p-4 text-[12px] text-fg-subtle">Not recorded for this run (it predates runtime capture).</p>
        )}
      </Panel>
      <Panel>
        <PanelHeader eyebrow="Now vs then" title="This environment compared with the recording" />
        <div className="p-4 text-[12px]">
          {differences.length ? (
            <ul className="space-y-1">{differences.map((d) => <li key={d} className="flex gap-1.5 text-signal"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" />{d}</li>)}</ul>
          ) : (
            <p className="flex items-center gap-1.5 text-ok"><CheckCircle2 className="size-3.5" /> No differences detected.</p>
          )}
          <ul className="mt-3 space-y-1 text-[11px] text-fg-subtle">{m.notes.map((n) => <li key={n}>{n}</li>)}</ul>
          <p className="mt-3 font-mono text-[10px] text-fg-subtle">{m.manifest_version} · {m.artifacts.length} trace artifacts · {formatBytes(m.artifacts.reduce((t, a) => t + a.bytes, 0))}</p>
        </div>
      </Panel>
    </div>
  );
}
