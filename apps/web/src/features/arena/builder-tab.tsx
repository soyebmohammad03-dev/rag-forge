"use client";

import type { AblationSpec, Arm, ArmInput, DatasetSummary, ResolveResponse } from "@rag-forge/shared";
import { Database, Play, Plus, ScanEye, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FormField, Input, Textarea } from "@/components/ui/input";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { shortHash } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { errorMessage } from "@/features/retrieval/errors";
import { armColor } from "./format";

const fetchPresets = () => api.GET("/api/v1/arena/presets");
const DEFAULT_ARMS = ["bm25", "dense", "hybrid-rrf"];

/** Arms whose retrieval needs the corpus version's dense index. */
export const needsDense = (a: Arm) => a.retrieval.mode === "adaptive" || a.retrieval.strategy !== "sparse";

/** Parse "1, 3,5" into sorted unique ks, or null when anything is not a whole number in 1..100. */
export function parseKs(text: string): number[] | null {
  const parts = text.split(",").map((s) => s.trim()).filter(Boolean);
  if (!parts.length) return null;
  const ks = parts.map(Number);
  if (ks.some((k) => !Number.isInteger(k) || k < 1 || k > 100)) return null;
  return [...new Set(ks)].sort((a, b) => a - b);
}

export function BuilderTab({ datasets, onStarted, onGoDatasets }: { datasets: DatasetSummary[]; onStarted: (runId: string) => void; onGoDatasets: () => void }) {
  const presets = useApi(fetchPresets);
  if (!datasets.length)
    return (
      <Panel>
        <EmptyState icon={Database} title="No benchmark dataset to run against" action={<Button size="sm" variant="primary" onClick={onGoDatasets}>Open datasets</Button>}>
          An experiment runs configurations over a registered, versioned dataset. Install the development benchmark or register your own first.
        </EmptyState>
      </Panel>
    );
  if (!presets.data)
    return <Panel>{presets.error ? <ErrorState title="Could not load arm templates">{presets.error}</ErrorState> : <LoadingState rows={6} label="Loading arm templates" />}</Panel>;
  return <Builder datasets={datasets} arms={presets.data.arms} suggested={presets.data.ablations} onStarted={onStarted} />;
}

function Builder({ datasets, arms, suggested, onStarted }: { datasets: DatasetSummary[]; arms: Arm[]; suggested: AblationSpec[]; onStarted: (runId: string) => void }) {
  const [datasetId, setDatasetId] = useState(datasets[0].id);
  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [chosen, setChosen] = useState<string[]>(DEFAULT_ARMS.filter((n) => arms.some((a) => a.name === n)));
  const [topK, setTopK] = useState<Record<string, number>>({});
  const [suggestedOn, setSuggestedOn] = useState<Set<string>>(new Set());
  const [custom, setCustom] = useState<AblationSpec[]>([]);
  const [draft, setDraft] = useState({ baseline: "", variant: "", factor: "" });
  const [ksText, setKsText] = useState("1, 3, 5, 10");
  const [maxCases, setMaxCases] = useState("");
  const [concurrency, setConcurrency] = useState(1);
  const [preview, setPreview] = useState<{ busy: boolean; data?: ResolveResponse; error?: string }>({ busy: false });
  const [start, setStart] = useState<{ busy: boolean; error?: string }>({ busy: false });

  const dataset = datasets.find((d) => d.id === datasetId) ?? datasets[0];
  const matrix = useMemo(
    () => chosen.map((n) => arms.find((a) => a.name === n)!).map((a) => ({ ...a, retrieval: { ...a.retrieval, top_k: topK[a.name] ?? a.retrieval.top_k } })),
    [arms, chosen, topK],
  );
  const key = (a: AblationSpec) => `${a.baseline}>${a.variant}`;
  const available = suggested.filter((a) => chosen.includes(a.baseline) && chosen.includes(a.variant));
  const ablations = [...available.filter((a) => suggestedOn.has(key(a))), ...custom.filter((a) => chosen.includes(a.baseline) && chosen.includes(a.variant))];
  const ks = parseKs(ksText);
  const max = maxCases.trim() ? Number(maxCases) : null;
  const maxValid = max === null || (Number.isInteger(max) && max >= 1);
  const valid = matrix.length > 0 && ks !== null && maxValid && name.trim().length > 0;
  const cases = Math.min(dataset.case_count, max ?? Infinity);

  const toggle = (n: string) => {
    setChosen((c) => (c.includes(n) ? c.filter((x) => x !== n) : [...c, n]));
    setPreview({ busy: false });
  };
  const body = () => ({
    dataset_id: datasetId,
    arms: matrix as ArmInput[],
    metrics: { ks: ks ?? [10] },
  });

  const resolve = async () => {
    setPreview({ busy: true });
    const { data, error, response } = await api.POST("/api/v1/arena/configurations/resolve", { body: body() });
    setPreview(data ? { busy: false, data } : { busy: false, error: errorMessage(error, response.status) });
  };

  const run = async () => {
    setStart({ busy: true });
    const created = await api.POST("/api/v1/experiments", {
      body: { ...body(), name: name.trim(), hypothesis: hypothesis.trim(), ablations, limits: { max_cases: max, concurrency } },
    });
    if (!created.data) return setStart({ busy: false, error: errorMessage(created.error, created.response.status) });
    const queued = await api.POST("/api/v1/experiments/{experiment_id}/runs", { params: { path: { experiment_id: created.data.id } } });
    if (!queued.data) return setStart({ busy: false, error: errorMessage(queued.error, queued.response.status) });
    setStart({ busy: false });
    onStarted(queued.data.id);
  };

  const denseWarning = preview.data && !preview.data.dense_index_ready && matrix.some(needsDense);

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[1fr_440px]">
      <div className="space-y-5">
        <Panel>
          <PanelHeader eyebrow="1 · Dataset" title="What every arm is evaluated on" />
          <div className="grid gap-4 p-4 sm:grid-cols-2">
            <FormField label="Benchmark dataset" htmlFor="ds">
              <select id="ds" value={datasetId} onChange={(e) => { setDatasetId(e.target.value); setPreview({ busy: false }); }} className="h-9 w-full rounded-md border border-line-strong bg-surface px-2 text-[13px]">
                {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} v{d.version} · {d.case_count} cases · {d.source}</option>)}
              </select>
            </FormField>
            <div className="self-end text-[11px] leading-relaxed text-fg-subtle">
              Pinned to corpus <span className="font-mono">{dataset.corpus_id.slice(0, 12)} v{dataset.corpus_version}</span>.
              {dataset.source === "development" && <span className="text-signal"> Development set: results validate the pipeline, they do not rank methods.</span>}
            </div>
          </div>
        </Panel>

        <Panel>
          <PanelHeader eyebrow="2 · Configurations" title={`${matrix.length} arm${matrix.length === 1 ? "" : "s"} selected (max 10)`} />
          <ul className="divide-y divide-line">
            {arms.map((a) => {
              const on = chosen.includes(a.name);
              const index = chosen.indexOf(a.name);
              return (
                <li key={a.name} className="flex flex-wrap items-center gap-3 px-4 py-2.5">
                  <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-2.5">
                    <input type="checkbox" checked={on} onChange={() => toggle(a.name)} disabled={!on && chosen.length >= 10} className="accent-[var(--color-signal)]" />
                    <span className="size-2 shrink-0 rounded-full" style={{ background: on ? armColor(index) : "var(--color-surface-3)" }} />
                    <span className="min-w-0">
                      <span className="block truncate text-[13px] text-fg">{a.label || a.name}</span>
                      <span className="block font-mono text-[10px] text-fg-subtle">
                        {a.name} · {a.pipeline} · {a.retrieval.mode === "adaptive" ? "router" : a.retrieval.strategy}
                        {a.retrieval.rerank.enabled && " + rerank"}
                        {a.generation && ` · ${a.generation.generator ?? "default generator"}`}
                      </span>
                    </span>
                  </label>
                  {on && (
                    <label className="flex items-center gap-1.5 font-mono text-[10px] text-fg-subtle">
                      top_k
                      <Input type="number" min={1} max={100} value={topK[a.name] ?? a.retrieval.top_k} aria-label={`top_k for ${a.name}`}
                        onChange={(e) => { setTopK((t) => ({ ...t, [a.name]: Math.max(1, Math.min(100, Number(e.target.value) || 1)) })); setPreview({ busy: false }); }}
                        className="h-7 w-16 px-2 text-xs" />
                    </label>
                  )}
                </li>
              );
            })}
          </ul>
          <p className="border-t border-line px-4 py-2.5 text-[11px] text-fg-subtle">
            Templates, not results. Other parameters (BM25, fusion weights, router, evidence, generation) can be set through <span className="font-mono">POST /api/v1/experiments</span>.
          </p>
        </Panel>

        <Panel>
          <PanelHeader eyebrow="3 · Ablations" title="Pairs to compare, and what each one varies" />
          <div className="space-y-3 p-4">
            {available.length ? (
              <ul className="space-y-1.5">
                {available.map((a) => (
                  <li key={key(a)}>
                    <label className="flex cursor-pointer items-center gap-2 text-[12px]">
                      <input type="checkbox" checked={suggestedOn.has(key(a))} onChange={() => setSuggestedOn((s) => { const n = new Set(s); if (n.has(key(a))) n.delete(key(a)); else n.add(key(a)); return n; })} className="accent-[var(--color-signal)]" />
                      <span className="font-mono">{a.baseline} → {a.variant}</span>
                      <Badge>{a.factor}</Badge>
                    </label>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[12px] text-fg-subtle">No suggested ablation fits the selected arms.</p>
            )}
            {custom.length > 0 && (
              <ul className="space-y-1.5">
                {custom.map((a, i) => (
                  <li key={`${key(a)}-${i}`} className="flex items-center gap-2 text-[12px]">
                    <span className="font-mono">{a.baseline} → {a.variant}</span>
                    <Badge tone="trace">{a.factor}</Badge>
                    {!(chosen.includes(a.baseline) && chosen.includes(a.variant)) && <span className="text-[11px] text-fg-subtle">(arm not selected; ignored)</span>}
                    <button type="button" aria-label={`Remove ablation ${a.baseline} to ${a.variant}`} onClick={() => setCustom((c) => c.filter((_, j) => j !== i))} className="text-fg-subtle hover:text-err"><X className="size-3.5" /></button>
                  </li>
                ))}
              </ul>
            )}
            <div className="flex flex-wrap items-end gap-2 border-t border-line pt-3">
              {(["baseline", "variant"] as const).map((side) => (
                <select key={side} aria-label={`Ablation ${side}`} value={draft[side]} onChange={(e) => setDraft((d) => ({ ...d, [side]: e.target.value }))} className="h-8 rounded-md border border-line-strong bg-surface px-2 text-xs">
                  <option value="">{side}…</option>
                  {chosen.map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              ))}
              <Input aria-label="Ablation factor" placeholder="factor, e.g. top_k" value={draft.factor} onChange={(e) => setDraft((d) => ({ ...d, factor: e.target.value }))} className="h-8 w-44 text-xs" />
              <Button size="sm" disabled={!draft.baseline || !draft.variant || draft.baseline === draft.variant || !draft.factor.trim()}
                onClick={() => { setCustom((c) => [...c, { ...draft, factor: draft.factor.trim(), note: "" }]); setDraft({ baseline: "", variant: "", factor: "" }); }}>
                <Plus className="size-3.5" /> Add ablation
              </Button>
            </div>
            <p className="text-[11px] leading-relaxed text-fg-subtle">
              The engine resolves both arms and records every setting that actually differs. An ablation that changes more than one factor is kept but flagged as confounded.
            </p>
          </div>
        </Panel>
      </div>

      <div className="space-y-5">
        <Panel>
          <PanelHeader eyebrow="4 · Experiment" title="Hypothesis, metrics and limits" />
          <div className="space-y-4 p-4">
            <FormField label="Name" htmlFor="exp-name" error={name.trim() ? null : undefined}>
              <Input id="exp-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. reranking on the dev set" />
            </FormField>
            <FormField label="Hypothesis" htmlFor="exp-hyp" hint="Stated before running; recorded with the experiment.">
              <Textarea id="exp-hyp" value={hypothesis} onChange={(e) => setHypothesis(e.target.value)} />
            </FormField>
            <FormField label="Cutoffs k" htmlFor="exp-ks" error={ks ? null : "Comma-separated whole numbers from 1 to 100"} hint="Recall, precision, hit rate and nDCG are reported at each k.">
              <Input id="exp-ks" value={ksText} onChange={(e) => { setKsText(e.target.value); setPreview({ busy: false }); }} aria-invalid={!ks} />
            </FormField>
            <div className="grid grid-cols-2 gap-3">
              <FormField label="Max cases" htmlFor="exp-max" error={maxValid ? null : "A whole number ≥ 1"} hint={`Blank = all ${dataset.case_count}`}>
                <Input id="exp-max" type="number" min={1} value={maxCases} onChange={(e) => setMaxCases(e.target.value)} aria-invalid={!maxValid} />
              </FormField>
              <FormField label="Concurrency" htmlFor="exp-conc" hint="Cases in parallel (1–4)">
                <Input id="exp-conc" type="number" min={1} max={4} value={concurrency} onChange={(e) => setConcurrency(Math.max(1, Math.min(4, Number(e.target.value) || 1)))} />
              </FormField>
            </div>
            <p className="font-mono text-[11px] text-fg-subtle">{matrix.length} arms × {cases} cases = {matrix.length * cases} evaluations · {ablations.length} ablation{ablations.length === 1 ? "" : "s"}</p>
            <div className="flex flex-wrap gap-2">
              <Button onClick={resolve} disabled={!matrix.length || !ks || preview.busy}><ScanEye className="size-3.5" /> {preview.busy ? "Resolving…" : "Preview snapshots"}</Button>
              <Button variant="primary" onClick={run} disabled={!valid || start.busy}><Play className="size-3.5" /> {start.busy ? "Starting…" : "Create & run"}</Button>
            </div>
            {start.error && <p role="alert" className="text-xs text-err">{start.error}</p>}
          </div>
        </Panel>

        <Panel>
          <PanelHeader eyebrow="Preview" title="Resolved configuration snapshots" />
          {preview.error ? (
            <ErrorState title="These arms cannot be resolved">{preview.error}</ErrorState>
          ) : !preview.data ? (
            <p className="p-4 text-[12px] leading-relaxed text-fg-subtle">
              Resolve the matrix to see each arm&apos;s snapshot hash, its models and what differs from the first arm. Nothing is created or run.
            </p>
          ) : (
            <div className="divide-y divide-line">
              {denseWarning && <p role="alert" className="px-4 py-2.5 text-[12px] text-signal">The corpus version has no dense index: dense, hybrid and adaptive arms will record every case as failed. Build it in the Retrieval Lab first.</p>}
              {preview.data.arms.map((r, i) => {
                const diff = preview.data!.diffs.find((d) => d.variant === r.snapshot.arm);
                return (
                  <div key={r.snapshot.arm} className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="size-2 rounded-full" style={{ background: armColor(i) }} />
                      <span className="font-mono text-[12px] text-fg">{r.snapshot.arm}</span>
                      <span className="ml-auto font-mono text-[10px] text-fg-subtle" title={r.config_hash}>{shortHash(r.config_hash, 12)}</span>
                    </div>
                    <div className="mt-1 font-mono text-[10px] text-fg-subtle">
                      {[r.snapshot.embedder?.model, r.snapshot.reranker?.model, r.snapshot.generator?.model, r.snapshot.router_policy].filter(Boolean).join(" · ") || "no learned components"}
                    </div>
                    {diff && <div className="mt-1 flex flex-wrap gap-1">{diff.factors.length ? diff.factors.map((f) => <Badge key={f} tone="trace">{f}</Badge>) : <Badge>identical settings</Badge>}</div>}
                  </div>
                );
              })}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
