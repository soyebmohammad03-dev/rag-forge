"use client";

import type { ExperimentRun, Replay, ReplayCase, ReplayOutcome } from "@rag-forge/shared";
import { History, Play } from "lucide-react";
import { useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { StatusIndicator } from "@/components/ui/status";
import { api } from "@/lib/api";
import { shortHash } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { errorMessage } from "@/features/retrieval/errors";

export const OUTCOME: Record<ReplayOutcome, { tone: "ok" | "trace" | "err" | "neutral"; label: string; help: string }> = {
  exact: { tone: "ok", label: "exact", help: "Every recorded stage hash matched, generation included." },
  equivalent: { tone: "trace", label: "equivalent", help: "Same configuration; every deterministic stage matched; a non-deterministic stage (generation) differed." },
  diverged: { tone: "err", label: "diverged", help: "Same configuration, but a deterministic stage produced a different output." },
  not_replayable: { tone: "neutral", label: "not replayable", help: "A component is unavailable here, or the arm resolves to a different configuration." },
};
const ORDER: ReplayOutcome[] = ["exact", "equivalent", "diverged", "not_replayable"];

const pollReplay = (r: Replay | undefined) => (r && (r.status === "queued" || r.status === "running") ? 1000 : undefined);

export function ReplayView({ run }: { run: ExperimentRun }) {
  const fetchList = useCallback(() => api.GET("/api/v1/runs/{run_id}/replays", { params: { path: { run_id: run.id } } }), [run.id]);
  const list = useApi(fetchList);
  const [selected, setSelected] = useState<string | null>(null);
  const [start, setStart] = useState<{ busy: boolean; error?: string }>({ busy: false });
  const current = selected ?? list.data?.[0]?.id ?? null;

  const replay = async (caseIds?: string[]) => {
    setStart({ busy: true });
    const { data, error, response } = await api.POST("/api/v1/runs/{run_id}/replays", {
      params: { path: { run_id: run.id } },
      body: caseIds ? { case_ids: caseIds } : {},
    });
    if (!data) return setStart({ busy: false, error: errorMessage(error, response.status) });
    setStart({ busy: false });
    setSelected(data.id);
    list.reload();
  };

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[340px_1fr]">
      <div className="space-y-5">
        <Panel>
          <PanelHeader eyebrow="Replay" title="Re-execute and compare every stage" />
          <div className="space-y-3 p-4 text-[12px] leading-relaxed text-fg-muted">
            <p>
              Each case is run again with the arm&apos;s recorded configuration on the pinned corpus version, using the models
              loaded here. Every stage&apos;s output hash and every quality metric is compared with the recording. Timings are
              not compared.
            </p>
            <div className="flex flex-wrap gap-2">
              <Button variant="primary" onClick={() => replay()} disabled={start.busy}><Play className="size-3.5" /> {start.busy ? "Starting…" : `Replay all ${run.total} evaluations`}</Button>
              {run.case_ids.length > 3 && <Button onClick={() => replay(run.case_ids.slice(0, 3))} disabled={start.busy}>First 3 cases</Button>}
            </div>
            {start.error && <p role="alert" className="text-err">{start.error}</p>}
            <ul className="space-y-1 border-t border-line pt-3 text-[11px]">
              {ORDER.map((o) => <li key={o} className="flex gap-2"><Badge tone={OUTCOME[o].tone}>{OUTCOME[o].label}</Badge><span className="text-fg-subtle">{OUTCOME[o].help}</span></li>)}
            </ul>
          </div>
        </Panel>
        <Panel>
          <PanelHeader eyebrow="History" title="Replays of this run" />
          {!list.data ? (
            list.error ? <ErrorState title="Could not load replays">{list.error}</ErrorState> : <LoadingState rows={2} label="Loading replays" />
          ) : list.data.length ? (
            <ul className="divide-y divide-line">
              {list.data.map((r) => (
                <li key={r.id}>
                  <button type="button" onClick={() => setSelected(r.id)} aria-pressed={current === r.id} className={`w-full px-4 py-2.5 text-left transition-colors hover:bg-surface-2 ${current === r.id ? "bg-surface-2" : ""}`}>
                    <div className="flex items-center justify-between font-mono text-[11px]"><span>{r.id}</span><span className="text-fg-subtle">{new Date(r.created_at).toLocaleString()}</span></div>
                    <OutcomeBar replay={r} />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="p-4 text-[12px] text-fg-subtle">This run has not been replayed yet.</p>
          )}
        </Panel>
      </div>
      {current ? <ReplayDetail key={current} replayId={current} onDone={list.reload} /> : (
        <Panel><EmptyState icon={History} title="No replay selected">Start a replay to see, for every case, which stages reproduced.</EmptyState></Panel>
      )}
    </div>
  );
}

function OutcomeBar({ replay }: { replay: Replay }) {
  const total = Math.max(1, replay.total);
  return (
    <div className="mt-1.5">
      <div className="flex h-1.5 overflow-hidden rounded-full bg-surface-3">
        {ORDER.map((o) => {
          const n = replay.outcomes[o] ?? 0;
          return n ? <span key={o} title={`${OUTCOME[o].label}: ${n}`} style={{ width: `${(n / total) * 100}%`, background: `var(--color-${o === "exact" ? "ok" : o === "equivalent" ? "trace" : o === "diverged" ? "err" : "fg-subtle"})` }} /> : null;
        })}
      </div>
      <div className="mt-1 font-mono text-[10px] text-fg-subtle">
        {replay.status === "completed" ? ORDER.filter((o) => replay.outcomes[o]).map((o) => `${replay.outcomes[o]} ${OUTCOME[o].label}`).join(" · ") : `${replay.status} · ${replay.completed}/${replay.total}`}
      </div>
    </div>
  );
}

function ReplayDetail({ replayId, onDone }: { replayId: string; onDone: () => void }) {
  const fetcher = useCallback(() => api.GET("/api/v1/replays/{replay_id}", { params: { path: { replay_id: replayId } } }), [replayId]);
  const poll = useCallback((r: Replay | undefined) => {
    const next = pollReplay(r);
    if (r && next === undefined) onDone();
    return next;
  }, [onDone]);
  const replay = useApi(fetcher, poll);
  const [open, setOpen] = useState<string | null>(null);
  if (!replay.data) return <Panel>{replay.error ? <ErrorState title="Could not load the replay">{replay.error}</ErrorState> : <LoadingState rows={6} label="Loading the replay" />}</Panel>;
  const r = replay.data;
  const opened = r.cases.find((c) => `${c.arm}|${c.case_id}` === open);
  return (
    <div className="space-y-5">
      <Panel>
        <PanelHeader
          eyebrow={`Replay ${r.id}`}
          title={r.status === "completed" ? `${r.total} evaluations replayed` : `Replaying… ${r.completed}/${r.total}`}
          actions={<StatusIndicator status={r.status === "completed" ? "ok" : r.status === "failed" ? "err" : "loading"} label={r.status} />}
        />
        <div className="grid gap-px bg-line sm:grid-cols-4">
          {ORDER.map((o) => (
            <div key={o} className="bg-surface px-4 py-3" title={OUTCOME[o].help}>
              <div className="num text-xl text-fg">{r.outcomes[o] ?? (r.status === "completed" ? 0 : "…")}</div>
              <div className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">{OUTCOME[o].label}</div>
            </div>
          ))}
        </div>
        {r.error && <p role="alert" className="border-t border-line px-4 py-2 text-[12px] text-err">{r.error}</p>}
        <p className="border-t border-line px-4 py-2.5 font-mono text-[10px] text-fg-subtle">
          replayed at commit {r.runtime.git_commit ? shortHash(r.runtime.git_commit, 10) : "unknown"}{r.runtime.git_dirty ? " (uncommitted changes)" : ""} · Python {r.environment.python_version} · {r.environment.platform}
        </p>
      </Panel>
      <Panel>
        <PanelHeader eyebrow="Per case" title="Outcome, first differing stage and metric differences (click a row)" />
        <div className="max-h-[520px] overflow-auto">
          <table className="w-full text-left text-[12px]">
            <thead className="sticky top-0 bg-surface font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
              <tr className="border-b border-line"><th className="px-4 py-2 font-normal">arm</th><th className="font-normal">case</th><th className="font-normal">outcome</th><th className="font-normal">first difference</th><th className="text-right font-normal">metrics differing</th><th className="px-4 font-normal">reason</th></tr>
            </thead>
            <tbody>
              {r.cases.map((c) => {
                const key = `${c.arm}|${c.case_id}`;
                return (
                  <tr key={key} tabIndex={0} onClick={() => setOpen(key)} onKeyDown={(e) => e.key === "Enter" && setOpen(key)} className={`cursor-pointer border-b border-line/60 align-top last:border-0 hover:bg-surface-2 ${open === key ? "bg-surface-2" : ""}`}>
                    <td className="px-4 py-1.5 font-mono text-[11px]">{c.arm}</td>
                    <td className="py-1.5 font-mono text-[11px] text-fg-muted">{c.case_id}</td>
                    <td className="py-1.5"><Badge tone={OUTCOME[c.outcome].tone}>{OUTCOME[c.outcome].label}</Badge></td>
                    <td className="py-1.5 font-mono text-[11px] text-fg-muted">{c.first_divergence ?? "—"}</td>
                    <td className="num py-1.5 text-right">{c.metric_differences.length}/{c.metrics_compared}</td>
                    <td className="px-4 py-1.5 text-[11px] text-fg-subtle">{c.reason}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
      {opened && <StageTable c={opened} />}
    </div>
  );
}

function StageTable({ c }: { c: ReplayCase }) {
  return (
    <Panel>
      <PanelHeader eyebrow={`${c.arm} · ${c.case_id}`} title="Recorded vs replayed output hash, per stage" />
      {c.stages.length ? (
        <table className="w-full text-left font-mono text-[11px]">
          <thead className="text-[10px] uppercase tracking-wider text-fg-subtle"><tr className="border-b border-line"><th className="px-4 py-2 font-normal">stage</th><th className="font-normal">deterministic</th><th className="font-normal">recorded</th><th className="font-normal">replayed</th><th className="px-4 font-normal">match</th></tr></thead>
          <tbody>
            {c.stages.map((s) => (
              <tr key={s.stage} className="border-b border-line/60 last:border-0">
                <td className="px-4 py-1.5 text-fg">{s.stage}</td>
                <td className={s.deterministic ? "text-fg-muted" : "text-signal"}>{s.deterministic ? "yes" : "no"}</td>
                <td className="text-fg-muted" title={s.recorded ?? ""}>{s.recorded ? shortHash(s.recorded, 14) : "—"}</td>
                <td className="text-fg-muted" title={s.replayed ?? ""}>{s.replayed ? shortHash(s.replayed, 14) : "—"}</td>
                <td className={`px-4 ${s.match ? "text-ok" : s.match === false ? "text-err" : "text-fg-subtle"}`}>{s.match ? "match" : s.match === false ? "differs" : "absent"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="p-4 text-[12px] text-fg-subtle">{c.reason}</p>
      )}
      {c.metric_differences.length > 0 && (
        <ul className="border-t border-line px-4 py-2.5 font-mono text-[11px] text-fg-muted">
          {c.metric_differences.map((d) => <li key={d.metric}>{d.metric}: {d.recorded ?? "—"} → {d.replayed ?? "—"}</li>)}
        </ul>
      )}
      <p className="border-t border-line px-4 py-2 font-mono text-[10px] text-fg-subtle">recorded artifact {c.recorded_artifact_id ?? "—"} · replayed artifact {c.replayed_artifact_id ?? "—"}</p>
    </Panel>
  );
}
