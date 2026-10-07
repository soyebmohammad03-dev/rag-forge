"use client";

import type { ArenaOverview, Replay } from "@rag-forge/shared";
import { ShieldCheck } from "lucide-react";
import Link from "next/link";
import { OriginBadge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState } from "@/components/ui/states";
import { shortHash } from "@/lib/format";
import { fetchArenaOverview, fetchCorpora, fetchEnvironment, fetchHealth, fetchReplays, useApi } from "@/lib/use-api";
import { HealthPanel } from "@/features/system/health-panel";
import { CorpusOverview } from "./corpus-overview";
import { PlatformPipeline } from "./platform-pipeline";
import { RecentExperiments } from "./recent-experiments";

export function OverviewPage() {
  const health = useApi(fetchHealth);
  const corpora = useApi(fetchCorpora);
  const arena = useApi(fetchArenaOverview);
  const replays = useApi(fetchReplays);
  return (
    <div className="mx-auto max-w-[1440px] space-y-5">
      <header className="flex flex-col gap-4 pb-1 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Retrieval · Grounding · Reproducible experiments</p>
          <h1 className="mt-1.5 text-2xl font-medium tracking-tight">RAG FORGE</h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-fg-muted">
            A research platform for one question: can query characteristics choose a retrieval pipeline better than a fixed
            one? Every stage below is implemented, versioned and recorded, so a measured result can be traced to the exact
            configuration, models and data that produced it, and replayed.
          </p>
        </div>
        <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
          <OriginBadge origin="live" /> everything on this page is read from the API
        </span>
      </header>

      <PlatformPipeline health={health.data} corpora={corpora.data} arena={arena.data} replays={replays.data} />

      <div className="grid items-start gap-5 xl:grid-cols-3">
        <RecentExperiments className="xl:col-span-2" />
        <ReproducibilityPanel arena={arena.data} replays={replays.data} />
      </div>

      <div className="grid items-start gap-5 xl:grid-cols-3">
        <HealthPanel className="xl:col-span-2" />
        <CorpusOverview />
      </div>
    </div>
  );
}

function ReproducibilityPanel({ arena, replays }: { arena?: ArenaOverview; replays?: Replay[] }) {
  const env = useApi(fetchEnvironment);
  const latest = replays?.find((r) => r.status === "completed");
  const cases = latest?.cases.length ?? 0;
  const exact = latest?.outcomes.exact ?? 0;
  return (
    <Panel>
      <PanelHeader eyebrow="Reproducibility" title="What every result is pinned to" actions={<ShieldCheck className="size-4 text-ok" />} />
      <dl className="divide-y divide-line text-xs">
        <Row k="Git commit" v={env.data?.environment.git_commit ? shortHash(env.data.environment.git_commit, 12) : "—"} />
        <Row k="RAG FORGE" v={env.data ? `${env.data.environment.rag_forge_version} · Python ${env.data.environment.python_version}` : "—"} />
        <Row k="Metric registry" v={arena?.metric_registry ?? "—"} />
        <Row k="Statistics" v={arena ? `${arena.stats_method.name}@${arena.stats_method.version}` : "—"} />
      </dl>
      {latest ? (
        <Link href={`/replay?run=${latest.run_id}`} className="block border-t border-line px-4 py-3 transition-colors hover:bg-surface-2">
          <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-fg-subtle">Latest replay</div>
          <div className="mt-1 text-[13px] text-fg">
            <span className="num">{exact}</span> of <span className="num">{cases}</span> cases reproduced every stage hash
          </div>
          <div className="font-mono text-[10px] text-fg-subtle">{latest.id} · run {latest.run_id}</div>
        </Link>
      ) : (
        <EmptyState icon={ShieldCheck} title="No replays yet" action={<Link href="/replay" className="text-xs text-trace hover:underline">Open Replay</Link>}>
          Replaying a recorded run re-executes it and compares every stage&apos;s output hash with the recording.
        </EmptyState>
      )}
    </Panel>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3 px-4 py-2.5">
      <dt className="text-fg-subtle">{k}</dt>
      <dd className="truncate font-mono">{v}</dd>
    </div>
  );
}
