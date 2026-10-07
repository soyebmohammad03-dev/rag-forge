"use client";

import type { ArenaOverview, CorpusSummary, HealthStatus, Replay } from "@rag-forge/shared";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { compact } from "@/lib/format";

interface Stage {
  id: string;
  label: string;
  href: string;
  what: string;
  /** Live state from the API, when there is something to say. */
  live?: string;
}

interface Group {
  name: string;
  color: string;
  stages: Stage[];
}

const component = (h: HealthStatus | undefined, name: string) => h?.components.find((c) => c.name === name)?.detail.split("@")[0];

/** The implemented pipeline, stage by stage, each linked to the area where it can be used. */
export function PlatformPipeline({
  health,
  corpora,
  arena,
  replays,
}: {
  health?: HealthStatus;
  corpora?: CorpusSummary[];
  arena?: ArenaOverview;
  replays?: Replay[];
}) {
  const docs = corpora?.reduce((s, c) => s + c.stats.document_count, 0);
  const finished = arena?.runs.filter((r) => r.status === "completed" || r.status === "partial").length;
  const groups: Group[] = [
    {
      name: "Engine",
      color: "var(--color-s-sparse)",
      stages: [
        { id: "ingest", label: "Ingest", href: "/corpus", what: "TXT, Markdown, PDF into versioned, hashed chunks", live: corpora && `${corpora.length} corpora · ${compact.format(docs ?? 0)} docs` },
        { id: "retrieve", label: "Retrieve", href: "/retrieval", what: "BM25 and exact dense search, per corpus version", live: component(health, "embedder") },
        { id: "fuse", label: "Fuse", href: "/retrieval", what: "Reciprocal rank fusion or weighted min-max" },
        { id: "rerank", label: "Rerank", href: "/retrieval", what: "Cross-encoder over a candidate pool, rank movement kept", live: component(health, "reranker") },
      ],
    },
    {
      name: "Router",
      color: "var(--color-signal)",
      stages: [
        { id: "analyze", label: "Analyze", href: "/router", what: "Query features, signals and labels with contributions" },
        { id: "route", label: "Route", href: "/router", what: "Rule policy picks strategy and reranking, with a trace", live: health?.components.find((c) => c.name === "router")?.detail },
      ],
    },
    {
      name: "Grounded generation",
      color: "var(--color-trace)",
      stages: [
        { id: "evidence", label: "Evidence", href: "/evidence", what: "Budgeted, diverse, version-pinned passages" },
        { id: "generate", label: "Generate", href: "/evidence", what: "Local model or extractive baseline, cited answer", live: component(health, "llm_provider") },
        { id: "ground", label: "Ground", href: "/evidence", what: "Claims checked against the evidence they cite" },
      ],
    },
    {
      name: "Arena",
      color: "var(--color-s-rerank)",
      stages: [
        { id: "benchmark", label: "Benchmark", href: "/arena?tab=datasets", what: "Versioned datasets and a versioned metric registry", live: arena && `${arena.datasets.length} datasets · ${arena.metric_registry}` },
        { id: "experiment", label: "Experiment", href: "/experiments", what: "Hashed snapshots, runs, ablations, paired statistics", live: arena && `${finished} finished runs` },
        { id: "replay", label: "Replay", href: "/replay", what: "Stage-hash replay, manifest, report and exports", live: replays && `${replays.length} replays` },
      ],
    },
  ];
  const offsets = groups.map((_, i) => groups.slice(0, i).reduce((t, g) => t + g.stages.length, 0));
  return (
    <Panel>
      <PanelHeader eyebrow="Pipeline" title="What runs, in order, and where to use it" />
      <ol className="grid gap-px bg-line sm:grid-cols-2 xl:grid-cols-4" aria-label="Implemented pipeline">
        {groups.map((g, gi) => (
          <li key={g.name} className="bg-surface p-4">
            <div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">
              <span className="h-3 w-0.5 rounded-full" style={{ background: g.color }} />
              {g.name}
            </div>
            <ol className="space-y-1.5">
              {g.stages.map((s, si) => {
                const n = offsets[gi] + si + 1;
                return (
                  <li key={s.id}>
                    <Link href={s.href} className="group flex gap-3 rounded-md p-2 transition-colors hover:bg-surface-2">
                      <span className="num mt-0.5 w-5 shrink-0 font-mono text-[11px] text-fg-subtle">{String(n).padStart(2, "0")}</span>
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-1.5 text-[13px] font-medium text-fg">
                          {s.label}
                          <ArrowRight className="size-3 text-fg-subtle opacity-0 transition-opacity group-hover:opacity-100" />
                        </span>
                        <span className="block text-[11px] leading-snug text-fg-muted">{s.what}</span>
                        {s.live && <span className="mt-0.5 block truncate font-mono text-[10px] text-trace" title={s.live}>{s.live}</span>}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ol>
          </li>
        ))}
      </ol>
    </Panel>
  );
}
