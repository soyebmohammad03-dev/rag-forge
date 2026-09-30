import { Construction } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { PipelineSteps } from "@/components/ui/pipeline";
import { EmptyState } from "@/components/ui/states";
import { findArea } from "@/lib/areas";

const LIFECYCLE = [
  { id: "ingest", label: "Ingest" },
  { id: "configure", label: "Configure" },
  { id: "route", label: "Route" },
  { id: "retrieve", label: "Retrieve" },
  { id: "ground", label: "Ground" },
  { id: "evaluate", label: "Evaluate" },
  { id: "replay", label: "Replay" },
];

const STAGE: Record<string, string> = {
  corpus: "ingest",
  knowledge: "ingest",
  forge: "configure",
  router: "route",
  retrieval: "retrieve",
  evidence: "ground",
  experiments: "evaluate",
  arena: "evaluate",
  results: "evaluate",
  replay: "replay",
};

/** One honest page for every area that is designed but not built: purpose, contracts, no fake UI. */
export function PlannedArea({ slug }: { slug: string }) {
  const area = findArea(slug)!;
  const Icon = area.icon;
  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <header className="flex items-start gap-4">
        <div className="grid size-11 shrink-0 place-items-center rounded-lg border border-line-strong bg-surface-2 text-signal">
          <Icon className="size-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-medium tracking-tight">{area.label}</h1>
            <Badge>planned</Badge>
          </div>
          <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-fg-muted">{area.summary}</p>
        </div>
      </header>

      <Panel className="p-4">
        <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">Where this sits in the research loop</div>
        <PipelineSteps steps={LIFECYCLE} current={STAGE[slug]} />
      </Panel>

      <div className="grid gap-5 md:grid-cols-5">
        <Panel className="md:col-span-3">
          <PanelHeader eyebrow="Scope" title="What this area will do" />
          <ul className="space-y-2.5 p-4">
            {area.capabilities?.map((c) => (
              <li key={c} className="flex gap-2.5 text-[13px] leading-relaxed text-fg-muted">
                <span className="mt-2 size-1 shrink-0 rounded-full bg-signal" />
                {c}
              </li>
            ))}
          </ul>
        </Panel>
        <Panel className="md:col-span-2">
          <PanelHeader eyebrow="Foundations" title="Contracts it builds on" />
          <ul className="flex flex-wrap gap-1.5 p-4">
            {area.contracts?.map((c) => (
              <li key={c} className="rounded border border-line-strong bg-surface-2 px-2 py-1 font-mono text-[11px] text-trace">
                {c}
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <Panel>
        <EmptyState icon={Construction} title="Not built yet">
          This area has a defined contract but no implementation. It will show real data from recorded runs when its
          phase lands; it will not show invented results before then.
        </EmptyState>
      </Panel>
    </div>
  );
}
