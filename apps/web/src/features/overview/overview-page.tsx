import { OriginBadge } from "@/components/ui/badge";
import { MetricCard } from "@/components/ui/metric-card";
import { HealthPanel } from "@/features/system/health-panel";
import { SAMPLE_ORIGIN, sampleMetrics } from "@/sample/preview-data";
import { CorpusOverview } from "./corpus-overview";
import { RecentExperiments } from "./recent-experiments";
import { RetrievalActivity } from "./retrieval-activity";
import { RouterBlueprint } from "./router-blueprint";

export function OverviewPage() {
  return (
    <div className="mx-auto max-w-[1440px] space-y-5">
      <header className="flex flex-col gap-4 pb-1 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Engine · Router · Arena</p>
          <h1 className="mt-1.5 text-2xl font-medium tracking-tight">Overview</h1>
          <p className="mt-1 max-w-xl text-[13px] leading-relaxed text-fg-muted">
            Can adaptive retrieval policies choose and compose strategies better than fixed pipelines? Every number
            below states whether it was measured or is a preview.
          </p>
        </div>
        <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
          <span className="flex items-center gap-1.5"><OriginBadge origin="live" /> from API</span>
          <span className="flex items-center gap-1.5"><OriginBadge origin={SAMPLE_ORIGIN} /> preview only</span>
        </div>
      </header>

      <section aria-label="Metric summary" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {sampleMetrics.map((m) => (
          <MetricCard key={m.id} {...m} origin={SAMPLE_ORIGIN} />
        ))}
      </section>

      <RouterBlueprint />

      <div className="grid gap-5 xl:grid-cols-3">
        <RetrievalActivity className="xl:col-span-2" />
        <HealthPanel />
      </div>

      <div className="grid items-start gap-5 xl:grid-cols-3">
        <RecentExperiments className="xl:col-span-2" />
        <CorpusOverview />
      </div>
    </div>
  );
}
