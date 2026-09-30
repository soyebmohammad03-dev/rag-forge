import { OriginBadge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { cn } from "@/lib/cn";
import { SAMPLE_ORIGIN, sampleCorpora } from "@/sample/preview-data";

const fmt = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

export function CorpusOverview({ className }: { className?: string }) {
  const totalChunks = sampleCorpora.reduce((s, c) => s + c.chunks, 0);
  return (
    <Panel className={className}>
      <PanelHeader eyebrow="Corpus" title="Indexed collections" actions={<OriginBadge origin={SAMPLE_ORIGIN} />} />
      <div className="grid grid-cols-2 gap-px border-b border-line bg-line">
        <Stat label="collections" value={String(sampleCorpora.length)} />
        <Stat label="chunks" value={fmt.format(totalChunks)} />
      </div>
      <ul className="divide-y divide-line/60">
        {sampleCorpora.map((c) => (
          <li key={c.name} className="px-4 py-2.5">
            <div className="flex items-baseline justify-between gap-2">
              <span className="truncate font-mono text-xs text-fg">
                {c.name}
                <span className="ml-1.5 text-fg-subtle">v{c.version}</span>
              </span>
              <span className="num shrink-0 text-[11px] text-fg-muted">{fmt.format(c.chunks)} chunks</span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <div
                className="h-1 flex-1 overflow-hidden rounded-full bg-surface-3"
                role="progressbar"
                aria-label={`${c.name} indexed`}
                aria-valuenow={Math.round(c.indexed * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className={cn("h-full rounded-full", c.indexed === 1 ? "bg-trace" : "bg-signal")}
                  style={{ width: `${c.indexed * 100}%` }}
                />
              </div>
              <span className="w-24 truncate text-right font-mono text-[10px] text-fg-subtle">
                {c.indexed === 0 ? "not indexed" : c.indexed < 1 ? `indexing ${Math.round(c.indexed * 100)}%` : c.embedding}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface px-4 py-3">
      <div className="num text-xl text-fg">{value}</div>
      <div className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">{label}</div>
    </div>
  );
}
