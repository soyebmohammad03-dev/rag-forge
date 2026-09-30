import type { TooltipContentProps } from "recharts";

/** Shared Recharts styling so every chart speaks the same visual language. */
export const chartAxis = {
  stroke: "var(--color-line-strong)",
  tick: { fill: "var(--color-fg-subtle)", fontSize: 10, fontFamily: "var(--font-mono)" },
  tickLine: false,
  axisLine: false,
} as const;

export const STRATEGY_COLOR: Record<string, string> = {
  sparse: "var(--color-s-sparse)",
  dense: "var(--color-s-dense)",
  hybrid: "var(--color-s-hybrid)",
  metadata: "var(--color-s-metadata)",
  graph: "var(--color-s-graph)",
  rerank: "var(--color-s-rerank)",
};

export function ChartTooltip({ active, payload, label }: Partial<TooltipContentProps<number, string>>) {
  if (!active || !payload?.length) return null;
  const total = payload.reduce((s, p) => s + Number(p.value ?? 0), 0);
  return (
    <div className="min-w-36 rounded-md border border-line-strong bg-surface-3/95 px-3 py-2 text-[11px] shadow-xl backdrop-blur">
      <div className="mb-1.5 font-mono text-fg-subtle">{label}</div>
      {[...payload].reverse().map((p) => (
        <div key={String(p.dataKey)} className="flex items-center justify-between gap-4 py-0.5">
          <span className="flex items-center gap-1.5 text-fg-muted">
            <span className="size-2 rounded-sm" style={{ background: p.color }} />
            {p.name}
          </span>
          <span className="num text-fg">{p.value}</span>
        </div>
      ))}
      <div className="mt-1.5 flex justify-between border-t border-line pt-1.5 text-fg-muted">
        <span>total</span>
        <span className="num text-fg">{total}</span>
      </div>
    </div>
  );
}
