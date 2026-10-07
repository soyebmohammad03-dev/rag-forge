"use client";

import type { CaseDifference } from "@rag-forge/shared";

/** Means with confidence intervals on one shared axis. Only measured values are drawn. */
export function IntervalPlot({
  rows,
  domain: [lo, hi],
  zero,
  label,
  format,
}: {
  rows: { key: string; label: string; color: string; mean: number | null; low: number | null; high: number | null }[];
  domain: [number, number];
  zero?: boolean;
  label: string;
  format: (v: number) => string;
}) {
  const W = 360;
  const row = 22;
  const H = rows.length * row + 18;
  const x = (v: number) => 8 + ((v - lo) / (hi - lo || 1)) * (W - 16);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={label}>
      {zero && lo < 0 && hi > 0 && <line x1={x(0)} x2={x(0)} y1={0} y2={H - 14} stroke="var(--color-line-strong)" strokeDasharray="3 3" />}
      {rows.map((r, i) => {
        const y = i * row + row / 2;
        return (
          <g key={r.key}>
            <line x1={8} x2={W - 8} y1={y} y2={y} stroke="var(--color-line)" />
            {r.low !== null && r.high !== null && (
              <line x1={x(r.low)} x2={x(r.high)} y1={y} y2={y} stroke={r.color} strokeWidth={3} strokeLinecap="round" opacity={0.45} />
            )}
            {r.mean !== null && <circle cx={x(r.mean)} cy={y} r={4} fill={r.color}><title>{`${r.label}: ${format(r.mean)}${r.low !== null && r.high !== null ? ` [${format(r.low)}, ${format(r.high)}]` : ""}`}</title></circle>}
          </g>
        );
      })}
      <text x={8} y={H - 3} fontSize={9} fill="var(--color-fg-subtle)" fontFamily="var(--font-mono)">{format(lo)}</text>
      <text x={W - 8} y={H - 3} fontSize={9} fill="var(--color-fg-subtle)" fontFamily="var(--font-mono)" textAnchor="end">{format(hi)}</text>
    </svg>
  );
}

/** Per-case paired differences (variant − baseline), sorted, around a zero line. */
export function DiffPlot({ differences, higherIsBetter, format }: { differences: CaseDifference[]; higherIsBetter: boolean | null; format: (v: number) => string }) {
  const W = 420;
  const H = 120;
  const sorted = [...differences].sort((a, b) => a.difference - b.difference);
  const ext = Math.max(1e-9, ...sorted.map((d) => Math.abs(d.difference)));
  const y = (v: number) => H / 2 - (v / ext) * (H / 2 - 10);
  const step = (W - 20) / Math.max(1, sorted.length);
  const good = (v: number) => (higherIsBetter === null || v === 0 ? "var(--color-fg-subtle)" : (v > 0) === higherIsBetter ? "var(--color-ok)" : "var(--color-err)");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Paired per-case differences">
      <line x1={10} x2={W - 10} y1={H / 2} y2={H / 2} stroke="var(--color-line-strong)" />
      {sorted.map((d, i) => {
        const cx = 10 + step * (i + 0.5);
        return (
          <g key={d.case_id}>
            <line x1={cx} x2={cx} y1={H / 2} y2={y(d.difference)} stroke={good(d.difference)} strokeWidth={Math.max(1, Math.min(6, step * 0.5))} opacity={0.8} />
            <title>{`${d.case_id}: ${format(d.baseline)} → ${format(d.variant)} (${d.difference >= 0 ? "+" : ""}${format(d.difference)})`}</title>
          </g>
        );
      })}
      <text x={10} y={10} fontSize={9} fill="var(--color-fg-subtle)" fontFamily="var(--font-mono)">+{format(ext)}</text>
      <text x={10} y={H - 3} fontSize={9} fill="var(--color-fg-subtle)" fontFamily="var(--font-mono)">−{format(ext)}</text>
    </svg>
  );
}

/** One dot per case per arm (e.g. latency), with the median marked. */
export function StripPlot({
  groups,
  format,
  label,
}: {
  groups: { key: string; label: string; color: string; values: number[] }[];
  format: (v: number) => string;
  label: string;
}) {
  const W = 420;
  const row = 26;
  const H = groups.length * row + 16;
  const all = groups.flatMap((g) => g.values);
  const lo = Math.min(0, ...all);
  const hi = Math.max(1e-9, ...all);
  const x = (v: number) => 110 + ((v - lo) / (hi - lo)) * (W - 120);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={label}>
      {groups.map((g, i) => {
        const y = i * row + row / 2;
        const s = [...g.values].sort((a, b) => a - b);
        const median = s.length ? s[Math.floor((s.length - 1) / 2)] : null;
        return (
          <g key={g.key}>
            <text x={4} y={y + 3} fontSize={10} fill="var(--color-fg-muted)" fontFamily="var(--font-mono)">{g.label.slice(0, 16)}</text>
            <line x1={110} x2={W - 10} y1={y} y2={y} stroke="var(--color-line)" />
            {g.values.map((v, j) => (
              <circle key={j} cx={x(v)} cy={y + ((j % 5) - 2) * 1.6} r={2.4} fill={g.color} opacity={0.7} />
            ))}
            {median !== null && <line x1={x(median)} x2={x(median)} y1={y - 8} y2={y + 8} stroke="var(--color-fg)" strokeWidth={1.5}><title>{`median ${format(median)}`}</title></line>}
          </g>
        );
      })}
      <text x={110} y={H - 2} fontSize={9} fill="var(--color-fg-subtle)" fontFamily="var(--font-mono)">{format(lo)}</text>
      <text x={W - 10} y={H - 2} fontSize={9} fill="var(--color-fg-subtle)" fontFamily="var(--font-mono)" textAnchor="end">{format(hi)}</text>
    </svg>
  );
}
