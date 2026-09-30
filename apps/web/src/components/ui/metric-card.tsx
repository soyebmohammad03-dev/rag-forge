import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import type { ContentOrigin } from "@rag-forge/shared";
import { cn } from "@/lib/cn";
import { OriginBadge } from "./badge";
import { Panel } from "./panel";
import { Tooltip } from "./tooltip";

export function Sparkline({
  values,
  className,
  color = "var(--color-trace)",
}: {
  values: number[];
  className?: string;
  color?: string;
}) {
  const w = 120;
  const h = 32;
  const min = Math.min(...values);
  const span = Math.max(...values) - min || 1;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * w, h - 2 - ((v - min) / span) * (h - 4)]);
  const line = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const gid = `spark-${color.replace(/\W/g, "")}`;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className={className} aria-hidden>
      <defs>
        <linearGradient id={gid} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.25" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${h} ${line} ${w},${h}`} fill={`url(#${gid})`} />
      <polyline points={line} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      <circle cx={pts.at(-1)![0]} cy={pts.at(-1)![1]} r="2.5" fill={color} />
    </svg>
  );
}

export function MetricCard({
  label,
  value,
  unit,
  delta,
  higherIsBetter = true,
  definition,
  series,
  origin,
  digits = 3,
}: {
  label: string;
  value: number;
  unit?: string;
  delta?: number;
  higherIsBetter?: boolean;
  definition?: string;
  series?: number[];
  origin: ContentOrigin | "live";
  digits?: number;
}) {
  const improved = delta !== undefined && (delta > 0) === higherIsBetter;
  const Arrow = delta !== undefined && delta > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <Panel className="group overflow-hidden p-4 transition-shadow hover:shadow-[0_0_0_1px_var(--color-line-strong),0_12px_32px_-16px_rgb(0_0_0/0.6)]">
      <div className="flex items-center justify-between">
        {definition ? (
          <Tooltip content={definition} side="bottom">
            <span
              tabIndex={0}
              className="cursor-help font-mono text-[11px] uppercase tracking-[0.12em] text-fg-muted underline decoration-line-strong decoration-dotted underline-offset-4"
            >
              {label}
            </span>
          </Tooltip>
        ) : (
          <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-fg-muted">{label}</span>
        )}
        <OriginBadge origin={origin} />
      </div>
      <div className="mt-3 flex items-end justify-between gap-3">
        <div>
          <div className="num text-[28px] leading-none tracking-tight text-fg">
            {value.toFixed(unit ? 2 : digits)}
            {unit && <span className="ml-0.5 text-base text-fg-muted">{unit}</span>}
          </div>
          {delta !== undefined && (
            <div className={cn("num mt-2 flex items-center gap-0.5 text-[11px]", improved ? "text-ok" : "text-err")}>
              <Arrow className="size-3" />
              {delta > 0 ? "+" : ""}
              {delta.toFixed(unit ? 2 : digits)}
              <span className="ml-1 whitespace-nowrap text-fg-subtle">vs baseline</span>
            </div>
          )}
        </div>
        {series && (
          <Sparkline
            values={series}
            color={improved ? "var(--color-trace)" : "var(--color-signal)"}
            className="h-9 w-24 shrink-0 opacity-80 transition-opacity group-hover:opacity-100"
          />
        )}
      </div>
    </Panel>
  );
}
