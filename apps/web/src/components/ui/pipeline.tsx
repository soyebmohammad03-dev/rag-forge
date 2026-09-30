import { Check } from "lucide-react";
import type { KeyboardEvent } from "react";
import { cn } from "@/lib/cn";

export type BuildState = "implemented" | "contract" | "planned";

export const BUILD_STATE_LABEL: Record<BuildState, string> = {
  implemented: "Implemented",
  contract: "API contract defined",
  planned: "Planned",
};

/** An SVG graph node. Coordinates are the node's top-left corner in the parent viewBox. */
export function GraphNode({
  x,
  y,
  w = 104,
  h = 40,
  label,
  sublabel,
  color = "var(--color-fg-muted)",
  state,
  active,
  dimmed,
  onSelect,
}: {
  x: number;
  y: number;
  w?: number;
  h?: number;
  label: string;
  sublabel?: string;
  color?: string;
  state: BuildState;
  active?: boolean;
  dimmed?: boolean;
  onSelect?: () => void;
}) {
  const onKey = (e: KeyboardEvent) => (e.key === "Enter" || e.key === " ") && onSelect?.();
  return (
    <g
      transform={`translate(${x} ${y})`}
      role="button"
      tabIndex={0}
      aria-label={`${label}: ${BUILD_STATE_LABEL[state]}`}
      aria-pressed={active}
      onClick={onSelect}
      onFocus={onSelect}
      onMouseEnter={onSelect}
      onKeyDown={onKey}
      className="cursor-pointer outline-none transition-opacity duration-300 [&:focus-visible>rect:first-child]:stroke-trace"
      style={{ opacity: dimmed ? 0.5 : 1 }}
    >
      <rect
        width={w}
        height={h}
        rx={7}
        fill="var(--color-surface-2)"
        stroke={active ? color : "var(--color-line-strong)"}
        strokeWidth={active ? 1.5 : 1}
        strokeDasharray={state === "planned" ? "3 3" : undefined}
        className="transition-[stroke] duration-200"
      />
      <rect x={0} y={8} width={2.5} height={h - 16} rx={1} fill={color} />
      <text x={12} y={sublabel ? 17 : h / 2 + 4} fill="var(--color-fg)" fontSize={13} fontWeight={500}>
        {label}
      </text>
      {sublabel && (
        <text x={12} y={31} fill="var(--color-fg-subtle)" fontSize={10} fontFamily="var(--font-mono)">
          {sublabel}
        </text>
      )}
      <circle
        cx={w - 10}
        cy={10}
        r={2.5}
        fill={state === "implemented" ? "var(--color-ok)" : state === "contract" ? "var(--color-trace)" : "var(--color-idle)"}
      />
    </g>
  );
}

export interface Step {
  id: string;
  label: string;
}

/** Horizontal lifecycle stepper; `current` is highlighted, earlier steps are shown as passed. */
export function PipelineSteps({ steps, current }: { steps: Step[]; current?: string }) {
  const idx = steps.findIndex((s) => s.id === current);
  return (
    <ol className="flex flex-wrap items-center gap-y-2">
      {steps.map((s, i) => {
        const state = i < idx ? "done" : i === idx ? "current" : "todo";
        return (
          <li key={s.id} className="flex items-center" aria-current={state === "current" ? "step" : undefined}>
            <span
              className={cn(
                "flex h-7 items-center gap-1.5 rounded-full px-3 font-mono text-[11px] ring-1 ring-inset transition-colors",
                state === "current" && "bg-signal-dim text-signal ring-signal/40",
                state === "done" && "text-fg-muted ring-line-strong",
                state === "todo" && "text-fg-subtle ring-line",
              )}
            >
              {state === "done" && <Check className="size-3" />}
              {s.label}
            </span>
            {i < steps.length - 1 && (
              <span className={cn("mx-1 h-px w-5", i < idx ? "bg-line-strong" : "bg-line")} />
            )}
          </li>
        );
      })}
    </ol>
  );
}
