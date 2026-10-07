import { Check } from "lucide-react";
import { cn } from "@/lib/cn";

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
