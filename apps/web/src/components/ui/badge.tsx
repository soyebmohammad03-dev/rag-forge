import type { ComponentProps } from "react";
import type { ContentOrigin } from "@rag-forge/shared";
import { cn } from "@/lib/cn";

const tones = {
  neutral: "text-fg-muted bg-surface-3 ring-line-strong",
  signal: "text-signal bg-signal-dim/60 ring-signal/25",
  trace: "text-trace bg-trace-dim/60 ring-trace/25",
  ok: "text-ok bg-ok/10 ring-ok/25",
  err: "text-err bg-err/10 ring-err/25",
};

export function Badge({
  tone = "neutral",
  className,
  ...props
}: ComponentProps<"span"> & { tone?: keyof typeof tones }) {
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center gap-1 whitespace-nowrap rounded px-1.5 font-mono text-[10px] uppercase tracking-wider ring-1 ring-inset",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}

/**
 * Every data-bearing surface declares where its numbers came from.
 * "simulated" (sample/preview data) is hatched so it can never pass for a result.
 */
export function OriginBadge({ origin }: { origin: ContentOrigin | "live" }) {
  if (origin === "simulated")
    return (
      <Badge className="hatch" title="Sample data for interface preview. Not a measured result.">
        sample
      </Badge>
    );
  if (origin === "live" || origin === "measured")
    return (
      <Badge tone="ok" title="Read from the running RAG FORGE API">
        <span className="size-1.5 animate-pulse-soft rounded-full bg-ok" />
        {origin}
      </Badge>
    );
  return <Badge tone="trace">{origin}</Badge>;
}
