import type { ChunkingConfig, FileOutcome, IngestionStatus } from "@rag-forge/shared";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cn";

/** One colour per ingestion outcome, reused by uploads, version diffs and provenance. */
export const OUTCOME: Record<FileOutcome, { tone: "ok" | "trace" | "neutral" | "signal" | "err"; sign: string }> = {
  added: { tone: "ok", sign: "+" },
  modified: { tone: "signal", sign: "~" },
  removed: { tone: "err", sign: "−" },
  unchanged: { tone: "neutral", sign: "=" },
  duplicate: { tone: "trace", sign: "≡" },
  rejected: { tone: "err", sign: "×" },
};

export function OutcomeBadge({ outcome }: { outcome: FileOutcome }) {
  return (
    <Badge tone={OUTCOME[outcome].tone}>
      <span aria-hidden>{OUTCOME[outcome].sign}</span>
      {outcome}
    </Badge>
  );
}

export function IngestionStatusBadge({ status }: { status: IngestionStatus }) {
  const tone = status === "completed" ? "ok" : status === "failed" ? "err" : "neutral";
  return <Badge tone={tone}>{status.replace("_", " ")}</Badge>;
}

export function ChunkingSpec({ config, className }: { config: ChunkingConfig; className?: string }) {
  return (
    <span className={cn("whitespace-nowrap font-mono text-[11px] text-fg-muted", className)}>
      {config.strategy} · {config.chunk_size}
      <span className="text-fg-subtle"> chars</span> · {config.chunk_overlap}
      <span className="text-fg-subtle"> overlap</span>
    </span>
  );
}

export function Stat({ label, value, className }: { label: string; value: React.ReactNode; className?: string }) {
  return (
    <div className={cn("bg-surface px-4 py-3", className)}>
      <div className="num text-xl leading-tight text-fg">{value}</div>
      <div className="mt-0.5 font-mono text-[10px] uppercase tracking-wider text-fg-subtle">{label}</div>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 px-3 py-2 text-xs">
      <dt className="shrink-0 text-fg-subtle">{label}</dt>
      <dd className="min-w-0 truncate text-right text-fg-muted">{children}</dd>
    </div>
  );
}
