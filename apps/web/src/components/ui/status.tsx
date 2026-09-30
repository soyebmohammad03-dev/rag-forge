import { cn } from "@/lib/cn";

export type Status = "ok" | "warn" | "err" | "idle" | "loading";

const color: Record<Status, string> = {
  ok: "bg-ok",
  warn: "bg-warn",
  err: "bg-err",
  idle: "bg-idle",
  loading: "bg-fg-subtle",
};

export function StatusDot({ status, className }: { status: Status; className?: string }) {
  return (
    <span className={cn("relative inline-flex size-2 shrink-0", className)} aria-hidden>
      {status === "ok" && (
        <span className="absolute inset-0 animate-ping rounded-full bg-ok opacity-40 [animation-duration:2.5s]" />
      )}
      <span
        className={cn(
          "relative size-2 rounded-full",
          color[status],
          status === "loading" && "animate-pulse-soft",
        )}
      />
    </span>
  );
}

export function StatusIndicator({
  status,
  label,
  className,
}: {
  status: Status;
  label: string;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-2 text-xs text-fg-muted", className)}>
      <StatusDot status={status} />
      <span>{label}</span>
    </span>
  );
}
