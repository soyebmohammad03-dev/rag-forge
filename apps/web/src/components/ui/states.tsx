import { AlertTriangle, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn(
        "animate-pulse-soft rounded bg-gradient-to-r from-surface-3 via-[#20252d] to-surface-3",
        className,
      )}
    />
  );
}

export function LoadingState({ rows = 3, label = "Loading" }: { rows?: number; label?: string }) {
  return (
    <div role="status" aria-label={label} className="space-y-2.5 p-4">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className={cn("h-4", i % 2 ? "w-2/3" : "w-full")} />
      ))}
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
  className,
}: {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center px-6 py-10 text-center", className)}>
      <div className="mb-3 grid size-10 place-items-center rounded-lg border border-dashed border-line-strong text-fg-subtle">
        <Icon className="size-4" />
      </div>
      <p className="text-[13px] font-medium text-fg">{title}</p>
      {children && <div className="mt-1 max-w-sm text-xs leading-relaxed text-fg-muted">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorState({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div role="alert" className="m-4 rounded-lg border border-err/25 bg-err/5 p-4">
      <div className="flex items-center gap-2 text-[13px] font-medium text-err">
        <AlertTriangle className="size-4" /> {title}
      </div>
      {children && <div className="mt-1.5 text-xs leading-relaxed text-fg-muted">{children}</div>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}
