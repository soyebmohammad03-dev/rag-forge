import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/cn";

export function Panel({ className, ...props }: ComponentProps<"section">) {
  return (
    <section
      className={cn(
        "relative rounded-[var(--radius-panel)] bg-surface/80 shadow-[var(--shadow-panel)] backdrop-blur-sm",
        className,
      )}
      {...props}
    />
  );
}

export function PanelHeader({
  title,
  eyebrow,
  actions,
  className,
}: {
  title: ReactNode;
  eyebrow?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "flex min-h-12 items-center justify-between gap-3 border-b border-line px-4 py-2.5",
        className,
      )}
    >
      <div className="min-w-0">
        {eyebrow && (
          <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">
            {eyebrow}
          </div>
        )}
        <h2 className="truncate text-[13px] font-medium text-fg">{title}</h2>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}
