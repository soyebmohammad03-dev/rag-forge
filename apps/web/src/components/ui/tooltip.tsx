import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

/** CSS-only tooltip: shows on hover and keyboard focus, no JS positioning needed yet. */
export function Tooltip({
  content,
  side = "top",
  children,
  className,
}: {
  content: ReactNode;
  side?: "top" | "right" | "bottom";
  children: ReactNode;
  className?: string;
}) {
  return (
    <span className={cn("group/tt relative inline-flex", className)}>
      {children}
      <span
        role="tooltip"
        className={cn(
          "pointer-events-none absolute z-50 w-max max-w-64 rounded-md border border-line-strong bg-surface-3 px-2 py-1 text-[11px] leading-snug text-fg-muted opacity-0 shadow-lg transition-opacity delay-150 duration-150",
          "group-hover/tt:opacity-100 group-focus-within/tt:opacity-100",
          side === "top" && "bottom-full left-1/2 mb-2 -translate-x-1/2",
          side === "bottom" && "top-full left-1/2 mt-2 -translate-x-1/2",
          side === "right" && "left-full top-1/2 ml-3 -translate-y-1/2",
        )}
      >
        {content}
      </span>
    </span>
  );
}
