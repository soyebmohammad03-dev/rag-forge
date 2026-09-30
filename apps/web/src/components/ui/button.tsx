import type { ComponentProps } from "react";
import { cn } from "@/lib/cn";

const variants = {
  primary: "bg-signal text-black hover:bg-[#f5b457] font-medium",
  secondary: "bg-surface-3 text-fg shadow-[0_0_0_1px_var(--color-line-strong)] hover:bg-[#222731]",
  ghost: "text-fg-muted hover:bg-surface-3 hover:text-fg",
};

const sizes = {
  sm: "h-7 px-2.5 text-xs gap-1.5",
  md: "h-8 px-3 text-[13px] gap-2",
  icon: "size-8 justify-center",
};

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ComponentProps<"button"> & { variant?: keyof typeof variants; size?: keyof typeof sizes }) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex select-none items-center rounded-md transition-colors disabled:pointer-events-none disabled:opacity-40",
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  );
}

export function Kbd({ className, ...props }: ComponentProps<"kbd">) {
  return (
    <kbd
      className={cn(
        "inline-flex h-5 min-w-5 items-center justify-center rounded border border-line-strong bg-surface-2 px-1 font-mono text-[10px] text-fg-muted",
        className,
      )}
      {...props}
    />
  );
}
