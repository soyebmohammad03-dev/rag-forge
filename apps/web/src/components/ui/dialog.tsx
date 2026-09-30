"use client";

import { X } from "lucide-react";
import { type ReactNode, useEffect, useRef } from "react";
import { cn } from "@/lib/cn";
import { Button } from "./button";

/**
 * Native <dialog>: focus trapping, Escape and the top layer come from the platform.
 * `variant="drawer"` docks it to the right edge.
 */
export function Dialog({
  open,
  onClose,
  title,
  description,
  variant = "dialog",
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  variant?: "dialog" | "drawer";
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className={cn(
        "m-0 border border-line-strong bg-surface-2 p-0 text-fg shadow-2xl backdrop:transition-opacity",
        variant === "dialog" &&
          "left-1/2 top-[12vh] w-[min(560px,calc(100vw-32px))] -translate-x-1/2 rounded-xl open:animate-[dialog-in_180ms_ease-out]",
        variant === "drawer" &&
          "left-auto right-0 top-0 h-dvh max-h-dvh w-[min(480px,100vw)] max-w-none rounded-l-xl open:animate-[drawer-in_220ms_cubic-bezier(.2,.8,.2,1)]",
      )}
    >
      <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
        <div>
          <h2 className="text-sm font-medium">{title}</h2>
          {description && <p className="mt-0.5 text-xs text-fg-muted">{description}</p>}
        </div>
        <Button variant="ghost" size="icon" aria-label="Close" onClick={onClose}>
          <X className="size-4" />
        </Button>
      </div>
      <div className={cn("p-5", variant === "drawer" && "h-[calc(100%-65px)] overflow-y-auto")}>
        {children}
      </div>
    </dialog>
  );
}
