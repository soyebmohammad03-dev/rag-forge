"use client";

import { Keyboard, Menu, Search } from "lucide-react";
import { usePathname } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";
import { Button, Kbd } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { AREAS } from "@/lib/areas";
import { CommandPalette } from "./command-palette";
import { LogoMark } from "./logo";
import { Sidebar, SidebarNav, isActive } from "./sidebar";

const SHORTCUTS: [string[], string][] = [
  [["⌘", "K"], "Open command palette"],
  [["?"], "Show keyboard shortcuts"],
  [["←", "→"], "Move between tabs"],
  [["↵"], "Open focused row"],
  [["Esc"], "Close dialog or drawer"],
];

const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [palette, setPalette] = useState(false);
  const [shortcuts, setShortcuts] = useState(false);
  const [nav, setNav] = useState(false);
  const area = AREAS.find((a) => isActive(pathname, a.slug));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setPalette((o) => !o);
      } else if (e.key === "?" && !isTyping(e.target)) {
        setShortcuts(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-surface-3 focus:px-3 focus:py-2 focus:text-xs">
        Skip to content
      </a>
      <Sidebar />
      <div className="relative flex min-w-0 flex-1 flex-col">
        <div className="graticule pointer-events-none absolute inset-x-0 top-0 h-[520px]" aria-hidden />
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-line bg-bg/70 px-4 backdrop-blur-md sm:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" aria-label="Open navigation" onClick={() => setNav(true)}>
            <Menu className="size-4" />
          </Button>
          <LogoMark className="size-5 lg:hidden" />
          <div className="flex min-w-0 items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em]">
            <span className="hidden text-fg-subtle sm:inline">RAG Forge</span>
            <span className="hidden text-fg-subtle sm:inline">/</span>
            <span className="truncate text-fg">{area?.label ?? "Not found"}</span>
          </div>
          <button
            type="button"
            onClick={() => setPalette(true)}
            className="ml-auto flex h-8 w-full max-w-72 items-center gap-2 rounded-md border border-line bg-surface/80 px-2.5 text-xs text-fg-subtle transition-colors hover:border-line-strong hover:text-fg-muted"
          >
            <Search className="size-3.5" />
            <span className="flex-1 text-left">Search or jump to…</span>
            <Kbd>⌘K</Kbd>
          </button>
          <Button variant="ghost" size="icon" aria-label="Keyboard shortcuts" onClick={() => setShortcuts(true)} className="hidden sm:inline-flex">
            <Keyboard className="size-4" />
          </Button>
        </header>
        <main id="main" className="relative flex-1 px-4 py-6 sm:px-6 lg:px-8">
          {children}
        </main>
      </div>

      <CommandPalette open={palette} onOpenChange={setPalette} onShowShortcuts={() => setShortcuts(true)} />

      <Dialog open={shortcuts} onClose={() => setShortcuts(false)} title="Keyboard shortcuts">
        <ul className="divide-y divide-line text-[13px]">
          {SHORTCUTS.map(([keys, label]) => (
            <li key={label} className="flex items-center justify-between py-2.5">
              <span className="text-fg-muted">{label}</span>
              <span className="flex gap-1">{keys.map((k) => <Kbd key={k}>{k}</Kbd>)}</span>
            </li>
          ))}
        </ul>
      </Dialog>

      <Dialog variant="drawer" open={nav} onClose={() => setNav(false)} title="Navigate">
        <SidebarNav onNavigate={() => setNav(false)} />
      </Dialog>
    </div>
  );
}
