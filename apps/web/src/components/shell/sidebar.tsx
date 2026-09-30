"use client";

import { motion } from "motion/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { StatusDot } from "@/components/ui/status";
import { AREAS, AREA_GROUPS, areaHref } from "@/lib/areas";
import { cn } from "@/lib/cn";
import { fetchHealth, useApi } from "@/lib/use-api";
import { LogoMark } from "./logo";

export function isActive(pathname: string, slug: string) {
  return slug === "" ? pathname === "/" : pathname === `/${slug}` || pathname.startsWith(`/${slug}/`);
}

export function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Product areas" className="space-y-5">
      {AREA_GROUPS.map((group) => (
        <div key={group}>
          <div className="mb-1 px-3 font-mono text-[10px] uppercase tracking-[0.16em] text-fg-subtle">{group}</div>
          <ul>
            {AREAS.filter((a) => a.group === group).map((a) => {
              const active = isActive(pathname, a.slug);
              const Icon = a.icon;
              return (
                <li key={a.slug}>
                  <Link
                    href={areaHref(a)}
                    onClick={onNavigate}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "group relative flex h-8 items-center gap-2.5 rounded-md px-3 text-[13px] transition-colors",
                      active ? "text-fg" : "text-fg-muted hover:bg-surface-2 hover:text-fg",
                    )}
                  >
                    {active && (
                      <motion.span
                        layoutId="nav-active"
                        className="absolute inset-0 rounded-md bg-surface-3 shadow-[0_0_0_1px_var(--color-line-strong)]"
                        transition={{ type: "spring", stiffness: 500, damping: 40 }}
                      >
                        <span className="absolute inset-y-2 -left-px w-0.5 rounded-full bg-signal" />
                      </motion.span>
                    )}
                    <Icon className={cn("relative size-4", active ? "text-signal" : "text-fg-subtle group-hover:text-fg-muted")} />
                    <span className="relative flex-1">{a.label}</span>
                    {a.status === "planned" && (
                      <span className="relative font-mono text-[9px] uppercase tracking-wider text-fg-subtle" title="Planned">
                        soon
                      </span>
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}

function ApiStatus() {
  const { data, error } = useApi(fetchHealth, 15_000);
  return (
    <div className="flex items-center gap-2.5 rounded-md border border-line bg-surface-2/60 px-3 py-2">
      <StatusDot status={data ? (data.status === "ok" ? "ok" : "warn") : error ? "err" : "loading"} />
      <div className="min-w-0 text-[11px] leading-tight">
        <div className="text-fg">{data ? "API connected" : error ? "API offline" : "Connecting…"}</div>
        <div className="truncate font-mono text-[10px] text-fg-subtle">
          {data ? `v${data.version} · local` : "localhost:8000"}
        </div>
      </div>
    </div>
  );
}

export function Sidebar() {
  return (
    <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col border-r border-line bg-surface/60 backdrop-blur lg:flex">
      <Link href="/" className="flex h-14 items-center gap-2.5 px-5">
        <LogoMark className="size-6" />
        <span className="text-[13px] font-semibold tracking-[0.14em]">
          RAG<span className="text-signal"> FORGE</span>
        </span>
      </Link>
      <div className="flex-1 overflow-y-auto px-2 py-3">
        <SidebarNav />
      </div>
      <div className="p-3">
        <ApiStatus />
      </div>
    </aside>
  );
}
