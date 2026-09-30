"use client";

import type { ComponentState } from "@rag-forge/shared";
import { RefreshCw } from "lucide-react";
import { OriginBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { ErrorState, LoadingState } from "@/components/ui/states";
import { type Status, StatusDot } from "@/components/ui/status";
import { fetchHealth, useApi } from "@/lib/use-api";

const STATE: Record<ComponentState, { status: Status; label: string }> = {
  ok: { status: "ok", label: "ok" },
  not_configured: { status: "idle", label: "not configured" },
  error: { status: "err", label: "error" },
};

export function formatUptime(seconds: number) {
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

export function HealthPanel({ className }: { className?: string }) {
  const { data, error, loading, reload } = useApi(fetchHealth, 10_000);

  return (
    <Panel className={className}>
      <PanelHeader
        eyebrow="System"
        title="Component health"
        actions={
          <>
            {data && <OriginBadge origin="live" />}
            <Button variant="ghost" size="icon" aria-label="Refresh health" onClick={reload}>
              <RefreshCw className="size-3.5" />
            </Button>
          </>
        }
      />
      {loading && !data && <LoadingState rows={5} label="Loading component health" />}
      {error && !data && (
        <ErrorState title="Backend unreachable" action={<Button size="sm" onClick={reload}>Retry</Button>}>
          {error}. Start it with <code className="font-mono text-fg">npm run dev:api</code>.
        </ErrorState>
      )}
      {data && (
        <>
          <ul className="divide-y divide-line/60">
            {data.components.map((c) => (
              <li key={c.name} className="flex items-center gap-3 px-4 py-2.5">
                <StatusDot status={STATE[c.state].status} />
                <div className="min-w-0 flex-1">
                  <div className="font-mono text-xs text-fg">{c.name}</div>
                  <div className="truncate text-[11px] text-fg-subtle">{c.detail}</div>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-fg-muted">
                  {STATE[c.state].label}
                </span>
              </li>
            ))}
          </ul>
          <div className="flex justify-between border-t border-line px-4 py-2 font-mono text-[10px] text-fg-subtle">
            <span>v{data.version}</span>
            <span className="num">uptime {formatUptime(data.uptime_seconds)}</span>
          </div>
        </>
      )}
    </Panel>
  );
}
