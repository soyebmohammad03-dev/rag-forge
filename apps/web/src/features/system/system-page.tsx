"use client";

import { OriginBadge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { ErrorState, LoadingState } from "@/components/ui/states";
import { API_URL } from "@/lib/api";
import { fetchEnvironment, useApi } from "@/lib/use-api";
import { HealthPanel } from "./health-panel";

export function SystemPage() {
  const { data, error, loading } = useApi(fetchEnvironment);
  const env = data?.environment;

  const rows: [string, string][] = env
    ? [
        ["rag_forge", env.rag_forge_version],
        ["python", env.python_version],
        ["platform", env.platform],
        ["git_commit", env.git_commit ?? "unavailable"],
        ...Object.entries(env.packages).map(([k, v]): [string, string] => [k, v]),
        ["api_url", API_URL],
      ]
    : [];

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <header>
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Operate</p>
        <h1 className="mt-1.5 text-2xl font-medium tracking-tight">System</h1>
        <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-fg-muted">
          Component health, and the environment snapshot the provenance layer records into every run.
        </p>
      </header>
      <div className="grid gap-5 lg:grid-cols-2">
        <HealthPanel />
        <Panel>
          <PanelHeader
            eyebrow="Provenance"
            title="Environment snapshot"
            actions={env && <OriginBadge origin="live" />}
          />
          {loading && <LoadingState rows={6} label="Loading environment" />}
          {error && <ErrorState title="Environment unavailable">{error}</ErrorState>}
          {env && (
            <dl className="divide-y divide-line/60">
              {rows.map(([k, v]) => (
                <div key={k} className="grid grid-cols-[140px_1fr] gap-4 px-4 py-2.5 text-xs">
                  <dt className="font-mono text-fg-subtle">{k}</dt>
                  <dd className="truncate font-mono text-fg" title={v}>{v}</dd>
                </div>
              ))}
            </dl>
          )}
        </Panel>
      </div>
    </div>
  );
}
