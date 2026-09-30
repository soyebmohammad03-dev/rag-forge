"use client";

import { useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { OriginBadge } from "@/components/ui/badge";
import { ChartTooltip, STRATEGY_COLOR, chartAxis } from "@/components/ui/chart";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { cn } from "@/lib/cn";
import { SAMPLE_ORIGIN, STRATEGY_KEYS, type StrategyKey, sampleActivity } from "@/sample/preview-data";

export function RetrievalActivity({ className }: { className?: string }) {
  const [hidden, setHidden] = useState<Set<StrategyKey>>(new Set());
  const toggle = (k: StrategyKey) =>
    setHidden((h) => {
      const next = new Set(h);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });

  return (
    <Panel className={className}>
      <PanelHeader
        eyebrow="Retrieval activity · 24h"
        title="Queries served by strategy"
        actions={<OriginBadge origin={SAMPLE_ORIGIN} />}
      />
      <div className="flex flex-wrap gap-1.5 px-4 pt-3" role="group" aria-label="Toggle strategies">
        {STRATEGY_KEYS.map((k) => (
          <button
            key={k}
            type="button"
            aria-pressed={!hidden.has(k)}
            onClick={() => toggle(k)}
            className={cn(
              "flex h-6 items-center gap-1.5 rounded px-2 font-mono text-[10px] uppercase tracking-wider ring-1 ring-inset ring-line-strong transition-opacity",
              hidden.has(k) ? "opacity-40" : "text-fg-muted",
            )}
          >
            <span className="size-2 rounded-sm" style={{ background: STRATEGY_COLOR[k] }} />
            {k}
          </button>
        ))}
      </div>
      <div className="h-60 px-2 pb-2 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={sampleActivity} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
            <defs>
              {STRATEGY_KEYS.map((k) => (
                <linearGradient key={k} id={`act-${k}`} x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0" stopColor={STRATEGY_COLOR[k]} stopOpacity={0.35} />
                  <stop offset="1" stopColor={STRATEGY_COLOR[k]} stopOpacity={0.02} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid vertical={false} stroke="var(--color-line)" strokeDasharray="2 4" />
            <XAxis dataKey="hour" {...chartAxis} interval={3} />
            <YAxis {...chartAxis} width={40} />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--color-line-strong)" }} />
            {STRATEGY_KEYS.filter((k) => !hidden.has(k)).map((k) => (
              <Area
                key={k}
                type="monotone"
                dataKey={k}
                name={k}
                stackId="1"
                stroke={STRATEGY_COLOR[k]}
                strokeWidth={1.25}
                fill={`url(#act-${k})`}
                animationDuration={700}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}
