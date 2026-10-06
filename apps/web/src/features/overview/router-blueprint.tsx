"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { cn } from "@/lib/cn";
import { BUILD_STATE_LABEL, type BuildState, GraphNode } from "@/components/ui/pipeline";

/*
 * The RAG ROUTER architecture as a navigable diagram. It shows the design and its build
 * state; it is not a live trace. Live traces will reuse this layout with real RouterDecisions.
 */

interface Node {
  id: string;
  label: string;
  sub?: string;
  x: number;
  y: number;
  color?: string;
  state: BuildState;
  detail: string;
}

const MID = 118;
const H = 40;

const NODES: Node[] = [
  { id: "query", label: "Query", sub: "text + meta", x: 8, y: MID, state: "implemented", detail: "The user question and its metadata become a Query with a stable id; everything downstream references it." },
  { id: "analyze", label: "Analyze", sub: "features", x: 140, y: MID, state: "implemented", detail: "Extract query features: length, entities, temporal cues, comparison intent, expected hop count." },
  { id: "route", label: "Route", sub: "policy", x: 272, y: MID, color: "var(--color-signal)", state: "implemented", detail: "A router policy maps features to a weighted set of strategies and records a RouterDecision with its rationale." },
  { id: "sparse", label: "BM25", sub: "sparse", x: 432, y: 16, color: "var(--color-s-sparse)", state: "implemented", detail: "Lexical retrieval. Strong on exact terms, identifiers and rare words." },
  { id: "dense", label: "Dense", sub: "vector", x: 432, y: 84, color: "var(--color-s-dense)", state: "implemented", detail: "Embedding similarity. Strong on paraphrase and semantic matches." },
  { id: "metadata", label: "Metadata", sub: "filters", x: 432, y: 152, color: "var(--color-s-metadata)", state: "contract", detail: "Structured filters on document metadata: dates, sources, types." },
  { id: "graph", label: "Graph", sub: "entities", x: 432, y: 220, color: "var(--color-s-graph)", state: "contract", detail: "Traverse entity and relation links to reach evidence that no single chunk contains." },
  { id: "fuse", label: "Fuse", sub: "RRF", x: 592, y: MID, color: "var(--color-s-hybrid)", state: "implemented", detail: "Merge candidate lists, e.g. reciprocal rank fusion, keeping each candidate's per-retriever rank." },
  { id: "rerank", label: "Rerank", sub: "cross-enc", x: 724, y: MID, color: "var(--color-s-rerank)", state: "implemented", detail: "Re-score fused candidates with a stronger, slower model; rank movement is recorded." },
  { id: "verify", label: "Verify", sub: "sufficiency", x: 856, y: MID, state: "planned", detail: "Check that evidence can support an answer. If not, loop back to Route for another hop." },
  { id: "generate", label: "Generate", sub: "grounded", x: 988, y: MID, color: "var(--color-trace)", state: "implemented", detail: "Generate from selected evidence only, then measure each claim's support by that evidence (Evidence Lab)." },
];

const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));
const W = 104;
const STRATEGIES = ["sparse", "dense", "metadata", "graph"];

const right = (n: Node) => [n.x + W, n.y + H / 2] as const;
const left = (n: Node) => [n.x, n.y + H / 2] as const;
const curve = (a: readonly [number, number], b: readonly [number, number]) => {
  const dx = (b[0] - a[0]) / 2;
  return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`;
};

const EDGES: { id: string; from: string; to: string; color?: string }[] = [
  { id: "e-q-a", from: "query", to: "analyze" },
  { id: "e-a-r", from: "analyze", to: "route" },
  ...STRATEGIES.map((s) => ({ id: `e-r-${s}`, from: "route", to: s, color: byId[s].color })),
  ...STRATEGIES.map((s) => ({ id: `e-${s}-f`, from: s, to: "fuse", color: byId[s].color })),
  { id: "e-f-rr", from: "fuse", to: "rerank" },
  { id: "e-rr-v", from: "rerank", to: "verify" },
  { id: "e-v-g", from: "verify", to: "generate" },
];

// Multi-hop loop: Verify back to Route, drawn under the main line.
const LOOP = `M${byId.verify.x + W / 2},${MID + H} C${byId.verify.x + W / 2},${MID + H + 180} ${byId.route.x + W / 2},${MID + H + 180} ${byId.route.x + W / 2},${MID + H}`;

export function RouterBlueprint({ className }: { className?: string }) {
  const [selected, setSelected] = useState("route");
  const reduce = useReducedMotion();
  const node = byId[selected];
  const connected = new Set(
    EDGES.filter((e) => e.from === selected || e.to === selected).flatMap((e) => [e.from, e.to]),
  );

  return (
    <Panel className={cn("overflow-hidden", className)}>
      <PanelHeader
        eyebrow="RAG Router · architecture"
        title="Query → strategy selection → evidence → answer"
        actions={<Badge tone="trace" title="Design diagram with build state. No queries are flowing.">blueprint</Badge>}
      />
      <div className="overflow-x-auto px-2 pt-3">
        <svg viewBox="0 0 1100 330" className="min-w-[760px]" role="group" aria-label="Router pipeline diagram">
          <defs>
            {EDGES.map((e) => (
              <path key={e.id} id={e.id} d={curve(right(byId[e.from]), left(byId[e.to]))} />
            ))}
            <path id="e-loop" d={LOOP} />
          </defs>

          {EDGES.map((e) => {
            const hot = e.from === selected || e.to === selected;
            return (
              <use
                key={e.id}
                href={`#${e.id}`}
                fill="none"
                stroke={hot ? (e.color ?? "var(--color-fg-muted)") : "var(--color-line-strong)"}
                strokeWidth={hot ? 1.5 : 1}
                className="transition-[stroke] duration-300"
              />
            );
          })}
          <use href="#e-loop" fill="none" stroke="var(--color-line-strong)" strokeDasharray="4 4" />
          <text x={(byId.route.x + byId.verify.x) / 2 + W / 2} y={MID + H + 158} textAnchor="middle" fill="var(--color-fg-subtle)" fontSize={10} fontFamily="var(--font-mono)">
            multi-hop: evidence insufficient → re-route
          </text>

          {!reduce &&
            EDGES.map((e, i) => (
              <circle key={e.id} r={2.2} fill={e.color ?? "var(--color-fg-muted)"} opacity={0.9}>
                <animateMotion dur="2.8s" begin={`-${(i % 5) * 0.55}s`} repeatCount="indefinite" calcMode="spline" keyTimes="0;1" keySplines="0.4 0 0.2 1">
                  <mpath href={`#${e.id}`} />
                </animateMotion>
              </circle>
            ))}

          {NODES.map((n) => (
            <GraphNode
              key={n.id}
              x={n.x}
              y={n.y}
              w={W}
              h={H}
              label={n.label}
              sublabel={n.sub}
              color={n.color}
              state={n.state}
              active={n.id === selected}
              dimmed={n.id !== selected && !connected.has(n.id)}
              onSelect={() => setSelected(n.id)}
            />
          ))}
        </svg>
      </div>

      <div className="flex min-h-[76px] flex-col gap-3 border-t border-line px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <AnimatePresence mode="wait">
          <motion.div
            key={node.id}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="min-w-0"
            aria-live="polite"
          >
            <div className="flex items-center gap-2 text-[13px] font-medium">
              <span className="size-2 rounded-sm" style={{ background: node.color ?? "var(--color-fg-muted)" }} />
              {node.label}
              <span className="font-mono text-[10px] font-normal uppercase tracking-wider text-fg-subtle">
                {BUILD_STATE_LABEL[node.state]}
              </span>
            </div>
            <p className="mt-1 max-w-2xl text-xs leading-relaxed text-fg-muted">{node.detail}</p>
          </motion.div>
        </AnimatePresence>
        <Legend />
      </div>
    </Panel>
  );
}

function Legend() {
  const items: [string, string, boolean][] = [
    ["Implemented", "var(--color-ok)", false],
    ["Contract", "var(--color-trace)", false],
    ["Planned", "var(--color-idle)", true],
  ];
  return (
    <ul className="flex shrink-0 gap-3 font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
      {items.map(([label, color, dashed]) => (
        <li key={label} className="flex items-center gap-1.5">
          <span
            className="size-2 rounded-full"
            style={{ background: color, outline: dashed ? "1px dashed var(--color-line-strong)" : undefined, outlineOffset: 2 }}
          />
          {label}
        </li>
      ))}
    </ul>
  );
}
