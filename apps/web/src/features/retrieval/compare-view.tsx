"use client";

import type { RetrievalResponse } from "@rag-forge/shared";
import { ArrowDown, ArrowUp } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Segmented } from "@/components/ui/input";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { Tabs } from "@/components/ui/tabs";
import { cn } from "@/lib/cn";
import {
  type AlignedChunk,
  RUN_COLOR,
  RUN_LABEL,
  RUN_ORDER,
  type RunKey,
  type Runs,
  alignRuns,
  fusionEffect,
  overlap,
} from "./compare";

type View = "sparse-dense" | "sparse-hybrid" | "dense-hybrid" | "all";
type HybridKey = "rrf" | "weighted";

const VIEWS = [
  { id: "sparse-dense", label: "BM25 vs Dense" },
  { id: "sparse-hybrid", label: "BM25 vs Hybrid" },
  { id: "dense-hybrid", label: "Dense vs Hybrid" },
  { id: "all", label: "All strategies" },
];

function keysFor(view: View, hybrid: HybridKey): RunKey[] {
  if (view === "sparse-dense") return ["sparse", "dense"];
  if (view === "sparse-hybrid") return ["sparse", hybrid];
  if (view === "dense-hybrid") return [hybrid, "dense"];
  return RUN_ORDER;
}

const label = (c: AlignedChunk) => `${c.hit.filename} #${c.hit.chunk.ordinal}`;
const AXIS_LABEL: Record<RunKey, string> = { sparse: "BM25", rrf: "RRF", weighted: "Weighted", dense: "Dense" };

/** Side-by-side inspection of up to four retrieval runs of one query. Not an evaluation. */
export function CompareView({ runs }: { runs: Required<Runs> }) {
  const [view, setView] = useState<View>("all");
  const [hybrid, setHybrid] = useState<HybridKey>("rrf");
  const [focus, setFocus] = useState<string | null>(null);
  const keys = keysFor(view, hybrid);
  const rows = alignRuns(runs, keys);
  const k = Math.max(...keys.map((key) => runs[key].hits.length), 1);
  const any = runs.sparse;

  return (
    <div className="space-y-5">
      <Panel>
        <PanelHeader
          eyebrow={`Comparison · corpus v${any.provenance.corpus_version}`}
          title={`“${any.query.text}”`}
          actions={<Tabs tabs={VIEWS} value={view} onChange={(v) => setView(v as View)} />}
        />
        {view !== "all" && view !== "sparse-dense" && (
          <div className="flex items-center gap-3 border-b border-line px-4 py-2 font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
            hybrid run
            <Segmented name="hybrid-kind" value={hybrid} options={[{ value: "rrf", label: "RRF" }, { value: "weighted", label: "Weighted" }]} onChange={setHybrid} />
          </div>
        )}
        {view === "all" ? <AllSummary runs={runs} /> : <PairSummary rows={rows} a={keys[0]} b={keys[1]} />}
      </Panel>

      <div className="grid items-start gap-5 2xl:grid-cols-2">
        <Panel>
          <PanelHeader eyebrow="Ranking movement" title="Where each chunk ranks in each run" />
          <RankFlow rows={rows} keys={keys} k={k} focus={focus} onFocus={setFocus} />
        </Panel>
        <div className="space-y-5">
          <Panel>
            <PanelHeader eyebrow="Scores" title="Each run on its own scale" />
            <div className="space-y-4 p-4">
              {keys.map((key) => (
                <Strip key={key} run={key} response={runs[key]} focus={focus} onFocus={setFocus} />
              ))}
              <p className="text-[10px] leading-relaxed text-fg-subtle">
                BM25 scores are unbounded, dense scores are cosine similarities and fused scores are RRF sums or weighted
                0..1 values. Positions within a row are comparable; values across rows are not.
              </p>
            </div>
          </Panel>
          <Panel>
            <PanelHeader eyebrow="Latency" title="Server-side time per request" />
            <Latency runs={runs} keys={keys} />
          </Panel>
        </div>
      </div>

      {view === "all" ? (
        <Panel>
          <PanelHeader eyebrow="Rank matrix" title="Every retrieved chunk across all four runs" />
          <RankMatrix rows={rows} focus={focus} onFocus={setFocus} />
        </Panel>
      ) : (
        <div className="grid items-start gap-5 lg:grid-cols-2">
          {keys.map((key) => (
            <SideList key={key} run={key} other={keys.find((x) => x !== key)!} response={runs[key]} rows={rows} focus={focus} onFocus={setFocus} />
          ))}
        </div>
      )}
    </div>
  );
}

function Metric({ label: l, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-surface px-4 py-3">
      <div className="num text-xl text-fg" style={color ? { color } : undefined}>{value}</div>
      <div className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
        {l}
        {sub && <span className="normal-case tracking-normal"> · {sub}</span>}
      </div>
    </div>
  );
}

function PairSummary({ rows, a, b }: { rows: AlignedChunk[]; a: RunKey; b: RunKey }) {
  const o = overlap(rows, a, b);
  const total = Math.max(rows.length, 1);
  const parts: [string, number, string | undefined][] = [
    [`${RUN_LABEL[a]} only`, o.aOnly, RUN_COLOR[a]],
    ["shared", o.shared, undefined],
    [`${RUN_LABEL[b]} only`, o.bOnly, RUN_COLOR[b]],
  ];
  return (
    <>
      <div className="grid gap-px bg-line sm:grid-cols-4">
        <Metric label={`${RUN_LABEL[a]} only`} value={`${o.aOnly}`} color={RUN_COLOR[a]} />
        <Metric label="shared" value={`${o.shared}`} sub={`of ${rows.length} distinct`} />
        <Metric label={`${RUN_LABEL[b]} only`} value={`${o.bOnly}`} color={RUN_COLOR[b]} />
        <Metric label="same rank" value={o.shared ? `${o.sameRank}/${o.shared}` : "—"} sub="shared chunks" />
      </div>
      <div className="border-t border-line px-4 py-3">
        <div className="flex h-2.5 overflow-hidden rounded-full bg-surface-3" role="img" aria-label={parts.map(([l, n]) => `${n} ${l}`).join(", ")}>
          {parts.map(([l, n, color]) => (
            <motion.span
              key={l}
              className={cn("h-full", !color && "hatch")}
              style={{ background: color }}
              initial={{ width: 0 }}
              animate={{ width: `${(n / total) * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          ))}
        </div>
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-fg-subtle">
          {parts.map(([l, n, color]) => (
            <li key={l} className="flex items-center gap-1.5">
              <span className={cn("size-2 rounded-sm", !color && "hatch bg-surface-3")} style={{ background: color }} />
              {l} <span className="num text-fg-muted">{n}</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

function AllSummary({ runs }: { runs: Required<Runs> }) {
  const rows = alignRuns(runs);
  const base = overlap(rows, "sparse", "dense");
  const effects = (h: HybridKey) => ({
    promoted: rows.filter((r) => fusionEffect(r, h) === "promoted").length,
    dropped: rows.filter((r) => fusionEffect(r, h) === "dropped").length,
  });
  const rrf = effects("rrf");
  const weighted = effects("weighted");
  return (
    <div className="grid gap-px bg-line sm:grid-cols-4">
      <Metric label="BM25 ∩ Dense" value={`${base.shared}`} sub={`${base.aOnly} BM25-only · ${base.bOnly} dense-only`} />
      <Metric label="distinct chunks" value={`${rows.length}`} sub="across all runs" />
      <Metric label="RRF effects" value={`+${rrf.promoted} / −${rrf.dropped}`} sub="promoted / dropped" color={RUN_COLOR.rrf} />
      <Metric label="Weighted effects" value={`+${weighted.promoted} / −${weighted.dropped}`} sub="promoted / dropped" color={RUN_COLOR.weighted} />
    </div>
  );
}

/** Parallel rank axes. A line joins a chunk's positions in adjacent runs; lone dots are unique to a run. */
function RankFlow({ rows, keys, k, focus, onFocus }: { rows: AlignedChunk[]; keys: RunKey[]; k: number; focus: string | null; onFocus: (id: string | null) => void }) {
  const row = 26;
  const top = 30;
  const h = top + k * row + 4;
  const left = 190;
  const right = 470;
  const x = (i: number) => (keys.length === 1 ? left : left + ((right - left) * i) / (keys.length - 1));
  const y = (rank: number) => top + (rank - 0.5) * row;
  return (
    <div className="overflow-x-auto p-2">
      <svg viewBox={`0 0 650 ${h}`} className="min-w-[560px]" role="img" aria-label={`Rank comparison across ${keys.map((key) => RUN_LABEL[key]).join(", ")}`}>
        {keys.map((key, i) => (
          <g key={key}>
            <line x1={x(i)} x2={x(i)} y1={top - 6} y2={h - 4} stroke="var(--color-line)" />
            <text x={x(i)} y={14} textAnchor="middle" fontSize={10} fill={RUN_COLOR[key]} fontFamily="var(--font-mono)">
              {AXIS_LABEL[key]}
            </text>
          </g>
        ))}
        {rows.map((r, n) => {
          const dim = focus !== null && focus !== r.chunkId;
          const segments = keys.slice(1).map((key, i) => {
            const a = r.ranks[keys[i]];
            const b = r.ranks[key];
            return a && b ? `M${x(i)},${y(a)} L${x(i + 1)},${y(b)}` : null;
          });
          const first = keys.findIndex((key) => r.ranks[key]);
          const last = keys.length - 1 - [...keys].reverse().findIndex((key) => r.ranks[key]);
          return (
            <g
              key={r.chunkId}
              opacity={dim ? 0.15 : 1}
              onMouseEnter={() => onFocus(r.chunkId)}
              onMouseLeave={() => onFocus(null)}
              className="transition-opacity duration-200"
            >
              <title>{`${label(r)} · ${keys.map((key) => `${RUN_LABEL[key]} ${r.ranks[key] ? `#${r.ranks[key]}` : "—"}`).join(" · ")}`}</title>
              {segments.map(
                (d, i) =>
                  d && (
                    <motion.path
                      key={`${r.chunkId}-${i}`}
                      d={d}
                      fill="none"
                      stroke="var(--color-fg-muted)"
                      strokeWidth={focus === r.chunkId ? 2 : 1.1}
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.5, delay: Math.min(n, 12) * 0.03, ease: "easeOut" }}
                    />
                  ),
              )}
              {keys.map((key, i) => {
                const rank = r.ranks[key];
                if (!rank) return null;
                const linked = Boolean((i > 0 && r.ranks[keys[i - 1]]) || (i < keys.length - 1 && r.ranks[keys[i + 1]]));
                return <circle key={key} cx={x(i)} cy={y(rank)} r={4.5} fill={linked ? "var(--color-surface)" : RUN_COLOR[key]} stroke={RUN_COLOR[key]} strokeWidth={1.5} />;
              })}
              {first === 0 && (
                <text x={x(0) - 12} y={y(r.ranks[keys[0]]!) + 3.5} textAnchor="end" fontSize={10.5} fill="var(--color-fg-muted)" fontFamily="var(--font-mono)">
                  {truncate(label(r), 24)}
                </text>
              )}
              {last === keys.length - 1 && (first !== 0 || keys.length > 1) && (
                <text x={x(keys.length - 1) + 12} y={y(r.ranks[keys[keys.length - 1]]!) + 3.5} fontSize={10.5} fill="var(--color-fg-muted)" fontFamily="var(--font-mono)">
                  {truncate(label(r), 24)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

const truncate = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function Strip({ run, response, focus, onFocus }: { run: RunKey; response: RetrievalResponse; focus: string | null; onFocus: (id: string | null) => void }) {
  const scores = response.hits.map((h) => h.result.score);
  const lo = Math.min(...scores, 0);
  const hi = Math.max(...scores, lo + 1e-9);
  const pct = (s: number) => ((s - lo) / (hi - lo)) * 100;
  const digits = run === "rrf" ? 4 : 3;
  return (
    <div>
      <div className="mb-1.5 flex justify-between font-mono text-[10px] text-fg-subtle">
        <span style={{ color: RUN_COLOR[run] }}>{RUN_LABEL[run]}</span>
        <span className="num">{scores.length ? `0 – ${hi.toFixed(digits)}` : "no results"}</span>
      </div>
      <div className="relative h-5 rounded bg-surface-3">
        {response.hits.map((h) => (
          <span
            key={h.result.chunk_id}
            title={`#${h.result.rank} ${h.filename} · ${h.result.score.toFixed(5)}`}
            onMouseEnter={() => onFocus(h.result.chunk_id)}
            onMouseLeave={() => onFocus(null)}
            className={cn("absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-surface transition-transform", focus === h.result.chunk_id && "scale-150")}
            style={{ left: `${pct(h.result.score)}%`, background: RUN_COLOR[run], opacity: focus && focus !== h.result.chunk_id ? 0.25 : 0.9 }}
          />
        ))}
      </div>
    </div>
  );
}

function Latency({ runs, keys }: { runs: Required<Runs>; keys: RunKey[] }) {
  const max = Math.max(...keys.map((key) => runs[key].provenance.elapsed_ms), 1e-6);
  return (
    <div className="space-y-3 p-4">
      {keys.map((key) => {
        const p = runs[key].provenance;
        const embed = p.statistics.query_embedding_ms ?? p.statistics["dense.query_embedding_ms"];
        return (
          <div key={key}>
            <div className="mb-1 flex justify-between font-mono text-[11px]">
              <span style={{ color: RUN_COLOR[key] }}>{RUN_LABEL[key]}</span>
              <span className="num text-fg">{p.elapsed_ms.toFixed(1)} ms</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
              <motion.div className="h-full rounded-full" style={{ background: RUN_COLOR[key] }} initial={{ width: 0 }} animate={{ width: `${(p.elapsed_ms / max) * 100}%` }} transition={{ duration: 0.5, ease: "easeOut" }} />
            </div>
            {embed !== undefined && <div className="mt-1 font-mono text-[10px] text-fg-subtle">incl. query embedding {embed.toFixed(1)} ms</div>}
          </div>
        );
      })}
      <p className="text-[10px] leading-relaxed text-fg-subtle">One request each, measured server-side; not a benchmark.</p>
    </div>
  );
}

function Delta({ from, to }: { from?: number; to?: number }) {
  if (!from || !to || from === to) return null;
  const up = to < from;
  const Icon = up ? ArrowUp : ArrowDown;
  return (
    <span className={cn("inline-flex items-center text-[10px]", up ? "text-ok" : "text-err")} title={`rank ${from} → ${to}`}>
      <Icon className="size-3" />
      {Math.abs(from - to)}
    </span>
  );
}

function RankMatrix({ rows, focus, onFocus }: { rows: AlignedChunk[]; focus: string | null; onFocus: (id: string | null) => void }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <caption className="sr-only">Rank of each chunk in each run; arrows show hybrid rank against the better single-strategy rank</caption>
        <thead>
          <tr className="border-b border-line font-mono text-[10px] uppercase tracking-[0.12em] text-fg-subtle">
            <th className="h-8 px-4 font-normal">Chunk</th>
            {RUN_ORDER.map((key) => (
              <th key={key} className="h-8 px-3 text-right font-normal" style={{ color: RUN_COLOR[key] }}>{RUN_LABEL[key]}</th>
            ))}
            <th className="h-8 px-4 font-normal">Fusion effect</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const best = Math.min(r.ranks.sparse ?? Infinity, r.ranks.dense ?? Infinity);
            const effects = (["rrf", "weighted"] as const).map((h) => [h, fusionEffect(r, h)] as const).filter(([, e]) => e);
            return (
              <tr
                key={r.chunkId}
                onMouseEnter={() => onFocus(r.chunkId)}
                onMouseLeave={() => onFocus(null)}
                className={cn("border-b border-line/60 transition-colors last:border-0", focus === r.chunkId && "bg-surface-3")}
              >
                <td className="h-10 px-4">
                  <div className="truncate font-mono text-[12px] text-fg">{label(r)}</div>
                </td>
                {RUN_ORDER.map((key) => (
                  <td key={key} className="h-10 px-3 text-right">
                    <span className="inline-flex items-center justify-end gap-1.5">
                      {(key === "rrf" || key === "weighted") && <Delta from={Number.isFinite(best) ? best : undefined} to={r.ranks[key]} />}
                      <span className="num font-mono text-fg">{r.ranks[key] ? `#${r.ranks[key]}` : <span className="text-fg-subtle">—</span>}</span>
                    </span>
                  </td>
                ))}
                <td className="h-10 px-4">
                  <span className="flex gap-1">
                    {effects.map(([h, e]) => (
                      <Badge key={h} tone={e === "promoted" ? "ok" : "err"} title={e === "promoted" ? "In the hybrid top-k but in neither single top-k" : "In a single top-k but not in the hybrid top-k"}>
                        {RUN_LABEL[h].replace("Hybrid ", "")} {e}
                      </Badge>
                    ))}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SideList({ run, other, response, rows, focus, onFocus }: { run: RunKey; other: RunKey; response: RetrievalResponse; rows: AlignedChunk[]; focus: string | null; onFocus: (id: string | null) => void }) {
  return (
    <Panel>
      <PanelHeader eyebrow={RUN_LABEL[run]} title={`${response.hits.length} results`} actions={<span className="size-2 rounded-full" style={{ background: RUN_COLOR[run] }} />} />
      <ol>
        {response.hits.map((h) => {
          const row = rows.find((x) => x.chunkId === h.result.chunk_id)!;
          const otherRank = row.ranks[other];
          return (
            <li
              key={h.result.chunk_id}
              onMouseEnter={() => onFocus(row.chunkId)}
              onMouseLeave={() => onFocus(null)}
              className={cn("border-b border-line/60 px-4 py-2.5 transition-colors last:border-0", focus === row.chunkId && "bg-surface-3")}
            >
              <div className="flex items-center gap-2 text-xs">
                <span className="num w-6 font-mono" style={{ color: RUN_COLOR[run] }}>#{h.result.rank}</span>
                <span className="min-w-0 flex-1 truncate font-mono text-fg">{label(row)}</span>
                <span className="num font-mono text-[11px] text-fg-muted">{h.result.score.toFixed(run === "rrf" ? 5 : 3)}</span>
                {otherRank ? (
                  <Badge>
                    {RUN_LABEL[other]} #{otherRank} <Delta from={otherRank} to={h.result.rank} />
                  </Badge>
                ) : (
                  <Badge tone={run === "dense" ? "trace" : "signal"}>{RUN_LABEL[run]} only</Badge>
                )}
              </div>
              <p className="mt-1 line-clamp-2 pl-8 text-[11px] leading-relaxed text-fg-subtle">{h.chunk.text}</p>
            </li>
          );
        })}
      </ol>
    </Panel>
  );
}
