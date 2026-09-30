"use client";

import type { RetrievalResponse } from "@rag-forge/shared";
import { motion } from "motion/react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { cn } from "@/lib/cn";
import { type ComparedChunk, type Membership, compareRuns } from "./compare";

const COLOR = { sparse: "var(--color-s-sparse)", dense: "var(--color-s-dense)", both: "var(--color-fg-muted)" };
const LABEL: Record<Membership, string> = { both: "both", sparse: "BM25 only", dense: "dense only" };
const label = (c: ComparedChunk) => `${c.hit.filename} #${c.hit.chunk.ordinal}`;

/** BM25 vs dense for one query, corpus and version: inspection only, no quality metrics. */
export function CompareView({ sparse, dense }: { sparse: RetrievalResponse; dense: RetrievalResponse }) {
  const cmp = compareRuns(sparse, dense);
  const [focus, setFocus] = useState<string | null>(null);
  const total = cmp.chunks.length;

  return (
    <div className="space-y-5">
      <Panel>
        <PanelHeader eyebrow="Comparison" title={`“${sparse.query.text}”`} actions={<Badge tone="trace">v{sparse.provenance.corpus_version}</Badge>} />
        <div className="grid gap-px bg-line sm:grid-cols-4">
          <Metric label="shared chunks" value={`${cmp.shared}`} sub={`of ${total} distinct`} />
          <Metric label="BM25 only" value={`${cmp.sparseOnly}`} color={COLOR.sparse} />
          <Metric label="dense only" value={`${cmp.denseOnly}`} color={COLOR.dense} />
          <Metric label="rank agreement" value={agreement(cmp.chunks)} sub="shared, same rank" />
        </div>
        <OverlapBar cmp={cmp} />
      </Panel>

      <div className="grid items-start gap-5 2xl:grid-cols-[1fr_1fr]">
        <Panel>
          <PanelHeader eyebrow="Ranking" title="Where each chunk ranks in each run" />
          <SlopeChart chunks={cmp.chunks} k={Math.max(sparse.hits.length, dense.hits.length)} focus={focus} onFocus={setFocus} />
        </Panel>
        <div className="space-y-5">
          <Panel>
            <PanelHeader eyebrow="Scores" title="Score distributions (different scales)" />
            <div className="space-y-4 p-4">
              <Strip title="BM25 · unbounded, query-relative" color={COLOR.sparse} response={sparse} focus={focus} onFocus={setFocus} />
              <Strip title="Dense · cosine similarity" color={COLOR.dense} response={dense} focus={focus} onFocus={setFocus} />
            </div>
          </Panel>
          <Panel>
            <PanelHeader eyebrow="Latency" title="End-to-end retrieval time" />
            <Latency sparse={sparse} dense={dense} />
          </Panel>
        </div>
      </div>

      <div className="grid items-start gap-5 lg:grid-cols-2">
        {([["sparse", sparse], ["dense", dense]] as const).map(([side, run]) => (
          <Panel key={side}>
            <PanelHeader eyebrow={side === "sparse" ? "BM25" : "Dense"} title={`${run.hits.length} results`} actions={<span className="size-2 rounded-full" style={{ background: COLOR[side] }} />} />
            <ol>
              {run.hits.map((h) => {
                const c = cmp.chunks.find((x) => x.chunkId === h.result.chunk_id)!;
                const other = side === "sparse" ? c.denseRank : c.sparseRank;
                return (
                  <li
                    key={h.result.chunk_id}
                    onMouseEnter={() => setFocus(c.chunkId)}
                    onMouseLeave={() => setFocus(null)}
                    className={cn("border-b border-line/60 px-4 py-2.5 transition-colors last:border-0", focus === c.chunkId && "bg-surface-3")}
                  >
                    <div className="flex items-center gap-2 text-xs">
                      <span className="num w-6 font-mono" style={{ color: COLOR[side] }}>#{h.result.rank}</span>
                      <span className="min-w-0 flex-1 truncate font-mono text-fg">{label(c)}</span>
                      <span className="num font-mono text-[11px] text-fg-muted">{h.result.score.toFixed(3)}</span>
                      <Badge tone={c.kind === "both" ? "neutral" : side === "sparse" ? "signal" : "trace"}>
                        {c.kind === "both" ? `both · #${other}` : LABEL[c.kind]}
                      </Badge>
                    </div>
                    <p className="mt-1 line-clamp-2 pl-8 text-[11px] leading-relaxed text-fg-subtle">{h.chunk.text}</p>
                  </li>
                );
              })}
            </ol>
          </Panel>
        ))}
      </div>
    </div>
  );
}

function agreement(chunks: ComparedChunk[]) {
  const shared = chunks.filter((c) => c.kind === "both");
  if (!shared.length) return "—";
  return `${shared.filter((c) => c.sparseRank === c.denseRank).length}/${shared.length}`;
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

function OverlapBar({ cmp }: { cmp: ReturnType<typeof compareRuns> }) {
  const total = Math.max(cmp.chunks.length, 1);
  const parts: [Membership, number][] = [["sparse", cmp.sparseOnly], ["both", cmp.shared], ["dense", cmp.denseOnly]];
  return (
    <div className="border-t border-line px-4 py-3">
      <div className="flex h-2.5 overflow-hidden rounded-full bg-surface-3" role="img" aria-label={`${cmp.sparseOnly} BM25 only, ${cmp.shared} shared, ${cmp.denseOnly} dense only`}>
        {parts.map(([k, n]) => (
          <motion.span
            key={k}
            className={cn("h-full", k === "both" && "hatch")}
            style={{ background: k === "both" ? undefined : COLOR[k] }}
            initial={{ width: 0 }}
            animate={{ width: `${(n / total) * 100}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-fg-subtle">
        {parts.map(([k, n]) => (
          <li key={k} className="flex items-center gap-1.5">
            <span className={cn("size-2 rounded-sm", k === "both" && "hatch bg-surface-3")} style={{ background: k === "both" ? undefined : COLOR[k] }} />
            {LABEL[k]} <span className="num text-fg-muted">{n}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Two rank columns; a line joins a chunk retrieved by both. Unconnected dots are unique. */
function SlopeChart({ chunks, k, focus, onFocus }: { chunks: ComparedChunk[]; k: number; focus: string | null; onFocus: (id: string | null) => void }) {
  const row = 26;
  const top = 26;
  const h = top + Math.max(k, 1) * row;
  const xL = 190;
  const xR = 410;
  const y = (rank: number) => top + (rank - 0.5) * row;
  return (
    <div className="overflow-x-auto p-2">
      <svg viewBox={`0 0 600 ${h}`} className="min-w-[520px]" role="img" aria-label="Rank comparison between BM25 and dense retrieval">
        <text x={xL} y={14} textAnchor="middle" fontSize={10} fill={COLOR.sparse} fontFamily="var(--font-mono)">BM25 rank</text>
        <text x={xR} y={14} textAnchor="middle" fontSize={10} fill={COLOR.dense} fontFamily="var(--font-mono)">dense rank</text>
        {chunks.map((c, i) => {
          const dim = focus !== null && focus !== c.chunkId;
          const opacity = dim ? 0.15 : 1;
          return (
            <motion.g
              key={c.chunkId}
              initial={{ opacity: 0 }}
              animate={{ opacity }}
              transition={{ delay: Math.min(i, 12) * 0.03 }}
              onMouseEnter={() => onFocus(c.chunkId)}
              onMouseLeave={() => onFocus(null)}
              className="cursor-default"
            >
              <title>{`${label(c)} · BM25 ${c.sparseRank ? `#${c.sparseRank}` : "—"} · dense ${c.denseRank ? `#${c.denseRank}` : "—"}`}</title>
              {c.sparseRank && c.denseRank && (
                <line x1={xL} y1={y(c.sparseRank)} x2={xR} y2={y(c.denseRank)} stroke={COLOR.both} strokeWidth={focus === c.chunkId ? 2 : 1.25} />
              )}
              {c.sparseRank && (
                <>
                  <circle cx={xL} cy={y(c.sparseRank)} r={4.5} fill={c.kind === "both" ? "var(--color-surface)" : COLOR.sparse} stroke={COLOR.sparse} strokeWidth={1.5} />
                  <text x={xL - 12} y={y(c.sparseRank) + 3.5} textAnchor="end" fontSize={10.5} fill="var(--color-fg-muted)" fontFamily="var(--font-mono)">
                    {truncate(label(c), 24)}
                  </text>
                </>
              )}
              {c.denseRank && (
                <>
                  <circle cx={xR} cy={y(c.denseRank)} r={4.5} fill={c.kind === "both" ? "var(--color-surface)" : COLOR.dense} stroke={COLOR.dense} strokeWidth={1.5} />
                  <text x={xR + 12} y={y(c.denseRank) + 3.5} fontSize={10.5} fill="var(--color-fg-muted)" fontFamily="var(--font-mono)">
                    {truncate(label(c), 24)}
                  </text>
                </>
              )}
            </motion.g>
          );
        })}
      </svg>
    </div>
  );
}

const truncate = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function Strip({ title, color, response, focus, onFocus }: { title: string; color: string; response: RetrievalResponse; focus: string | null; onFocus: (id: string | null) => void }) {
  const scores = response.hits.map((h) => h.result.score);
  const lo = Math.min(...scores, 0);
  const hi = Math.max(...scores, lo + 1e-9);
  const pct = (s: number) => ((s - lo) / (hi - lo)) * 100;
  return (
    <div>
      <div className="mb-1.5 flex justify-between font-mono text-[10px] text-fg-subtle">
        <span>{title}</span>
        <span className="num">{scores.length ? `${lo.toFixed(2)} – ${hi.toFixed(3)}` : "no results"}</span>
      </div>
      <div className="relative h-6 rounded bg-surface-3">
        {response.hits.map((h) => (
          <span
            key={h.result.chunk_id}
            title={`#${h.result.rank} ${h.filename} · ${h.result.score.toFixed(4)}`}
            onMouseEnter={() => onFocus(h.result.chunk_id)}
            onMouseLeave={() => onFocus(null)}
            className={cn("absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-surface transition-transform", focus === h.result.chunk_id && "scale-150")}
            style={{ left: `${pct(h.result.score)}%`, background: color, opacity: focus && focus !== h.result.chunk_id ? 0.25 : 0.9 }}
          />
        ))}
      </div>
    </div>
  );
}

function Latency({ sparse, dense }: { sparse: RetrievalResponse; dense: RetrievalResponse }) {
  const max = Math.max(sparse.provenance.elapsed_ms, dense.provenance.elapsed_ms, 1e-6);
  const embed = dense.provenance.statistics.query_embedding_ms ?? 0;
  const rows = [
    { name: "BM25", ms: sparse.provenance.elapsed_ms, color: COLOR.sparse, note: sparse.provenance.statistics.indexed_now ? `incl. indexing ${sparse.provenance.statistics.indexed_now} chunks` : "" },
    { name: "Dense", ms: dense.provenance.elapsed_ms, color: COLOR.dense, note: `incl. query embedding ${embed.toFixed(1)} ms` },
  ];
  return (
    <div className="space-y-3 p-4">
      {rows.map((r) => (
        <div key={r.name}>
          <div className="mb-1 flex justify-between font-mono text-[11px]">
            <span style={{ color: r.color }}>{r.name}</span>
            <span className="num text-fg">{r.ms.toFixed(1)} ms</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
            <motion.div className="h-full rounded-full" style={{ background: r.color }} initial={{ width: 0 }} animate={{ width: `${(r.ms / max) * 100}%` }} transition={{ duration: 0.5, ease: "easeOut" }} />
          </div>
          {r.note && <div className="mt-1 font-mono text-[10px] text-fg-subtle">{r.note}</div>}
        </div>
      ))}
      <p className="text-[10px] leading-relaxed text-fg-subtle">Measured server-side for this single request; not a benchmark.</p>
    </div>
  );
}
