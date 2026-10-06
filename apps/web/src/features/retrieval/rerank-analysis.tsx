"use client";

import type { RerankCandidate, RetrievalResponse } from "@rag-forge/shared";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { STRATEGY_COLOR } from "@/components/ui/chart";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { cn } from "@/lib/cn";
import { shortHash } from "@/lib/format";
import { MOVEMENT_COLOR, candidateLabel, crossings, formatMs, summarize, visible } from "./rerank";

const RERANK = STRATEGY_COLOR.rerank;
const UPSTREAM_LABEL: Record<string, string> = { sparse: "BM25", dense: "Dense", hybrid: "Hybrid" };
const truncate = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

/** Rank movement of one reranked request: what the cross-encoder changed and by how much. */
export function RerankAnalysis({ response }: { response: RetrievalResponse }) {
  const prov = response.provenance.reranking;
  const pool = response.reranking?.candidates ?? [];
  const [focus, setFocus] = useState<string | null>(null);
  if (!prov) return null;
  const k = prov.final_top_k;
  const s = summarize(pool, k);
  const upstream = UPSTREAM_LABEL[response.provenance.strategy] ?? response.provenance.strategy;
  return (
    <Panel>
      <PanelHeader
        eyebrow={`Reranking · ${upstream} → cross-encoder`}
        title="Rank movement"
        actions={
          <Badge tone="neutral" title={`${prov.info.spec.model}@${prov.info.spec.revision}`}>
            <span className="size-1.5 rounded-full" style={{ background: RERANK }} />
            {prov.info.spec.model.split("/").pop()}@{shortHash(prov.info.spec.revision, 7)}
          </Badge>
        }
      />
      <div className="grid gap-px bg-line sm:grid-cols-3 lg:grid-cols-6">
        <Metric label="scored" value={`${s.pool}`} sub={`of ${prov.candidate_k} requested`} />
        <Metric label="kept in top-k" value={`${s.kept}/${Math.min(k, s.pool)}`} sub={`k = ${k}`} />
        <Metric label="entered" value={`+${s.entered}`} color="var(--color-ok)" sub="from below k" />
        <Metric label="left" value={`−${s.left}`} color="var(--color-err)" sub="fell below k" />
        <Metric label="mean |Δ|" value={s.displacement.toFixed(1)} sub="ranks, final top-k" />
        <Metric label="scoring" value={formatMs(prov.latency_ms)} color={RERANK} sub={`${s.pool} pairs`} />
      </div>
      <div className="grid gap-px border-t border-line bg-line lg:grid-cols-[1.2fr_1fr]">
        <div className="bg-surface">
          <SlopeChart pool={pool} k={k} upstream={upstream} focus={focus} onFocus={setFocus} />
        </div>
        <div className="bg-surface">
          <ScoreScatter pool={pool} k={k} upstream={upstream} focus={focus} onFocus={setFocus} />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line px-4 py-2 font-mono text-[10px] text-fg-subtle">
        <Legend color={MOVEMENT_COLOR.promoted} label={`promoted ${s.promoted}`} />
        <Legend color={MOVEMENT_COLOR.demoted} label={`demoted ${s.demoted}`} />
        <Legend color={MOVEMENT_COLOR.unchanged} label={`unchanged ${s.unchanged}`} />
        <span className="ml-auto">counts over the final top-k · Δ = upstream rank − final rank</span>
      </div>
      {s.left > 0 && <LeftTopK pool={pool} focus={focus} onFocus={setFocus} />}
    </Panel>
  );
}

function Metric({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-surface px-4 py-3">
      <div className="num text-lg text-fg" style={color ? { color } : undefined}>{value}</div>
      <div className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
        {label}
        {sub && <span className="block normal-case tracking-normal">{sub}</span>}
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-0.5 w-3 rounded" style={{ background: color }} />
      {label}
    </span>
  );
}

type FocusProps = { focus: string | null; onFocus: (id: string | null) => void };

/** Upstream rank (left) to final rank (right). The dashed line is the top-k cutoff on both axes. */
function SlopeChart({ pool, k, upstream, focus, onFocus }: { pool: RerankCandidate[]; k: number; upstream: string } & FocusProps) {
  const rows = visible(pool, k);
  const depth = Math.max(k, ...rows.map((c) => Math.max(c.rerank.original_rank, c.rerank.final_rank)), 1);
  const row = depth > 30 ? 12 : 22;
  const top = 30;
  const h = top + depth * row + 8;
  const x0 = 150;
  const x1 = 330;
  const y = (rank: number) => top + (rank - 0.5) * row;
  const cut = top + k * row;
  return (
    <div className="p-2">
      <svg viewBox={`0 0 480 ${h}`} className="mx-auto w-full max-w-[620px]" role="img" aria-label={`Rank movement from ${upstream} to the reranked order`}>
        <text x={x0} y={14} textAnchor="middle" fontSize={10} fill={STRATEGY_COLOR[upstream.toLowerCase()] ?? "var(--color-fg-muted)"} fontFamily="var(--font-mono)">
          {upstream} rank
        </text>
        <text x={x1} y={14} textAnchor="middle" fontSize={10} fill={RERANK} fontFamily="var(--font-mono)">
          reranked
        </text>
        <line x1={x0} x2={x0} y1={top - 6} y2={h - 4} stroke="var(--color-line)" />
        <line x1={x1} x2={x1} y1={top - 6} y2={h - 4} stroke="var(--color-line)" />
        {depth > k && (
          <g>
            <line x1={x0 - 20} x2={x1 + 20} y1={cut} y2={cut} stroke="var(--color-line-strong)" strokeDasharray="3 3" />
            <text x={x1 + 24} y={cut + 3} fontSize={9} fill="var(--color-fg-subtle)" fontFamily="var(--font-mono)">top-{k}</text>
          </g>
        )}
        {rows.map((c, n) => {
          const d = c.rerank;
          const color = MOVEMENT_COLOR[d.movement];
          const dim = focus !== null && focus !== c.chunk_id;
          return (
            <g key={c.chunk_id} opacity={dim ? 0.15 : 1} onMouseEnter={() => onFocus(c.chunk_id)} onMouseLeave={() => onFocus(null)} className="transition-opacity duration-200">
              <title>{`${candidateLabel(c)} · ${upstream} #${d.original_rank} → #${d.final_rank} (Δ ${d.rank_delta > 0 ? "+" : ""}${d.rank_delta})`}</title>
              <motion.path
                d={`M${x0},${y(d.original_rank)} L${x1},${y(d.final_rank)}`}
                fill="none"
                stroke={color}
                strokeWidth={focus === c.chunk_id ? 2.2 : 1.2}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.5, delay: Math.min(n, 14) * 0.025, ease: "easeOut" }}
              />
              <circle cx={x0} cy={y(d.original_rank)} r={3.5} fill="var(--color-surface)" stroke={color} strokeWidth={1.4} />
              <circle cx={x1} cy={y(d.final_rank)} r={3.5} fill={d.final_rank <= k ? color : "var(--color-surface)"} stroke={color} strokeWidth={1.4} />
              {row >= 20 && (
                <>
                  <text x={x0 - 10} y={y(d.original_rank) + 3.5} textAnchor="end" fontSize={10} fill="var(--color-fg-muted)" fontFamily="var(--font-mono)">
                    {truncate(candidateLabel(c), 20)}
                  </text>
                  <text x={x1 + 10} y={y(d.final_rank) + 3.5} fontSize={10} fill={color} fontFamily="var(--font-mono)">
                    #{d.final_rank} {d.rank_delta === 0 ? "=" : d.rank_delta > 0 ? `▲${d.rank_delta}` : `▼${-d.rank_delta}`}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Each candidate's upstream score against its reranker score. Final top-k are filled. */
function ScoreScatter({ pool, k, upstream, focus, onFocus }: { pool: RerankCandidate[]; k: number; upstream: string } & FocusProps) {
  if (!pool.length) return null;
  const xs = pool.map((c) => c.rerank.original_score);
  const ys = pool.map((c) => c.rerank.reranker_score);
  const [xl, xh] = [Math.min(...xs), Math.max(...xs)];
  const [yl, yh] = [Math.min(...ys), Math.max(...ys)];
  const W = 360;
  const H = 240;
  const pad = { l: 44, r: 12, t: 14, b: 32 };
  const sx = (v: number) => pad.l + (xh === xl ? 0.5 : (v - xl) / (xh - xl)) * (W - pad.l - pad.r);
  const sy = (v: number) => H - pad.b - (yh === yl ? 0.5 : (v - yl) / (yh - yl)) * (H - pad.t - pad.b);
  // the reranker score of the k-th result: everything above the line made the cut
  const kth = [...pool].sort((a, b) => a.rerank.final_rank - b.rerank.final_rank)[Math.min(k, pool.length) - 1]?.rerank.reranker_score;
  const axis = { fontSize: 9, fill: "var(--color-fg-subtle)", fontFamily: "var(--font-mono)" } as const;
  return (
    <div className="p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto w-full max-w-[460px]" role="img" aria-label={`${upstream} score against reranker score for ${pool.length} candidates`}>
        <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} stroke="var(--color-line-strong)" />
        <line x1={pad.l} x2={pad.l} y1={pad.t} y2={H - pad.b} stroke="var(--color-line-strong)" />
        <text x={pad.l + 2} y={H - pad.b + 12} {...axis}>{xl.toFixed(2)}</text>
        <text x={W - pad.r} y={H - pad.b + 12} textAnchor="end" {...axis}>{xh.toFixed(2)}</text>
        <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" {...axis}>{upstream} score →</text>
        <text x={pad.l - 4} y={sy(yh) + 3} textAnchor="end" {...axis}>{yh.toFixed(1)}</text>
        <text x={pad.l - 4} y={sy(yl) - 2} textAnchor="end" {...axis}>{yl.toFixed(1)}</text>
        <text transform={`translate(10 ${(pad.t + H - pad.b) / 2}) rotate(-90)`} textAnchor="middle" {...axis} fill={RERANK}>reranker score →</text>
        {kth !== undefined && pool.length > k && (
          <g>
            <line x1={pad.l} x2={W - pad.r} y1={sy(kth)} y2={sy(kth)} stroke="var(--color-line-strong)" strokeDasharray="3 3" />
            <text x={W - pad.r} y={sy(kth) - 3} textAnchor="end" {...axis}>top-{k} cut</text>
          </g>
        )}
        {pool.map((c) => {
          const d = c.rerank;
          const color = MOVEMENT_COLOR[d.movement];
          const inTop = d.final_rank <= k;
          return (
            <circle
              key={c.chunk_id}
              cx={sx(d.original_score)}
              cy={sy(d.reranker_score)}
              r={focus === c.chunk_id ? 6 : 4}
              fill={inTop ? color : "var(--color-surface)"}
              stroke={color}
              strokeWidth={1.4}
              opacity={focus !== null && focus !== c.chunk_id ? 0.2 : 0.95}
              onMouseEnter={() => onFocus(c.chunk_id)}
              onMouseLeave={() => onFocus(null)}
              className="transition-[opacity,r] duration-150"
            >
              <title>{`${candidateLabel(c)} · ${upstream} ${d.original_score.toFixed(4)} (#${d.original_rank}) · reranker ${d.reranker_score.toFixed(4)} (#${d.final_rank})`}</title>
            </circle>
          );
        })}
      </svg>
      <p className="px-2 pb-1 font-mono text-[10px] leading-relaxed text-fg-subtle">
        Points on a rising diagonal mean the two stages agree. Scales differ and are not comparable in value, only in order.
      </p>
    </div>
  );
}

function LeftTopK({ pool, focus, onFocus }: { pool: RerankCandidate[] } & FocusProps) {
  const left = pool.filter((c) => c.rerank.left_top_k).sort((a, b) => a.rerank.original_rank - b.rerank.original_rank);
  return (
    <div className="border-t border-line">
      <div className="px-4 pt-3 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">Left the top-k · not in the results below</div>
      <ul className="px-4 py-2">
        {left.map((c) => (
          <li
            key={c.chunk_id}
            onMouseEnter={() => onFocus(c.chunk_id)}
            onMouseLeave={() => onFocus(null)}
            className={cn("flex items-center gap-3 rounded px-1 py-1 font-mono text-[11px]", focus === c.chunk_id && "bg-surface-3")}
          >
            <span className="min-w-0 flex-1 truncate text-fg-muted">{candidateLabel(c)}</span>
            <span className="num text-fg-subtle">#{c.rerank.original_rank} → #{c.rerank.final_rank}</span>
            <span className="num" style={{ color: RERANK }}>{c.rerank.reranker_score.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Movement chip for an evidence card header. */
export function MovementBadge({ delta, entered }: { delta: number; entered: boolean }) {
  const Icon = delta > 0 ? ArrowUp : delta < 0 ? ArrowDown : Minus;
  const tone = delta > 0 ? "ok" : delta < 0 ? "err" : "neutral";
  return (
    <span className="inline-flex items-center gap-1">
      <Badge tone={tone} title="Δ = upstream rank − final rank">
        <Icon className="size-3" />
        {delta === 0 ? "unchanged" : `${delta > 0 ? "promoted" : "demoted"} ${Math.abs(delta)}`}
      </Badge>
      {entered && <Badge tone="ok" title="Ranked below the top-k upstream; the reranker moved it in">entered top-k</Badge>}
    </span>
  );
}

/**
 * Why one result moved, stated only in measured quantities: its two ranks and scores, and the
 * candidates it swapped places with (with their scores). No natural-language rationale is inferred.
 */
export function MovementExplain({ pool, chunkId, upstream }: { pool: RerankCandidate[]; chunkId: string; upstream: string }) {
  const me = pool.find((c) => c.chunk_id === chunkId);
  if (!me) return null;
  const d = me.rerank;
  const { overtook, overtakenBy } = crossings(pool, chunkId);
  return (
    <div className="space-y-3 font-mono text-[11px]" aria-label={`Rank movement for result ${d.final_rank}`}>
      <div className="grid gap-2 sm:grid-cols-2">
        <Stage label={`${upstream} (upstream)`} color={STRATEGY_COLOR[upstream.toLowerCase()]} rank={d.original_rank} score={d.original_score} of={pool.length} />
        <Stage label="cross-encoder (final)" color={RERANK} rank={d.final_rank} score={d.reranker_score} of={pool.length} />
      </div>
      <p className="text-fg-muted">
        Δ <span className="num text-fg">{d.rank_delta > 0 ? "+" : ""}{d.rank_delta}</span> ={" "}
        <span className="num text-ok">{overtook.length}</span> overtaken −{" "}
        <span className="num text-err">{overtakenBy.length}</span> overtook it
      </p>
      {overtook.length > 0 && <Swaps title="Overtook (ranked higher upstream, scored lower by the reranker)" rows={overtook} mine={d.reranker_score} />}
      {overtakenBy.length > 0 && <Swaps title="Overtaken by (ranked lower upstream, scored higher by the reranker)" rows={overtakenBy} mine={d.reranker_score} />}
      {!overtook.length && !overtakenBy.length && (
        <p className="text-fg-subtle">No candidate swapped places with this one: both stages agree on its position.</p>
      )}
    </div>
  );
}

function Stage({ label, color, rank, score, of }: { label: string; color?: string; rank: number; score: number; of: number }) {
  return (
    <div className="rounded-md border border-line bg-surface-2/60 px-3 py-2">
      <div className="flex items-center gap-1.5" style={{ color }}>
        <span className="size-2 rounded-sm" style={{ background: color }} />
        {label}
      </div>
      <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-fg-subtle">
        <dt>rank</dt>
        <dd className="num text-right text-fg">#{rank} <span className="text-fg-subtle">of {of}</span></dd>
        <dt>score</dt>
        <dd className="num text-right text-fg">{score.toFixed(4)}</dd>
      </dl>
    </div>
  );
}

function Swaps({ title, rows, mine }: { title: string; rows: RerankCandidate[]; mine: number }) {
  const shown = rows.slice(0, 6);
  return (
    <div>
      <div className="mb-1 text-[10px] text-fg-subtle">{title}</div>
      <ul className="space-y-0.5">
        {shown.map((c) => (
          <li key={c.chunk_id} className="flex items-center gap-3">
            <span className="min-w-0 flex-1 truncate text-fg-muted">{candidateLabel(c)}</span>
            <span className="num text-fg-subtle">#{c.rerank.original_rank}→#{c.rerank.final_rank}</span>
            <span className="num w-28 text-right" style={{ color: RERANK }}>
              {c.rerank.reranker_score.toFixed(3)} <span className="text-fg-subtle">({(mine - c.rerank.reranker_score >= 0 ? "+" : "") + (mine - c.rerank.reranker_score).toFixed(3)})</span>
            </span>
          </li>
        ))}
      </ul>
      {rows.length > shown.length && <div className="mt-0.5 text-[10px] text-fg-subtle">+{rows.length - shown.length} more</div>}
    </div>
  );
}
