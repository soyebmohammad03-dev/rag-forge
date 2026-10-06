"use client";

import type { QueryAnalysis, QuerySignal, RetrievalResponse, RouteOption, RouterDecision, RuleEvaluation } from "@rag-forge/shared";
import { Check, ChevronDown, CircleSlash, Play, X } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { STRATEGY_COLOR } from "@/components/ui/chart";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { cn } from "@/lib/cn";
import { shortHash } from "@/lib/format";
import { RUN_COLOR, RUN_LABEL } from "./compare";
import { formatMs } from "./rerank";
import { OPTIONS, OPTION_RUN, formatInput, latencyBreakdown, topKOverlap } from "./routing";

const optionLabel = (o: RouteOption) => RUN_LABEL[OPTION_RUN[o]];
const optionColor = (o: RouteOption) => RUN_COLOR[OPTION_RUN[o]];
const SIGNAL_COLOR: Record<string, string> = {
  lexical: "var(--color-s-sparse)",
  semantic: "var(--color-s-dense)",
  complexity: "var(--color-signal)",
};

function Section({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("border-t border-line px-4 py-3", className)}>
      <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">{title}</div>
      {children}
    </div>
  );
}

function Chips({ items, empty = "none", tone }: { items: string[]; empty?: string; tone?: "trace" | "signal" }) {
  if (!items.length) return <span className="font-mono text-[11px] text-fg-subtle">{empty}</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {items.map((t) => (
        <Badge key={t} tone={tone} className="normal-case tracking-normal">{t}</Badge>
      ))}
    </span>
  );
}

// --- query intelligence -------------------------------------------------------------------

/** Everything the analyzer measured, with each score broken into its weighted features. */
export function QueryIntelligencePanel({ analysis }: { analysis: QueryAnalysis }) {
  const { features: f, labels: l, corpus } = analysis;
  const lexical = analysis.signals.find((s) => s.name === "lexical")!;
  const semantic = analysis.signals.find((s) => s.name === "semantic")!;
  return (
    <Panel>
      <PanelHeader
        eyebrow={`Query intelligence · ${analysis.analyzer_version}`}
        title="What the analyzer measured"
        actions={<Badge title={`analysis ${analysis.analysis_hash}`}>{shortHash(analysis.analysis_hash, 8)}</Badge>}
      />
      <div className="grid gap-px border-t border-line bg-line sm:grid-cols-3">
        <Label label="class" value={l.query_class} sub={`margin ${l.class_margin >= 0 ? "+" : ""}${l.class_margin.toFixed(2)}`} color={l.query_class === "lexical" ? SIGNAL_COLOR.lexical : l.query_class === "semantic" ? SIGNAL_COLOR.semantic : "var(--color-s-hybrid)"} />
        <Label label="complexity" value={l.complexity} sub={`score ${analysis.signals.find((s) => s.name === "complexity")!.score.toFixed(2)}`} color={SIGNAL_COLOR.complexity} />
        <Label label="question" value={f.question_type} sub={f.question_word ? `“${f.question_word}”` : f.is_question ? "ends with ?" : "keyword form"} />
      </div>
      <div className="flex flex-wrap gap-1.5 border-t border-line px-4 py-2.5">
        <Flag on={l.multi_hop_likely} label="likely multi-hop" />
        <Flag on={l.ambiguous} label="ambiguous" />
        <Badge tone="neutral">{l.evidence_need.replace("_", " ")}</Badge>
      </div>
      <ClassBalance lexical={lexical.score} semantic={semantic.score} />
      <Section title="Signals · score = Σ value × weight">
        <div className="space-y-2">
          {analysis.signals.map((s) => <SignalRow key={s.name} s={s} />)}
        </div>
      </Section>
      <Section title="Features">
        <dl className="grid grid-cols-[110px_1fr] gap-x-3 gap-y-1.5 text-[11px]">
          <dt className="font-mono text-fg-subtle">tokens</dt>
          <dd className="num font-mono text-fg">{f.token_count} words · {f.char_count} chars · {Math.round(f.function_word_ratio * 100)}% function words</dd>
          <dt className="font-mono text-fg-subtle">key terms</dt>
          <dd><Chips items={f.key_terms} /></dd>
          <dt className="font-mono text-fg-subtle">entities</dt>
          <dd><Chips items={f.entities} tone="signal" /></dd>
          <dt className="font-mono text-fg-subtle">concepts</dt>
          <dd><Chips items={f.concept_segments} /></dd>
          {[
            ["comparison", f.comparison_markers],
            ["multi-hop", f.multi_hop_markers],
            ["temporal", f.temporal_markers],
            ["negation", f.negation_markers],
            ["vague", f.ambiguity_markers],
          ].map(([k, v]) => (
            <Marker key={k as string} label={k as string} items={v as string[]} />
          ))}
        </dl>
      </Section>
      {corpus && (
        <Section title={`Corpus v${corpus.corpus_version} · ${corpus.chunk_count} chunks`}>
          <div className="mb-2 flex items-center gap-3 font-mono text-[11px]">
            <span className="text-fg-muted">coverage</span>
            <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-surface-3">
              <motion.span className="absolute inset-y-0 left-0 rounded-full bg-ok" initial={{ width: 0 }} animate={{ width: `${corpus.coverage * 100}%` }} transition={{ duration: 0.4 }} />
            </span>
            <span className="num text-fg">{Math.round(corpus.coverage * 100)}%</span>
            {corpus.mean_idf !== null && <span className="num text-fg-subtle">mean idf {corpus.mean_idf.toFixed(2)}</span>}
          </div>
          <div className="flex flex-wrap gap-1">
            {corpus.terms.map((t) => (
              <span
                key={t.term}
                title={`df ${t.document_frequency} of ${corpus.chunk_count} · idf ${t.idf.toFixed(3)}`}
                className={cn("rounded px-1.5 py-0.5 font-mono text-[10px]", t.document_frequency ? "bg-surface-3 text-fg-muted" : "bg-err/10 text-err line-through")}
              >
                {t.term} <span className="num text-fg-subtle">{t.document_frequency}</span>
              </span>
            ))}
          </div>
        </Section>
      )}
      <p className="border-t border-line px-4 py-2 text-[10px] leading-relaxed text-fg-subtle">
        Deterministic heuristics over the query text and this corpus version&apos;s term statistics. Not a trained model; scores are not probabilities.
      </p>
    </Panel>
  );
}

function Label({ label, value, sub, color }: { label: string; value: string; sub: string; color?: string }) {
  return (
    <div className="bg-surface px-4 py-3">
      <div className="font-mono text-sm text-fg" style={color ? { color } : undefined}>{value}</div>
      <div className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
        {label} <span className="normal-case tracking-normal">· {sub}</span>
      </div>
    </div>
  );
}

function Flag({ on, label }: { on: boolean; label: string }) {
  return (
    <Badge tone={on ? "signal" : "neutral"} className={cn(!on && "opacity-50")}>
      {on ? <Check className="size-3" /> : <X className="size-3" />} {label}
    </Badge>
  );
}

function Marker({ label, items }: { label: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <>
      <dt className="font-mono text-fg-subtle">{label}</dt>
      <dd><Chips items={items} /></dd>
    </>
  );
}

/** Lexical and semantic scores on one axis; the dashed band is where the class is "mixed". */
function ClassBalance({ lexical, semantic }: { lexical: number; semantic: number }) {
  const margin = lexical - semantic;
  const x = (m: number) => `${((1 - m) / 2) * 100}%`; // +1 (lexical) on the left, −1 on the right
  return (
    <div className="border-t border-line px-4 py-3">
      <div className="mb-1.5 flex justify-between font-mono text-[10px]">
        <span style={{ color: SIGNAL_COLOR.lexical }}>lexical {lexical.toFixed(2)}</span>
        <span className="text-fg-subtle">margin {margin >= 0 ? "+" : ""}{margin.toFixed(2)}</span>
        <span style={{ color: SIGNAL_COLOR.semantic }}>semantic {semantic.toFixed(2)}</span>
      </div>
      <div className="relative h-3 rounded bg-surface-3" role="img" aria-label={`Class margin ${margin.toFixed(2)}`}>
        <span className="hatch absolute inset-y-0 rounded-sm opacity-60" style={{ left: x(0.15), right: `calc(100% - ${x(-0.15)})` }} />
        <motion.span
          className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-surface"
          style={{ background: margin >= 0.15 ? SIGNAL_COLOR.lexical : margin <= -0.15 ? SIGNAL_COLOR.semantic : "var(--color-s-hybrid)" }}
          initial={{ left: "50%" }}
          animate={{ left: x(Math.max(-1, Math.min(1, margin))) }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
      <div className="mt-1 text-center font-mono text-[9px] text-fg-subtle">mixed band ±0.15</div>
    </div>
  );
}

function SignalRow({ s }: { s: QuerySignal }) {
  const [open, setOpen] = useState(false);
  const color = SIGNAL_COLOR[s.name] ?? "var(--color-fg-muted)";
  return (
    <div>
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} className="flex w-full items-center gap-3 font-mono text-[11px]">
        <span className="w-20 text-left" style={{ color }}>{s.name}</span>
        <span className="flex h-2 flex-1 overflow-hidden rounded-full bg-surface-3" aria-hidden>
          {s.contributions.map((c, i) => (
            <motion.span
              key={c.feature}
              className="h-full border-r border-surface last:border-0"
              style={{ background: color, opacity: 1 - i * 0.14 }}
              initial={{ width: 0 }}
              animate={{ width: `${c.contribution * 100}%` }}
              transition={{ duration: 0.4, delay: i * 0.04 }}
            />
          ))}
        </span>
        <span className="num w-10 text-right text-fg">{s.score.toFixed(2)}</span>
        <ChevronDown className={cn("size-3 text-fg-subtle transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <table className="mt-1.5 w-full font-mono text-[10px]">
          <tbody>
            {s.contributions.map((c) => (
              <tr key={c.feature} className={cn(c.contribution === 0 && "text-fg-subtle")}>
                <td className="py-0.5 pl-[92px] text-fg-muted">{c.feature}</td>
                <td className="num text-right">{c.value.toFixed(2)}</td>
                <td className="num text-right text-fg-subtle">× {c.weight}</td>
                <td className="num text-right text-fg">= {c.contribution.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// --- router decision ----------------------------------------------------------------------

/** The selected configuration and the exact rule trace that produced it. */
export function RouterDecisionPanel({ decision }: { decision: RouterDecision }) {
  const constrained = decision.option !== decision.preferred;
  const h = decision.hybrid;
  const r = decision.rerank;
  return (
    <Panel>
      <PanelHeader
        eyebrow={`Router decision · ${decision.policy_version}`}
        title="Selected retrieval configuration"
        actions={<Badge title={`decision ${decision.decision_hash}`}>{shortHash(decision.decision_hash, 8)}</Badge>}
      />
      <div className="flex flex-wrap items-center gap-3 border-t border-line px-4 py-3">
        <span className="flex items-center gap-2 font-mono text-base" style={{ color: optionColor(decision.option) }}>
          <span className="size-2.5 rounded-sm" style={{ background: optionColor(decision.option) }} />
          {optionLabel(decision.option)}
        </span>
        {r.enabled && <Badge style={{ color: STRATEGY_COLOR.rerank }}>+ rerank · pool {r.candidate_k}</Badge>}
        {constrained && (
          <Badge tone="err" title="The policy preferred another option the corpus version cannot serve">
            preferred {optionLabel(decision.preferred)}
          </Badge>
        )}
        <span className="ml-auto font-mono text-[10px] text-fg-subtle" title="Distance of the deciding inputs from the nearest threshold">
          margin {decision.margin === null ? "—" : decision.margin.toFixed(3)}
        </span>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-line px-4 py-2.5 font-mono text-[11px] sm:grid-cols-4">
        <Param k="strategy" v={decision.strategy} />
        <Param k="fusion" v={h ? `${h.fusion}${h.fusion === "rrf" ? ` · k ${h.rrf_k}` : ""}` : "—"} />
        <Param k="weights" v={h?.fusion === "weighted" ? `BM25 ${h.weights?.sparse} · dense ${h.weights?.dense}` : "—"} />
        <Param k="per-component" v={h ? `${h.candidate_k}` : "—"} />
      </dl>
      <Section title="Rationale · fired rules with measured values">
        <ul className="space-y-1 font-mono text-[11px] text-fg-muted">
          {decision.rationale.map((line) => (
            <li key={line} className="flex gap-2">
              <span className="text-signal">›</span>
              <span>{line}</span>
            </li>
          ))}
        </ul>
      </Section>
      <Section title="Strategy rules · evaluated in order, first match decides">
        <RuleTrace rules={decision.rules} />
      </Section>
      <Section title="Rerank rules">
        <RuleTrace rules={decision.rerank_rules} />
      </Section>
      <Section title="Alternatives">
        <table className="w-full font-mono text-[11px]">
          <tbody>
            {decision.alternatives.map((a) => (
              <tr key={a.option} className="border-b border-line/60 last:border-0">
                <td className="py-1.5" style={{ color: optionColor(a.option) }}>{optionLabel(a.option)}</td>
                <td className="py-1.5">
                  {a.selected ? <Badge tone="ok">selected</Badge> : a.available ? <span className="text-fg-subtle">not chosen</span> : <Badge tone="err" title={a.unavailable_reason ?? ""}>unavailable</Badge>}
                </td>
                <td className="py-1.5 text-right text-fg-subtle">{a.rules.join(" · ") || "no rule selects it"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
      <dl className="grid grid-cols-2 gap-x-4 border-t border-line px-4 py-2 font-mono text-[10px] text-fg-subtle">
        <Param k="selected configuration" v={shortHash(decision.configuration_hash, 12)} title={decision.configuration_hash} />
        <Param k="policy config" v={shortHash(decision.policy_config_hash, 12)} title={decision.policy_config_hash} />
      </dl>
    </Panel>
  );
}

function Param({ k, v, title }: { k: string; v: string; title?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[10px] uppercase tracking-wider text-fg-subtle">{k}</dt>
      <dd className="truncate text-fg" title={title}>{v}</dd>
    </div>
  );
}

function RuleTrace({ rules }: { rules: RuleEvaluation[] }) {
  return (
    <ol className="space-y-1.5">
      {rules.map((r) => (
        <li key={r.rule} className={cn("rounded-md border px-2.5 py-1.5", r.matched ? "border-ok/30 bg-ok/5" : "border-line")}>
          <div className="flex items-center gap-2 font-mono text-[11px]">
            {r.matched ? <Check className="size-3.5 text-ok" /> : <CircleSlash className="size-3.5 text-fg-subtle" />}
            <span className={r.matched ? "text-fg" : "text-fg-muted"}>{r.rule}</span>
            {r.outcome && <span className="text-fg-subtle">→ {r.outcome}</span>}
            {r.margin !== null && r.margin !== undefined && <span className="num ml-auto text-[10px] text-fg-subtle">margin {r.margin.toFixed(3)}</span>}
          </div>
          <div className="mt-0.5 pl-5 text-[10px] text-fg-subtle">{r.description}</div>
          <div className="mt-1 flex flex-wrap gap-x-3 pl-5 font-mono text-[10px]">
            {Object.entries(r.inputs).map(([k, v]) => (
              <span key={k}>
                <span className="text-fg-subtle">{k}=</span>
                <span className="num text-fg-muted">{formatInput(v)}</span>
              </span>
            ))}
          </div>
        </li>
      ))}
    </ol>
  );
}

// --- latency, fixed vs adaptive, alternatives ---------------------------------------------

const SEGMENT_COLOR: Record<string, string> = {
  analysis: "var(--color-signal)",
  decision: "var(--color-trace)",
  retrieval: "var(--color-s-hybrid)",
  rerank: "var(--color-s-rerank)",
};

export function LatencyBreakdown({ response }: { response: RetrievalResponse }) {
  const segments = latencyBreakdown(response);
  const total = response.provenance.elapsed_ms || 1;
  return (
    <Panel>
      <PanelHeader eyebrow="Latency" title={`${formatMs(response.provenance.elapsed_ms)} server-side`} />
      <div className="space-y-2 border-t border-line px-4 py-3">
        <div className="flex h-2.5 overflow-hidden rounded-full bg-surface-3" role="img" aria-label="Latency by stage">
          {segments.map((s) => (
            <motion.span key={s.label} className="h-full" style={{ background: SEGMENT_COLOR[s.label] }} initial={{ width: 0 }} animate={{ width: `${(s.ms / total) * 100}%` }} transition={{ duration: 0.4 }} />
          ))}
        </div>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-fg-subtle">
          {segments.map((s) => (
            <li key={s.label} className="flex items-center gap-1.5">
              <span className="size-2 rounded-sm" style={{ background: SEGMENT_COLOR[s.label] }} />
              {s.label} <span className="num text-fg-muted">{formatMs(s.ms)}</span>
            </li>
          ))}
        </ul>
        <p className="text-[10px] text-fg-subtle">One request, measured server-side; retrieval is the remainder of the total. Not a benchmark.</p>
      </div>
    </Panel>
  );
}

/** The adaptive run next to one fixed configuration on the same query. Inspection, not evaluation. */
export function FixedVsAdaptive({ adaptive, fixed, fixedLabel }: { adaptive: RetrievalResponse; fixed: RetrievalResponse; fixedLabel: string }) {
  const o = topKOverlap(adaptive, fixed);
  const decision = adaptive.provenance.routing!.decision;
  const same = decision.configuration_hash === fixed.provenance.configuration_hash;
  const fixedRank = new Map(fixed.hits.map((h) => [h.result.chunk_id, h.result.rank]));
  const adaptiveRank = new Map(adaptive.hits.map((h) => [h.result.chunk_id, h.result.rank]));
  return (
    <Panel>
      <PanelHeader
        eyebrow="Fixed vs adaptive"
        title={`Router (${optionLabel(decision.option)}${decision.rerank.enabled ? " + rerank" : ""}) vs fixed ${fixedLabel}`}
        actions={same ? <Badge tone="ok" title="The router selected exactly this fixed configuration">same configuration</Badge> : undefined}
      />
      <div className="grid gap-px border-t border-line bg-line sm:grid-cols-4">
        <Label label="shared" value={`${o.shared}`} sub={`jaccard ${o.jaccard.toFixed(2)}`} />
        <Label label="adaptive only" value={`${o.aOnly}`} sub="not in fixed top-k" color={optionColor(decision.option)} />
        <Label label="fixed only" value={`${o.bOnly}`} sub="not in adaptive top-k" />
        <Label label="same rank" value={o.shared ? `${o.sameRank}/${o.shared}` : "—"} sub="shared chunks" />
      </div>
      <div className="grid gap-px border-t border-line bg-line md:grid-cols-2">
        {[
          { name: "adaptive", r: adaptive, other: fixedRank },
          { name: `fixed ${fixedLabel}`, r: fixed, other: adaptiveRank },
        ].map(({ name, r, other }) => (
          <ol key={name} className="bg-surface px-4 py-2" aria-label={`${name} ranking`}>
            <li className="mb-1 font-mono text-[10px] uppercase tracking-wider text-fg-subtle">{name} · {formatMs(r.provenance.elapsed_ms)}</li>
            {r.hits.map((h) => {
              const otherRank = other.get(h.result.chunk_id);
              return (
                <li key={h.result.chunk_id} className="flex items-center gap-2 py-0.5 font-mono text-[11px]">
                  <span className="num w-6 text-fg-muted">#{h.result.rank}</span>
                  <span className="min-w-0 flex-1 truncate text-fg">{h.filename}:{h.chunk.ordinal}</span>
                  <span className={cn("num text-[10px]", otherRank ? "text-fg-subtle" : "text-signal")}>{otherRank ? `#${otherRank}` : "only here"}</span>
                </li>
              );
            })}
          </ol>
        ))}
      </div>
      <p className="border-t border-line px-4 py-2 text-[10px] leading-relaxed text-fg-subtle">
        Agreement between rankings is not quality. Whether routing beats fixed pipelines is measured in the Arena against judged queries, not here.
      </p>
    </Panel>
  );
}

export type AlternativeRuns = Partial<Record<RouteOption, RetrievalResponse | string>>;

/** What each fixed option returns for the same query, beside the router's choice. */
export function AlternativesPanel({ adaptive, runs, busy, onRun }: { adaptive: RetrievalResponse; runs: AlternativeRuns; busy: boolean; onRun: () => void }) {
  const decision = adaptive.provenance.routing!.decision;
  const ran = Object.keys(runs).length > 0;
  return (
    <Panel>
      <PanelHeader
        eyebrow="Alternatives"
        title="Every option on this query"
        actions={
          <Button size="sm" onClick={onRun} disabled={busy}>
            <Play className="size-3.5" /> {busy ? "Running…" : ran ? "Run again" : "Run all options"}
          </Button>
        }
      />
      {ran ? (
        <table className="w-full border-t border-line text-left font-mono text-[11px]">
          <thead>
            <tr className="text-[10px] uppercase tracking-wider text-fg-subtle">
              <th className="px-4 py-2 font-normal">option</th>
              <th className="px-2 py-2 text-right font-normal">overlap with router</th>
              <th className="px-2 py-2 font-normal">top result</th>
              <th className="px-4 py-2 text-right font-normal">latency</th>
            </tr>
          </thead>
          <tbody>
            {OPTIONS.map((o) => {
              const r = runs[o];
              const selected = o === decision.option;
              return (
                <tr key={o} className={cn("border-t border-line/60", selected && "bg-surface-3/50")}>
                  <td className="px-4 py-1.5" style={{ color: optionColor(o) }}>
                    {optionLabel(o)} {selected && <Badge tone="ok">router</Badge>}
                  </td>
                  {typeof r === "string" || r === undefined ? (
                    <td colSpan={3} className="px-2 py-1.5 text-fg-subtle">{r ?? "—"}</td>
                  ) : (
                    <>
                      <td className="num px-2 py-1.5 text-right text-fg">
                        {topKOverlap(adaptive, r).shared}/{adaptive.hits.length}
                      </td>
                      <td className="max-w-[220px] truncate px-2 py-1.5 text-fg-muted">{r.hits[0] ? `${r.hits[0].filename}:${r.hits[0].chunk.ordinal}` : "no results"}</td>
                      <td className="num px-4 py-1.5 text-right text-fg-muted">{formatMs(r.provenance.elapsed_ms)}</td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <p className="border-t border-line px-4 py-3 text-[11px] leading-relaxed text-fg-subtle">
          Runs BM25, dense, hybrid RRF and hybrid weighted as fixed configurations, with the router&apos;s rerank setting, so you can see what each alternative would have returned.
        </p>
      )}
    </Panel>
  );
}
