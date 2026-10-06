"use client";

import type { Claim, Evidence, EvidenceSupport, GroundingReport, RagResponse } from "@rag-forge/shared";
import { ChevronRight, MessageSquareQuote, ShieldAlert } from "lucide-react";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState } from "@/components/ui/states";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/cn";
import { shortHash } from "@/lib/format";
import { ANSWER_STATUS, FLAG_HELP, GROUNDING_STATUS, SUPPORT, answerSegments, pct } from "./grounding";

export interface Focus {
  claim: string | null;
  evidence: string | null;
}

export function AnswerPanel({
  response,
  focus,
  onFocus,
}: {
  response: RagResponse;
  focus: Focus;
  onFocus: (f: Focus) => void;
}) {
  const { answer, claims, status } = response;
  const s = ANSWER_STATUS[status];
  const g = answer?.generation;
  return (
    <Panel>
      <PanelHeader
        eyebrow="Answer"
        title={g ? `${g.generator.model} · ${g.completion_tokens ?? "?"} tokens · ${g.finish_reason}` : s.label}
        actions={
          <>
            <Badge tone={s.tone} title={s.help}>{s.label}</Badge>
            {answer && <OriginBadge origin="generated" />}
          </>
        }
      />
      {!answer ? (
        <EmptyState icon={MessageSquareQuote} title={s.label}>
          {s.help} {response.warnings.join(" ")}
        </EmptyState>
      ) : (
        <div className="space-y-3 p-4">
          <p className="whitespace-pre-wrap text-[15px] leading-7" aria-label="Generated answer">
            {answerSegments(answer.text, claims).map((seg, i) => {
              if (seg.kind === "text") return <span key={i}>{seg.text}</span>;
              const active = focus.claim === seg.claim.id;
              if (seg.kind === "citation")
                return (
                  <span key={i} className="mx-0.5 inline-flex gap-0.5 align-baseline">
                    {seg.labels.map((l) => (
                      <button
                        key={l.label}
                        type="button"
                        onClick={() => onFocus({ claim: seg.claim.id, evidence: l.evidenceId })}
                        title={l.evidenceId ? `Cited by the model: ${l.evidenceId}` : `${l.label} names no supplied evidence`}
                        className={cn(
                          "rounded px-1 font-mono text-[11px] ring-1 ring-inset",
                          l.evidenceId ? "text-trace ring-trace/40 hover:bg-trace-dim" : "text-err ring-err/50 line-through",
                        )}
                      >
                        {l.label}
                      </button>
                    ))}
                  </span>
                );
              const meta = SUPPORT[seg.claim.support];
              return (
                <span
                  key={i}
                  role="button"
                  tabIndex={0}
                  onClick={() => onFocus({ claim: seg.claim.id, evidence: seg.claim.supporting_evidence_ids[0] ?? null })}
                  onKeyDown={(e) => e.key === "Enter" && onFocus({ claim: seg.claim.id, evidence: seg.claim.supporting_evidence_ids[0] ?? null })}
                  data-support={seg.claim.support}
                  className={cn("cursor-pointer rounded-sm decoration-2 underline-offset-[5px] transition-colors", active && "bg-surface-3")}
                  style={{ textDecorationLine: "underline", textDecorationColor: meta.color, textDecorationStyle: seg.claim.support === "weakly_supported" ? "dashed" : "solid" }}
                >
                  {seg.text}
                </span>
              );
            })}
          </p>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line pt-3 font-mono text-[10px] text-fg-subtle">
            {(["supported", "weakly_supported", "unsupported"] as const).map((k) => (
              <span key={k} className="flex items-center gap-1.5">
                <span className="h-0.5 w-4" style={{ background: SUPPORT[k].color }} /> {SUPPORT[k].label}
              </span>
            ))}
            <span>underline = measured support · <span className="text-trace">E#</span> = citation written by the model</span>
          </div>
          {response.warnings.map((w) => (
            <p key={w} role="status" className="rounded-md border border-signal/30 bg-signal-dim/40 px-3 py-2 text-xs text-signal">{w}</p>
          ))}
        </div>
      )}
    </Panel>
  );
}

function Tile({ label, value, sub, tone, help }: { label: string; value: string; sub?: string; tone?: string; help: string }) {
  return (
    <div className="rounded-md bg-surface-2 p-3 ring-1 ring-line">
      <Tooltip content={help} side="bottom">
        <span tabIndex={0} className="cursor-help font-mono text-[10px] uppercase tracking-[0.12em] text-fg-muted underline decoration-line-strong decoration-dotted underline-offset-4">
          {label}
        </span>
      </Tooltip>
      <div className="num mt-1.5 text-[22px] leading-none" style={{ color: tone }}>{value}</div>
      {sub && <div className="mt-1 font-mono text-[10px] text-fg-subtle">{sub}</div>}
    </div>
  );
}

export function GroundingPanel({ report }: { report: GroundingReport }) {
  const s = GROUNDING_STATUS[report.status];
  const f = report.factual_claims;
  return (
    <Panel>
      <PanelHeader
        eyebrow={`Grounding · ${report.verifier}@${report.verifier_version}`}
        title="Measured support of each claim by the supplied evidence"
        actions={<><Badge tone={s.tone}>{s.label}</Badge><OriginBadge origin="measured" /></>}
      />
      <div className="grid grid-cols-2 gap-2 p-3 sm:grid-cols-4">
        <Tile label="Supported" value={`${report.supported}/${f}`} tone={SUPPORT.supported.color} help="Factual claims meeting both supported thresholds." />
        <Tile label="Weak" value={`${report.weakly_supported}/${f}`} tone={SUPPORT.weakly_supported.color} help="Factual claims meeting only the weak thresholds, or downgraded by a number or negation mismatch." />
        <Tile label="Unsupported" value={`${report.unsupported}/${f}`} tone={SUPPORT.unsupported.color} help="No evidence was measured to support them. This does not mean they are false." />
        <Tile label="Grounding score" value={report.grounding_score === null ? "—" : report.grounding_score.toFixed(2)} sub="(sup + ½ weak) / factual" help="(supported + 0.5 × weakly supported) / factual claims. Null when there are no factual claims." />
        <Tile label="Evidence coverage" value={pct(report.evidence_coverage)} help="Share of the selected evidence that supports at least one claim." />
        <Tile label="Citation coverage" value={pct(report.citation_coverage)} help="Share of factual claims carrying at least one valid citation written by the generator." />
        <Tile label="Citation precision" value={pct(report.citation_precision)} help="Share of the generator's valid citations whose passage the verifier measured as supporting that claim." />
        <Tile label="Invalid citations" value={String(report.invalid_citations)} tone={report.invalid_citations ? "var(--color-err)" : undefined} help="Citations naming a passage id that was not supplied. Kept and counted, never remapped." />
      </div>
      <div className="space-y-1 border-t border-line px-4 py-2.5 text-[11px] leading-relaxed text-fg-subtle">
        <p>
          {report.claims} claim{report.claims === 1 ? "" : "s"}: {f} factual · {report.abstentions} abstention · {report.non_assertive} non-assertive ·
          {" "}thresholds {Object.entries(report.thresholds).map(([k, v]) => `${k} ${v}`).join(" · ")} · {report.latency_ms.toFixed(1)} ms
          {report.load_ms > 0 && ` + ${report.load_ms.toFixed(0)} ms model load`}
        </p>
        <p className="flex items-start gap-1.5">
          <ShieldAlert className="mt-0.5 size-3.5 shrink-0 text-signal" />
          Grounding measures term and meaning overlap with the evidence, not answer quality or truth.
          {!report.detects_contradiction && " This verifier cannot detect contradiction: unsupported means no support was measured, not that the claim is false."}
        </p>
      </div>
    </Panel>
  );
}

export function ClaimList({
  claims,
  evidence,
  focus,
  onFocus,
}: {
  claims: Claim[];
  evidence: Evidence[];
  focus: Focus;
  onFocus: (f: Focus) => void;
}) {
  const byId = new Map(evidence.map((e) => [e.id, e]));
  return (
    <Panel>
      <PanelHeader eyebrow="Claims" title="Claim → evidence relationships" actions={<OriginBadge origin="generated" />} />
      {claims.length === 0 ? (
        <p className="px-4 py-6 text-center text-xs text-fg-subtle">The answer contains no sentences to check.</p>
      ) : (
        <ol className="divide-y divide-line">
          {claims.map((c) => (
            <ClaimRow
              key={c.id}
              claim={c}
              byId={byId}
              open={focus.claim === c.id}
              focusEvidence={focus.evidence}
              onToggle={() => onFocus(focus.claim === c.id ? { claim: null, evidence: null } : { claim: c.id, evidence: c.supporting_evidence_ids[0] ?? null })}
              onEvidence={(id) => onFocus({ claim: c.id, evidence: id })}
            />
          ))}
        </ol>
      )}
    </Panel>
  );
}

function ClaimRow({
  claim,
  byId,
  open,
  focusEvidence,
  onToggle,
  onEvidence,
}: {
  claim: Claim;
  byId: Map<string, Evidence>;
  open: boolean;
  focusEvidence: string | null;
  onToggle: () => void;
  onEvidence: (id: string) => void;
}) {
  const meta = SUPPORT[claim.support];
  const cite = (ids: string[]) => ids.map((id) => byId.get(id)?.citation ?? "?").join(" ");
  return (
    <li data-claim={claim.id}>
      <button type="button" onClick={onToggle} aria-expanded={open} className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-surface-2/60">
        <ChevronRight className={cn("mt-0.5 size-3.5 shrink-0 text-fg-subtle transition-transform", open && "rotate-90")} />
        <span className="num w-5 shrink-0 font-mono text-[11px] text-fg-subtle">{claim.index + 1}</span>
        <span className="min-w-0 flex-1 text-[13px] leading-relaxed">{claim.text}</span>
        <span className="flex shrink-0 flex-col items-end gap-1">
          <Badge tone={meta.tone}>{claim.kind === "factual" ? meta.label : claim.kind.replace("_", "-")}</Badge>
          {claim.support_score !== null && <span className="num font-mono text-[10px] text-fg-subtle">score {claim.support_score.toFixed(3)}</span>}
        </span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-line bg-surface-2/40 px-4 py-3 pl-12">
          <dl className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-1 font-mono text-[11px]">
            <dt className="text-fg-subtle">cited (generated)</dt>
            <dd>{claim.citations.length ? claim.citations.map((c) => <span key={`${c.char_start}${c.label}`} className={cn("mr-1.5", c.valid ? "text-trace" : "text-err line-through")}>{c.label}</span>) : <span className="text-fg-subtle">none</span>}</dd>
            <dt className="text-fg-subtle">supporting (measured)</dt>
            <dd>{claim.supporting_evidence_ids.length ? <span className="text-ok">{cite(claim.supporting_evidence_ids)}</span> : <span className="text-fg-subtle">none</span>}</dd>
            <dt className="text-fg-subtle">rationale</dt>
            <dd className="text-fg-muted">{claim.rationale}</dd>
            {claim.missing_terms.length > 0 && (<><dt className="text-fg-subtle">missing terms</dt><dd className="text-signal">{claim.missing_terms.join(" · ")}</dd></>)}
            {claim.unmatched_numbers.length > 0 && (<><dt className="text-fg-subtle">unmatched numbers</dt><dd className="text-err">{claim.unmatched_numbers.join(" · ")}</dd></>)}
            {claim.flags.length > 0 && (
              <>
                <dt className="text-fg-subtle">flags</dt>
                <dd className="flex flex-wrap gap-1">{claim.flags.map((f) => <Badge key={f} tone="signal" title={FLAG_HELP[f] ?? f}>{f}</Badge>)}</dd>
              </>
            )}
            <dt className="text-fg-subtle">claim id</dt>
            <dd className="text-fg-subtle" title={claim.id}>{shortHash(claim.id, 16)}</dd>
          </dl>
          {claim.evidence.length > 0 && (
            <table className="w-full text-left font-mono text-[11px]">
              <caption className="sr-only">Support of this claim by each evidence passage</caption>
              <thead className="text-[10px] uppercase tracking-wider text-fg-subtle">
                <tr><th className="py-1 font-normal">evidence</th><th className="font-normal">lexical</th><th className="font-normal">semantic</th><th className="font-normal">status</th><th className="font-normal">closest sentence</th></tr>
              </thead>
              <tbody>
                {claim.evidence.map((es) => <SupportRow key={es.evidence_id} es={es} active={focusEvidence === es.evidence_id} onClick={() => onEvidence(es.evidence_id)} />)}
              </tbody>
            </table>
          )}
        </div>
      )}
    </li>
  );
}

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-1.5 w-12 overflow-hidden rounded-full bg-surface-3"><span className="block h-full" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }} /></span>
      <span className="num">{value.toFixed(2)}</span>
    </span>
  );
}

function SupportRow({ es, active, onClick }: { es: EvidenceSupport; active: boolean; onClick: () => void }) {
  const meta = SUPPORT[es.status];
  return (
    <tr onClick={onClick} className={cn("cursor-pointer border-t border-line align-top hover:bg-surface-3/50", active && "bg-surface-3/70")}>
      <td className="py-1.5 pr-2"><span className="text-trace">{es.citation}</span>{es.cited && <span className="ml-1 text-[9px] text-fg-subtle">cited</span>}</td>
      <td className="pr-2"><Bar value={es.lexical_coverage} color="var(--color-s-sparse)" /></td>
      <td className="pr-2"><Bar value={es.semantic_similarity} color="var(--color-s-dense)" /></td>
      <td className="pr-2" style={{ color: meta.color }}>{meta.label}</td>
      <td className="font-sans text-fg-muted">{es.best_sentence}</td>
    </tr>
  );
}
