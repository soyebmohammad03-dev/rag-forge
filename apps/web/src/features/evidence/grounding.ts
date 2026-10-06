import type { AnswerGrounding, AnswerStatus, Claim, SelectionOutcome, SupportStatus } from "@rag-forge/shared";

/** One run of the raw answer: plain text, part of a claim, or a citation marker the model wrote. */
export type Segment =
  | { kind: "text"; text: string }
  | { kind: "claim"; text: string; claim: Claim }
  | { kind: "citation"; text: string; claim: Claim; labels: { label: string; evidenceId: string | null }[] };

/**
 * Split the raw answer into renderable segments using the server's character offsets, so the
 * answer is shown exactly as generated: nothing is rewritten, and markers stay where they were.
 */
export function answerSegments(text: string, claims: Claim[]): Segment[] {
  const out: Segment[] = [];
  let pos = 0;
  for (const claim of [...claims].sort((a, b) => a.char_start - b.char_start)) {
    if (claim.char_start < pos) continue; // overlapping spans would duplicate text; never expected
    if (claim.char_start > pos) out.push({ kind: "text", text: text.slice(pos, claim.char_start) });
    let cur = claim.char_start;
    const markers = [...new Map(claim.citations.map((c) => [c.char_start, c])).values()].sort(
      (a, b) => a.char_start - b.char_start,
    );
    for (const m of markers) {
      if (m.char_start < cur || m.char_end > claim.char_end) continue;
      if (m.char_start > cur) out.push({ kind: "claim", text: text.slice(cur, m.char_start), claim });
      const labels = claim.citations
        .filter((c) => c.char_start === m.char_start)
        .map((c) => ({ label: c.label, evidenceId: c.evidence_id }));
      out.push({ kind: "citation", text: text.slice(m.char_start, m.char_end), claim, labels });
      cur = m.char_end;
    }
    if (claim.char_end > cur) out.push({ kind: "claim", text: text.slice(cur, claim.char_end), claim });
    pos = claim.char_end;
  }
  if (pos < text.length) out.push({ kind: "text", text: text.slice(pos) });
  return out;
}

export const SUPPORT: Record<SupportStatus, { label: string; tone: "ok" | "signal" | "err" | "neutral"; color: string }> = {
  supported: { label: "supported", tone: "ok", color: "var(--color-ok)" },
  weakly_supported: { label: "weak", tone: "signal", color: "var(--color-signal)" },
  unsupported: { label: "unsupported", tone: "err", color: "var(--color-err)" },
  contradicted: { label: "contradicted", tone: "err", color: "var(--color-err)" },
  not_applicable: { label: "n/a", tone: "neutral", color: "var(--color-fg-subtle)" },
};

export const ANSWER_STATUS: Record<AnswerStatus, { label: string; tone: "ok" | "signal" | "trace" | "neutral"; help: string }> = {
  answered: { label: "answered", tone: "trace", help: "The generator wrote an answer; its claims were checked against the evidence." },
  abstained: { label: "abstained", tone: "signal", help: "The generator said the evidence is insufficient." },
  insufficient_evidence: {
    label: "insufficient evidence",
    tone: "signal",
    help: "No passage was usable as evidence, so the generator was not called.",
  },
  not_generated: { label: "context only", tone: "neutral", help: "Stopped after evidence selection and context assembly." },
};

export const GROUNDING_STATUS: Record<AnswerGrounding, { label: string; tone: "ok" | "signal" | "err" | "neutral" }> = {
  grounded: { label: "grounded", tone: "ok" },
  partially_grounded: { label: "partially grounded", tone: "signal" },
  ungrounded: { label: "ungrounded", tone: "err" },
  abstained: { label: "abstained", tone: "neutral" },
  no_claims: { label: "no claims", tone: "neutral" },
};

export const OUTCOME_LABEL: Record<SelectionOutcome, string> = {
  selected: "selected",
  near_duplicate: "near duplicate",
  document_cap: "document cap",
  below_min_score: "below min score",
  over_token_budget: "over token budget",
  over_item_budget: "over item budget",
};

export const FLAG_HELP: Record<string, string> = {
  uncited: "The generator wrote no citation for this claim.",
  invalid_citation: "The claim cites a passage id that was not supplied.",
  cited_not_supporting: "A cited passage was not measured as supporting the claim.",
  number_mismatch: "A number in the claim does not occur in the supporting evidence.",
  negation_mismatch: "Negation differs between the claim and its closest evidence sentence.",
  multi_passage: "Supported only by several passages together.",
};

export const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
