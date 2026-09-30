"use client";

import type { Chunk } from "@rag-forge/shared";
import { motion } from "motion/react";
import { useState } from "react";
import { cn } from "@/lib/cn";

/** Characters each chunk repeats from the one before it (0 for the first). */
export function overlaps(chunks: Chunk[]): number[] {
  return chunks.map((c, i) => (i ? Math.max(0, chunks[i - 1].char_end - c.char_start) : 0));
}

/**
 * Where chunks fall in the extracted text. Each band is a chunk; brighter where neighbours
 * overlap, gaps are whitespace the chunker trimmed.
 */
export function ChunkMap({
  chunks,
  textChars,
  active,
  onSelect,
}: {
  chunks: Chunk[];
  textChars: number;
  active: number | null;
  onSelect: (ordinal: number) => void;
}) {
  const pct = (n: number) => `${(n / Math.max(textChars, 1)) * 100}%`;
  return (
    <div className="relative h-7 overflow-hidden rounded bg-surface-3" role="img" aria-label={`${chunks.length} chunks across ${textChars} characters`}>
      {chunks.map((c, i) => (
        <button
          key={c.id}
          type="button"
          tabIndex={-1}
          title={`#${c.ordinal} · ${c.char_start}–${c.char_end}`}
          onClick={() => onSelect(c.ordinal)}
          className={cn(
            "absolute inset-y-1 rounded-sm border-x transition-colors",
            i % 2 ? "top-3.5 bg-trace/25 border-trace/60" : "bottom-3.5 bg-signal/25 border-signal/60",
            active === c.ordinal && "bg-fg/60",
          )}
          style={{ left: pct(c.char_start), width: pct(c.char_end - c.char_start) }}
        />
      ))}
    </div>
  );
}

export function ChunkList({ chunks, textChars }: { chunks: Chunk[]; textChars: number }) {
  const [active, setActive] = useState<number | null>(null);
  const over = overlaps(chunks);
  const select = (ordinal: number) => {
    setActive(ordinal);
    document.getElementById(`chunk-${ordinal}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  return (
    <div className="space-y-3">
      <ChunkMap chunks={chunks} textChars={textChars} active={active} onSelect={select} />
      <ol className="space-y-2">
        {chunks.map((c, i) => (
          <motion.li
            key={c.id}
            id={`chunk-${c.ordinal}`}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(i, 10) * 0.02 }}
            onMouseEnter={() => setActive(c.ordinal)}
            className={cn(
              "rounded-md border bg-surface px-3 py-2 transition-colors",
              active === c.ordinal ? "border-line-strong" : "border-line",
            )}
          >
            <div className="mb-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-fg-subtle">
              <span className={cn(i % 2 ? "text-trace" : "text-signal")}>#{c.ordinal}</span>
              <span className="num">{c.char_start}–{c.char_end}</span>
              <span className="num">{c.char_end - c.char_start} chars</span>
              {typeof c.metadata.words === "number" && <span className="num">{c.metadata.words} words</span>}
              {typeof c.metadata.page_start === "number" && (
                <span>
                  p.{c.metadata.page_start}
                  {c.metadata.page_end !== c.metadata.page_start && `–${c.metadata.page_end}`}
                </span>
              )}
              {over[i] > 0 && <span className="text-fg-muted">↩ {over[i]} overlap</span>}
            </div>
            <p className="line-clamp-6 whitespace-pre-wrap break-words text-xs leading-relaxed text-fg-muted">
              {over[i] > 0 && (
                <mark className="hatch rounded-sm bg-transparent text-fg-subtle" title="Repeated from the previous chunk">
                  {c.text.slice(0, over[i])}
                </mark>
              )}
              {c.text.slice(over[i])}
            </p>
          </motion.li>
        ))}
      </ol>
    </div>
  );
}
