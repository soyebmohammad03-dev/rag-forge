"use client";

import type { ChunkingStrategy } from "@rag-forge/shared";
import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { FormField, Input, Segmented, Textarea } from "@/components/ui/input";
import { api } from "@/lib/api";

const STRATEGIES: { value: ChunkingStrategy; label: string }[] = [
  { value: "recursive", label: "recursive" },
  { value: "fixed", label: "fixed" },
];

export function CreateCorpusDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [strategy, setStrategy] = useState<ChunkingStrategy>("recursive");
  const [size, setSize] = useState(1000);
  const [overlap, setOverlap] = useState(150);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Mirrors the API's validation so mistakes surface before submit; the API remains the authority.
  const sizeError = size < 50 || size > 20000 ? "Between 50 and 20,000 characters" : null;
  const overlapError = overlap < 0 || overlap >= size ? "Must be ≥ 0 and smaller than chunk size" : null;
  const valid = name.trim() && !sizeError && !overlapError;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    setBusy(true);
    setError(null);
    try {
      const { data, response } = await api.POST("/api/v1/corpora", {
        body: {
          name: name.trim(),
          description,
          chunking: { strategy, chunk_size: size, chunk_overlap: overlap },
        },
      });
      if (data) {
        onClose();
        router.push(`/corpus/${data.corpus.id}`);
      } else {
        setError(response.status === 409 ? "A corpus with this name already exists." : `API responded ${response.status}`);
      }
    } catch {
      setError("API unreachable.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} title="New corpus" description="Chunking is fixed per corpus so every version is comparable.">
      <form onSubmit={submit} className="space-y-4" noValidate>
        <FormField label="Name" htmlFor="corpus-name" error={error}>
          <Input id="corpus-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. scifact" autoFocus maxLength={200} required />
        </FormField>
        <FormField label="Description" htmlFor="corpus-desc">
          <Textarea id="corpus-desc" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What this corpus is for" />
        </FormField>
        <fieldset className="space-y-3 rounded-lg border border-line p-3">
          <legend className="px-1 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">Chunking</legend>
          <Segmented name="strategy" value={strategy} options={STRATEGIES} onChange={setStrategy} />
          <p className="text-[11px] leading-relaxed text-fg-subtle">
            {strategy === "recursive"
              ? "Ends chunks at paragraph, line, sentence or word boundaries; overlap starts on a boundary."
              : "Fixed character windows. The naive baseline for chunking ablations."}
          </p>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Chunk size" htmlFor="chunk-size" error={sizeError} hint="characters">
              <Input id="chunk-size" type="number" inputMode="numeric" value={size} min={50} max={20000} aria-invalid={!!sizeError} onChange={(e) => setSize(e.target.valueAsNumber || 0)} />
            </FormField>
            <FormField label="Overlap" htmlFor="chunk-overlap" error={overlapError} hint="characters">
              <Input id="chunk-overlap" type="number" inputMode="numeric" value={overlap} min={0} aria-invalid={!!overlapError} onChange={(e) => setOverlap(e.target.valueAsNumber || 0)} />
            </FormField>
          </div>
        </fieldset>
        <div className="flex justify-end gap-2 pt-1">
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button type="submit" variant="primary" disabled={!valid || busy}>
            {busy ? "Creating…" : "Create corpus"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
