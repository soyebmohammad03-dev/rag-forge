import type { Metadata } from "next";
import { Suspense } from "react";
import { LoadingState } from "@/components/ui/states";
import { EvidencePage } from "@/features/evidence/evidence-page";

export const metadata: Metadata = { title: "Evidence Lab" };

export default function Page() {
  // useSearchParams needs a Suspense boundary for static rendering.
  return (
    <Suspense fallback={<LoadingState rows={5} label="Loading" />}>
      <EvidencePage />
    </Suspense>
  );
}
