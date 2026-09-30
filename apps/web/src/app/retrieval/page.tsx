import type { Metadata } from "next";
import { Suspense } from "react";
import { LoadingState } from "@/components/ui/states";
import { RetrievalPage } from "@/features/retrieval/retrieval-page";

export const metadata: Metadata = { title: "Retrieval Lab" };

export default function Page() {
  // useSearchParams needs a Suspense boundary for static rendering.
  return (
    <Suspense fallback={<LoadingState rows={5} label="Loading" />}>
      <RetrievalPage />
    </Suspense>
  );
}
