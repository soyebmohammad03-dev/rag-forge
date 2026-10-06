import type { Metadata } from "next";
import { Suspense } from "react";
import { LoadingState } from "@/components/ui/states";
import { RetrievalPage } from "@/features/retrieval/retrieval-page";

export const metadata: Metadata = { title: "Router" };

/** The Retrieval Lab, opened in adaptive mode: query intelligence and the router's decision. */
export default function Page() {
  return (
    <Suspense fallback={<LoadingState rows={5} label="Loading" />}>
      <RetrievalPage initialRouting="adaptive" />
    </Suspense>
  );
}
