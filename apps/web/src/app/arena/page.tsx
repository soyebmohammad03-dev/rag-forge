import type { Metadata } from "next";
import { Suspense } from "react";
import { LoadingState } from "@/components/ui/states";
import { ArenaPage } from "@/features/arena/arena-page";

export const metadata: Metadata = { title: "Arena" };

export default function Page() {
  return (
    <Suspense fallback={<LoadingState rows={5} label="Loading" />}>
      <ArenaPage initialTab="overview" />
    </Suspense>
  );
}
