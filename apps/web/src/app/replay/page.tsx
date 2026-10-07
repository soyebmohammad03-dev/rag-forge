import type { Metadata } from "next";
import { Suspense } from "react";
import { LoadingState } from "@/components/ui/states";
import { ReplayPage } from "@/features/replay/replay-page";

export const metadata: Metadata = { title: "Replay" };

export default function Page() {
  return (
    <Suspense fallback={<LoadingState rows={5} label="Loading" />}>
      <ReplayPage />
    </Suspense>
  );
}
