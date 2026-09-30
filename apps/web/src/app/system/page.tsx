import type { Metadata } from "next";
import { SystemPage } from "@/features/system/system-page";

export const metadata: Metadata = { title: "System" };

export default function Page() {
  return <SystemPage />;
}
