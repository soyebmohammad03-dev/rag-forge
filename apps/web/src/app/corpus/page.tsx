import type { Metadata } from "next";
import { CorpusListPage } from "@/features/corpus/corpus-list-page";

export const metadata: Metadata = { title: "Corpus" };

export default function Page() {
  return <CorpusListPage />;
}
