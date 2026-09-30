import type { Metadata } from "next";
import { CorpusDetailPage } from "@/features/corpus/corpus-detail-page";

export const metadata: Metadata = { title: "Corpus" };

export default async function Page(props: PageProps<"/corpus/[corpusId]">) {
  const { corpusId } = await props.params;
  return <CorpusDetailPage corpusId={corpusId} />;
}
