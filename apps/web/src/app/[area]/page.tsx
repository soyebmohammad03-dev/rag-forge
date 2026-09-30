import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { PlannedArea } from "@/features/planned/planned-area";
import { AREAS, findArea } from "@/lib/areas";

export const dynamicParams = false;

export function generateStaticParams() {
  return AREAS.filter((a) => a.status === "planned").map((a) => ({ area: a.slug }));
}

export async function generateMetadata(props: PageProps<"/[area]">): Promise<Metadata> {
  const { area } = await props.params;
  return { title: findArea(area)?.label };
}

export default async function Page(props: PageProps<"/[area]">) {
  const area = findArea((await props.params).area);
  if (!area || area.status !== "planned") notFound();
  return <PlannedArea slug={area.slug} />;
}
