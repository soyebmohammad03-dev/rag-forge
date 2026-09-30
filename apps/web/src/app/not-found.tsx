import { Compass } from "lucide-react";
import Link from "next/link";
import { EmptyState } from "@/components/ui/states";

export default function NotFound() {
  return (
    <EmptyState icon={Compass} title="Nothing at this address" className="mt-16"
      action={<Link href="/" className="text-xs text-trace underline-offset-4 hover:underline">Back to overview</Link>}>
      Use <span className="font-mono">⌘K</span> to jump to any area.
    </EmptyState>
  );
}
