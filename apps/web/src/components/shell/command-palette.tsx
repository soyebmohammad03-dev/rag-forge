"use client";

import { Command } from "cmdk";
import { BookOpen, CornerDownLeft, Keyboard } from "lucide-react";
import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { AREAS, AREA_GROUPS, areaHref } from "@/lib/areas";
import { API_URL } from "@/lib/api";

const itemClass =
  "flex h-9 cursor-pointer items-center gap-3 rounded-md px-3 text-[13px] text-fg-muted data-[selected=true]:bg-surface-3 data-[selected=true]:text-fg";
const groupClass =
  "[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:pb-1 [&_[cmdk-group-heading]]:pt-3 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.16em] [&_[cmdk-group-heading]]:text-fg-subtle";

export function CommandPalette({
  open,
  onOpenChange,
  onShowShortcuts,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onShowShortcuts: () => void;
}) {
  const router = useRouter();
  const run = (fn: () => void) => {
    onOpenChange(false);
    fn();
  };

  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Command palette"
      overlayClassName="fixed inset-0 z-50 bg-[rgb(4_5_7/0.7)] backdrop-blur-sm"
      contentClassName="fixed left-1/2 top-[14vh] z-50 w-[min(600px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-xl border border-line-strong bg-surface-2 shadow-2xl animate-[dialog-in_160ms_ease-out]"
    >
      <Command.Input
        placeholder="Jump to an area or run a command…"
        className="h-12 w-full border-b border-line bg-transparent px-4 text-sm text-fg outline-none placeholder:text-fg-subtle"
      />
      <Command.List className="max-h-[min(420px,60vh)] overflow-y-auto p-2">
        <Command.Empty className="px-3 py-8 text-center text-xs text-fg-subtle">No matching areas or commands.</Command.Empty>
        {AREA_GROUPS.map((group) => (
          <Command.Group key={group} heading={group} className={groupClass}>
            {AREAS.filter((a) => a.group === group).map((a) => (
              <Item
                key={a.slug}
                value={a.label}
                keywords={[a.group]}
                icon={<a.icon className="size-4" />}
                hint={a.status === "planned" ? "planned" : undefined}
                onSelect={() => run(() => router.push(areaHref(a)))}
              >
                {a.label}
              </Item>
            ))}
          </Command.Group>
        ))}
        <Command.Group heading="Help" className={groupClass}>
          <Item value="Open API reference"
            keywords={["openapi", "docs"]} icon={<BookOpen className="size-4" />} onSelect={() => run(() => window.open(`${API_URL}/docs`, "_blank", "noopener"))}>
            Open API reference
          </Item>
          <Item value="Keyboard shortcuts"
            keywords={["help", "keys"]} icon={<Keyboard className="size-4" />} onSelect={() => run(onShowShortcuts)}>
            Keyboard shortcuts
          </Item>
        </Command.Group>
      </Command.List>
      <div className="flex items-center justify-end gap-1.5 border-t border-line px-3 py-2 font-mono text-[10px] text-fg-subtle">
        <CornerDownLeft className="size-3" /> select · esc close
      </div>
    </Command.Dialog>
  );
}

function Item({
  value,
  keywords,
  icon,
  hint,
  onSelect,
  children,
}: {
  value: string;
  keywords?: string[];
  icon: ReactNode;
  hint?: string;
  onSelect: () => void;
  children: ReactNode;
}) {
  return (
    <Command.Item value={value} keywords={keywords} onSelect={onSelect} className={itemClass}>
      <span className="text-fg-subtle">{icon}</span>
      <span className="flex-1">{children}</span>
      {hint && <span className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">{hint}</span>}
    </Command.Item>
  );
}
