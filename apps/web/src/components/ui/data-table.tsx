import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export interface Column<T> {
  key: string;
  header: ReactNode;
  cell: (row: T) => ReactNode;
  align?: "left" | "right";
  className?: string;
}

/** Dense, keyboard-navigable table. Rows are focusable and activate on Enter when clickable. */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  caption,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  caption?: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left text-xs">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-line">
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={cn(
                  "h-8 px-4 font-mono text-[10px] font-normal uppercase tracking-[0.12em] text-fg-subtle",
                  c.align === "right" && "text-right",
                  c.className,
                )}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              tabIndex={onRowClick ? 0 : undefined}
              onClick={onRowClick && (() => onRowClick(row))}
              onKeyDown={onRowClick && ((e) => e.key === "Enter" && onRowClick(row))}
              className={cn(
                "border-b border-line/60 transition-colors last:border-0",
                onRowClick && "cursor-pointer hover:bg-surface-2 focus-visible:bg-surface-2 focus-visible:outline-none",
              )}
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={cn("h-11 px-4 text-fg-muted", c.align === "right" && "text-right", c.className)}
                >
                  {c.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
