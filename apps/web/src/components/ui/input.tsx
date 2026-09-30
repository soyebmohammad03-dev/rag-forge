import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/cn";

const control =
  "w-full rounded-md border border-line-strong bg-surface px-3 text-[13px] text-fg placeholder:text-fg-subtle transition-colors hover:border-[#3a434f] focus:border-trace focus:outline-none aria-invalid:border-err";

export function Input({ className, ...props }: ComponentProps<"input">) {
  return <input className={cn(control, "h-9", className)} {...props} />;
}

export function Textarea({ className, ...props }: ComponentProps<"textarea">) {
  return <textarea className={cn(control, "min-h-16 resize-y py-2", className)} {...props} />;
}

export function FormField({
  label,
  hint,
  error,
  htmlFor,
  children,
}: {
  label: string;
  hint?: ReactNode;
  error?: string | null;
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted">
        {label}
      </label>
      {children}
      {error ? (
        <p role="alert" className="text-[11px] text-err">
          {error}
        </p>
      ) : (
        hint && <p className="text-[11px] text-fg-subtle">{hint}</p>
      )}
    </div>
  );
}

/** Radio group styled as a segmented control; arrow keys work natively. */
export function Segmented<T extends string>({
  name,
  value,
  options,
  onChange,
}: {
  name: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div role="radiogroup" className="inline-flex rounded-md border border-line-strong bg-surface p-0.5">
      {options.map((o) => (
        <label
          key={o.value}
          className={cn(
            "cursor-pointer rounded px-3 py-1 font-mono text-[11px] transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-trace",
            value === o.value ? "bg-surface-3 text-fg" : "text-fg-subtle hover:text-fg-muted",
          )}
        >
          <input
            type="radio"
            name={name}
            value={o.value}
            checked={value === o.value}
            onChange={() => onChange(o.value)}
            className="sr-only"
          />
          {o.label}
        </label>
      ))}
    </div>
  );
}
