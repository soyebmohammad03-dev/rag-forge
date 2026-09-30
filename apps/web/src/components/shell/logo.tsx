/** Mark: several candidate lists converging into one ranked answer. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden>
      <rect x="0.5" y="0.5" width="23" height="23" rx="6" fill="var(--color-signal-dim)" stroke="var(--color-signal)" strokeOpacity="0.45" />
      <path d="M5 7 C11 7 12 12 17 12 M5 12 H17 M5 17 C11 17 12 12 17 12" fill="none" stroke="var(--color-signal)" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="18.5" cy="12" r="1.8" fill="var(--color-signal)" />
    </svg>
  );
}
