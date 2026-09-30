const bytesFmt = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
export const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

export function formatBytes(n: number) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${bytesFmt.format(n)} ${units[i]}`;
}

export const shortHash = (h: string, n = 10) => h.slice(0, n);

const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
const steps: [Intl.RelativeTimeFormatUnit, number][] = [
  ["second", 60],
  ["minute", 60],
  ["hour", 24],
  ["day", 30],
  ["month", 12],
  ["year", Infinity],
];

export function timeAgo(iso: string, now = Date.now()) {
  let v = (new Date(iso).getTime() - now) / 1000;
  for (const [unit, size] of steps) {
    if (Math.abs(v) < size) return rtf.format(Math.round(v), unit);
    v /= size;
  }
  return iso;
}
