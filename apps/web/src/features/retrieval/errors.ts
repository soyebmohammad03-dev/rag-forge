/** Turn FastAPI error bodies (detail as string, list, or NotImplementedDetail) into one line. */
export function errorMessage(error: unknown, status: number): string {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg).join("; ");
  if (detail && typeof detail === "object" && "message" in detail) return String(detail.message);
  return `API responded ${status}`;
}
