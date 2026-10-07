import type { HealthStatus } from "@rag-forge/shared";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { HealthPanel, formatUptime } from "./health-panel";

const health: HealthStatus = {
  status: "ok",
  version: "1.0.0",
  started_at: "2026-09-30T00:00:00Z",
  uptime_seconds: 125,
  components: [
    { name: "api", state: "ok", detail: "rag-forge 1.0.0" },
    { name: "vector_index", state: "not_configured", detail: "no index backend" },
  ],
};

afterEach(() => vi.unstubAllGlobals());

describe("HealthPanel", () => {
  it("renders component states from the typed health endpoint", async () => {
    const fetchMock = vi.fn(async () => Response.json(health));
    vi.stubGlobal("fetch", fetchMock);
    render(<HealthPanel />);
    expect(await screen.findByText("vector_index")).toBeInTheDocument();
    expect(screen.getByText("not configured")).toBeInTheDocument();
    expect(screen.getByText("uptime 2m 5s")).toBeInTheDocument();
    const req = (fetchMock.mock.calls[0] as unknown as [Request])[0];
    expect(req.url).toBe("http://localhost:8000/api/v1/health");
  });

  it("shows an actionable error when the API is down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("fetch failed"); }));
    render(<HealthPanel />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/unreachable at http:\/\/localhost:8000/);
  });
});

describe("formatUptime", () => {
  it("formats across units", () => {
    expect(formatUptime(9.7)).toBe("9s");
    expect(formatUptime(3725)).toBe("1h 2m");
  });
});
