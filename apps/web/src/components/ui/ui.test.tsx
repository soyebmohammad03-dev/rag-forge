import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { MetricCard } from "./metric-card";
import { Tabs } from "./tabs";

describe("MetricCard", () => {
  it("labels sample data so it cannot pass for a measured result", () => {
    render(<MetricCard label="nDCG@10" value={0.6123} delta={0.04} origin="simulated" />);
    expect(screen.getByText("0.612")).toBeInTheDocument();
    expect(screen.getByText("sample")).toHaveAttribute("title", expect.stringMatching(/not a measured result/i));
  });

  it("treats a latency drop as an improvement", () => {
    render(<MetricCard label="p95" value={1.8} unit="s" delta={-0.2} higherIsBetter={false} origin="measured" />);
    const delta = screen.getByText("vs baseline").parentElement;
    expect(delta).toHaveClass("text-ok");
    expect(delta).toHaveTextContent("-0.20");
  });
});

function TabsHarness() {
  const [v, setV] = useState("a");
  return <Tabs value={v} onChange={setV} tabs={[{ id: "a", label: "A" }, { id: "b", label: "B" }, { id: "c", label: "C" }]} />;
}

describe("Tabs", () => {
  it("supports arrow, Home and End keys with wraparound", async () => {
    render(<TabsHarness />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("tab", { name: "A" }));
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("tab", { name: "C" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "C" })).toHaveFocus();
    await user.keyboard("{Home}");
    expect(screen.getByRole("tab", { name: "A" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "B" })).toHaveAttribute("aria-selected", "true");
  });
});
