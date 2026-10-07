import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { OriginBadge } from "./badge";
import { Tabs } from "./tabs";

describe("OriginBadge", () => {
  it("labels simulated data so it cannot pass for a measured result", () => {
    render(<OriginBadge origin="simulated" />);
    expect(screen.getByText("sample")).toHaveAttribute("title", expect.stringMatching(/not a measured result/i));
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
