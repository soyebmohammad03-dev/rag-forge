import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { CommandPalette } from "./command-palette";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

describe("CommandPalette", () => {
  it("filters areas and navigates on Enter", async () => {
    const onOpenChange = vi.fn();
    render(<CommandPalette open onOpenChange={onOpenChange} onShowShortcuts={() => {}} />);
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText(/jump to an area/i), "arena");
    expect(screen.queryByText("Keyboard shortcuts")).not.toBeInTheDocument();
    await user.keyboard("{Enter}");
    expect(push).toHaveBeenCalledWith("/arena");
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
