import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModeSwitcher } from "@/components/ModeSwitcher";

const navigation = vi.hoisted(() => ({
  replace: vi.fn(),
  refresh: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/train",
  useRouter: () => navigation,
}));

afterEach(() => {
  cleanup();
  navigation.replace.mockReset();
  navigation.refresh.mockReset();
});

describe("ModeSwitcher", () => {
  it("replaces navigation with only the validated selected mode", async () => {
    const user = userEvent.setup();
    render(<ModeSwitcher current="precomputed" />);

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Choose a saved sample state" }),
      ["fetch_failure"],
    );

    expect(navigation.replace).toHaveBeenCalledWith("/train?mode=fetch_failure");
    expect(navigation.refresh).toHaveBeenCalledOnce();
  });
});
