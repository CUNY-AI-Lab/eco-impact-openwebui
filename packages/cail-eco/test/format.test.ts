import { describe, expect, it } from "vitest";
import { comparison, formatRange } from "../src/format.js";

describe("formatRange", () => {
  it("renders a gram range like the Python filter", () => {
    expect(formatRange(0.29894, 0.48697, "g")).toBe("0.30–0.49 g");
  });
  it("scales large gram totals to kg", () => {
    expect(formatRange(1500, 2600, "g")).toBe("1.50–2.60 kg");
  });
  it("picks the unit after rounding at the kg boundary", () => {
    expect(formatRange(999.6, 999.7, "g")).toBe("~1.00 kg");
    expect(formatRange(1000, 1000, "g")).toBe("~1.00 kg");
  });
  it("never prints 100.0 mg at the mg boundary", () => {
    expect(formatRange(0.0999, 0.09996, "g")).toBe("~0.10 g");
    expect(formatRange(0.05, 0.09996, "g")).toBe("0.05–0.10 g");
  });
  it("renders zero in the requested unit", () => {
    expect(formatRange(0, 0, "g")).toBe("0 g");
    expect(formatRange(0, 0, "Wh")).toBe("0 Wh");
  });
  it("scales small energy to mWh", () => {
    expect(formatRange(0.05, 0.08, "Wh")).toBe("50.0–80.0 mWh");
  });
});

describe("comparison", () => {
  it("uses the breaths tier below 1 g", () => {
    expect(comparison(0.4)).toBe("🌬️ ~2.00 breaths");
  });
});
