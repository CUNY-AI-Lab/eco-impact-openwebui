import { describe, expect, it } from "vitest";
import { addTotals, estimateRows } from "../src/aggregate.js";
import { computeImpacts } from "../src/impacts.js";
import { EMBEDDED_REGISTRY } from "../src/embedded-registry.js";

// SAFETY: the embedded registry has non-skip ids; entries are objects (asserted by isRegistry and the generator).
const knownId = Object.keys(EMBEDDED_REGISTRY.ids).find((id) => !("skip" in (EMBEDDED_REGISTRY.ids[id] as object)))!;

describe("estimateRows", () => {
  it("sums per tool and matches computeImpacts for one row", () => {
    const rows = [{ tool: "sandbox" as const, model: knownId, requests: 3, output_tokens: 1200 }];
    const e = estimateRows(rows, EMBEDDED_REGISTRY);
    // SAFETY: knownId is a non-skip registry entry with total/active ranges; its zone exists in the registry.
    const entry = EMBEDDED_REGISTRY.ids[knownId] as { total: { min: number; max: number }; active: { min: number; max: number }; zone?: string };
    const zone = EMBEDDED_REGISTRY.zones[entry.zone ?? "USA"]!;
    const direct = computeImpacts(entry, 1200, zone, EMBEDDED_REGISTRY.defaults);
    expect(e.totals.gwp_ug.min).toBe(Math.round(direct.gwpG[0] * 1e6));
    expect(e.totals.gwp_ug.max).toBe(Math.round(direct.gwpG[1] * 1e6));
    expect(e.totals.energy_mwh.min).toBe(Math.round(direct.energyWh[0] * 1e3));
    expect(e.tools).toEqual([{ tool: "sandbox", ...e.totals }]);
    expect(e.totals.generic_models).toBe(0);
  });
  it("unknown model uses the generic estimate and is counted", () => {
    const e = estimateRows([{ tool: "api-key", model: "no-such-model", requests: 1, output_tokens: 100 }], EMBEDDED_REGISTRY);
    expect(e.totals.generic_models).toBe(1);
    expect(e.totals.gwp_ug.max).toBeGreaterThan(e.totals.gwp_ug.min);
  });
  it("skip and unresolved models add requests but no impact", () => {
    const e = estimateRows([{ tool: "media-tools", model: "whisper", requests: 4, output_tokens: 0 }], EMBEDDED_REGISTRY);
    expect(e.totals).toEqual({ requests: 4, output_tokens: 0, gwp_ug: { min: 0, max: 0 }, energy_mwh: { min: 0, max: 0 }, generic_models: 0 });
  });
  it("zero tokens estimate zero", () => {
    const e = estimateRows([{ tool: "sandbox", model: knownId, requests: 2, output_tokens: 0 }], EMBEDDED_REGISTRY);
    expect(e.totals.gwp_ug).toEqual({ min: 0, max: 0 });
    expect(e.totals.requests).toBe(2);
  });
  it("orders tools by gwp max descending", () => {
    const e = estimateRows([
      { tool: "api-key", model: knownId, requests: 1, output_tokens: 10 },
      { tool: "sandbox", model: knownId, requests: 1, output_tokens: 1000 },
    ], EMBEDDED_REGISTRY);
    expect(e.tools.map((t) => t.tool)).toEqual(["sandbox", "api-key"]);
  });
});

describe("addTotals", () => {
  it("adds every field", () => {
    const a = { requests: 1, output_tokens: 2, gwp_ug: { min: 3, max: 4 }, energy_mwh: { min: 5, max: 6 }, generic_models: 1 };
    expect(addTotals(a, a)).toEqual({ requests: 2, output_tokens: 4, gwp_ug: { min: 6, max: 8 }, energy_mwh: { min: 10, max: 12 }, generic_models: 2 });
  });
});
