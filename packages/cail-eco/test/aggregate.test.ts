import { describe, expect, it } from "vitest";
import { addTotals, estimateRows, type UsageRow } from "../src/aggregate.js";
import { computeImpacts } from "../src/impacts.js";
import { genericEntry, type ModelEntry, type Registry, type Zone } from "../src/registry.js";

const knownId = "glm-4.7";
const known: ModelEntry = { total: { min: 355, max: 355 }, active: { min: 32, max: 32 }, zone: "USA" };
const timedId = "glm-5.2";
const timed: ModelEntry = { total: { min: 744, max: 744 }, active: { min: 40, max: 40 }, tps: 57.5, ttft: 0.64, zone: "USA" };
const usa: Zone = { gwp: 0.3844 };
const wor: Zone = { gwp: 0.45829 };

const registry: Registry = {
  version: 2, source: "test", generated: "2026-10-12",
  defaults: { zone: "WOR", pue: { min: 1.09, max: 1.2 }, unknown_params: { total: { min: 8, max: 440 }, active: { min: 8, max: 110 } } },
  zones: { WOR: wor, USA: usa },
  zone_prefixes: [],
  provider_zones: {},
  ids: { [knownId]: known, [timedId]: timed, whisper: { skip: true } },
  unresolved_ids: ["glm-5.3"],
  patterns: [{ match: "glm-5", total: { min: 1, max: 1 }, active: { min: 1, max: 1 } }],
};

describe("estimateRows", () => {
  it("sums per tool and matches computeImpacts for one row", () => {
    const e = estimateRows([{ tool: "sandbox", model: knownId, requests: 3, output_tokens: 1200 }], registry);
    const direct = computeImpacts(known, 1200, usa, registry.defaults);
    expect(e.totals.gwp_ug.min).toBe(Math.round(direct.gwpG[0] * 1e6));
    expect(e.totals.gwp_ug.max).toBe(Math.round(direct.gwpG[1] * 1e6));
    expect(e.totals.energy_mwh.min).toBe(Math.round(direct.energyWh[0] * 1e3));
    expect(e.totals.energy_mwh.max).toBe(Math.round(direct.energyWh[1] * 1e3));
    expect(e.tools).toEqual([{ tool: "sandbox", ...e.totals }]);
    expect(e.totals.generic_models).toBe(0);
  });
  it("unknown model uses the generic estimate and is counted", () => {
    const e = estimateRows([{ tool: "api-key", model: "no-such-model", requests: 1, output_tokens: 100 }], registry);
    expect(e.totals.generic_models).toBe(1);
    expect(e.totals.gwp_ug.max).toBeGreaterThan(e.totals.gwp_ug.min);
  });
  it("unresolved gateway id gets the generic estimate, not a look-alike pattern", () => {
    const e = estimateRows([{ tool: "workbench", model: "glm-5.3", requests: 1, output_tokens: 100 }], registry);
    const generic = computeImpacts(genericEntry(registry), 100, wor, registry.defaults);
    expect(e.totals.generic_models).toBe(1);
    expect(e.totals.gwp_ug.min).toBeGreaterThan(0);
    expect(e.totals.gwp_ug).toEqual({ min: Math.round(generic.gwpG[0] * 1e6), max: Math.round(generic.gwpG[1] * 1e6) });
  });
  it("skip entries add requests but no impact", () => {
    const e = estimateRows([{ tool: "media-tools", model: "whisper", requests: 4, output_tokens: 0 }], registry);
    expect(e.totals).toEqual({ requests: 4, output_tokens: 0, gwp_ug: { min: 0, max: 0 }, energy_mwh: { min: 0, max: 0 }, generic_models: 0 });
  });
  it("zero tokens estimate zero", () => {
    const e = estimateRows([{ tool: "sandbox", model: knownId, requests: 2, output_tokens: 0 }], registry);
    expect(e.totals.gwp_ug).toEqual({ min: 0, max: 0 });
    expect(e.totals.requests).toBe(2);
  });
  it("charges a ttft entry's per-request time once per request, like summing per-message estimates", () => {
    const requests = 10;
    const e = estimateRows([{ tool: "sandbox", model: timedId, requests, output_tokens: 5000 }], registry);
    let sum = { gwpMin: 0, gwpMax: 0, energyMin: 0, energyMax: 0 };
    for (let i = 0; i < requests; i++) {
      const one = computeImpacts(timed, 500, usa, registry.defaults);
      const [gwpMin, gwpMax] = one.gwpG;
      const [energyMin, energyMax] = one.energyWh;
      sum = { gwpMin: sum.gwpMin + gwpMin, gwpMax: sum.gwpMax + gwpMax, energyMin: sum.energyMin + energyMin, energyMax: sum.energyMax + energyMax };
    }
    // The chain is affine in latency, so one evaluation with ttft scaled by the request count equals the sum.
    const scaled = computeImpacts({ ...timed, ttft: 0.64 * requests }, 5000, usa, registry.defaults);
    const relative = (a: number, b: number): number => Math.abs(a / b - 1);
    expect(relative(scaled.gwpG[0], sum.gwpMin)).toBeLessThan(1e-9);
    expect(relative(scaled.gwpG[1], sum.gwpMax)).toBeLessThan(1e-9);
    expect(relative(scaled.energyWh[0], sum.energyMin)).toBeLessThan(1e-9);
    expect(relative(scaled.energyWh[1], sum.energyMax)).toBeLessThan(1e-9);
    // Rounding the summed floats once and rounding the single evaluation once can differ by at most 1 µg (or 1 mWh)
    // from floating-point summation order; within 1e-9 relative they are the same number.
    const pairs: Array<[number, number]> = [
      [e.totals.gwp_ug.min, Math.round(sum.gwpMin * 1e6)], [e.totals.gwp_ug.max, Math.round(sum.gwpMax * 1e6)],
      [e.totals.energy_mwh.min, Math.round(sum.energyMin * 1e3)], [e.totals.energy_mwh.max, Math.round(sum.energyMax * 1e3)],
    ];
    for (const [actual, want] of pairs) expect(Math.abs(actual - want)).toBeLessThanOrEqual(Math.max(1, want * 1e-9));
    expect(registry.ids[timedId]).toEqual({ total: { min: 744, max: 744 }, active: { min: 40, max: 40 }, tps: 57.5, ttft: 0.64, zone: "USA" });
  });
  it("orders tools by gwp max descending", () => {
    const e = estimateRows([
      { tool: "api-key", model: knownId, requests: 1, output_tokens: 10 },
      { tool: "sandbox", model: knownId, requests: 1, output_tokens: 1000 },
    ], registry);
    expect(e.tools.map((t) => t.tool)).toEqual(["sandbox", "api-key"]);
  });
});

describe("estimateRows input checks", () => {
  it("rejects a negative token count", () => {
    expect(() => estimateRows([{ tool: "sandbox", model: knownId, requests: 1, output_tokens: -5 }], registry))
      .toThrow(new RangeError("output_tokens must be a safe nonnegative integer, got -5"));
  });
  it("rejects a non-integer request count", () => {
    expect(() => estimateRows([{ tool: "sandbox", model: knownId, requests: 1.5, output_tokens: 10 }], registry))
      .toThrow(new RangeError("requests must be a safe nonnegative integer, got 1.5"));
  });
  it("rejects a tool outside the enum", () => {
    // Deliberately invalid row: JSON.parse returns `any`, standing in for unvalidated caller data.
    const row: UsageRow = JSON.parse(JSON.stringify({ tool: "chat", model: knownId, requests: 1, output_tokens: 10 }));
    expect(() => estimateRows([row], registry)).toThrow(new RangeError('tool must be one of the CAIL tools, got "chat"'));
  });
  it("rejects a non-string model", () => {
    // Deliberately invalid row: JSON.parse returns `any`, standing in for unvalidated caller data.
    const row: UsageRow = JSON.parse(JSON.stringify({ tool: "sandbox", model: 42, requests: 1, output_tokens: 10 }));
    expect(() => estimateRows([row], registry)).toThrow(new RangeError("model must be a string, got 42"));
  });
});

describe("addTotals", () => {
  it("adds every field", () => {
    const a = { requests: 1, output_tokens: 2, gwp_ug: { min: 3, max: 4 }, energy_mwh: { min: 5, max: 6 }, generic_models: 1 };
    expect(addTotals(a, a)).toEqual({ requests: 2, output_tokens: 4, gwp_ug: { min: 6, max: 8 }, energy_mwh: { min: 10, max: 12 }, generic_models: 2 });
  });
});
