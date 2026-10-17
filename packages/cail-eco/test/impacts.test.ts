import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { computeImpacts, impactsSingle, type ImpactInput } from "../src/impacts.js";

interface GoldenCase {
  name: string; active_params_b: number; total_params_b: number; output_tokens: number;
  mix_gwp: number; pue: number; tps: number | null; ttft: number | null;
  energy_kwh: number; gwp_kg: number;
}
interface Golden { relative_tolerance: number; cases: GoldenCase[] }

// The fixture is checked in at the repository root and validated by the Python suite; JSON.parse returns `any`.
const golden: Golden = JSON.parse(readFileSync(new URL("../../../fixtures/ecologits-golden.json", import.meta.url), "utf8"));

const close = (actual: number, expected: number): void => {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * golden.relative_tolerance);
};

describe("impactsSingle", () => {
  for (const c of golden.cases) {
    it(`reproduces EcoLogits for ${c.name}`, () => {
      const input: ImpactInput = {
        activeParamsB: c.active_params_b, totalParamsB: c.total_params_b, outputTokens: c.output_tokens,
        mixGwp: c.mix_gwp, pue: c.pue,
      };
      if (c.tps !== null) input.tps = c.tps;
      if (c.ttft !== null) input.ttft = c.ttft;
      const result = impactsSingle(input);
      close(result.energyKwh, c.energy_kwh);
      close(result.gwpKg, c.gwp_kg);
    });
  }
  it("is monotonic in tokens", () => {
    const a = impactsSingle({ activeParamsB: 8, totalParamsB: 8, outputTokens: 100, mixGwp: 0.45829, pue: 1.2 });
    const b = impactsSingle({ activeParamsB: 8, totalParamsB: 8, outputTokens: 200, mixGwp: 0.45829, pue: 1.2 });
    expect(b.energyKwh).toBeGreaterThan(a.energyKwh);
    expect(b.gwpKg).toBeGreaterThan(a.gwpKg);
  });
  it("zero tokens estimate zero when no ttft", () => {
    const r = impactsSingle({ activeParamsB: 8, totalParamsB: 8, outputTokens: 0, mixGwp: 0.45829, pue: 1.2 });
    expect(r.energyKwh).toBe(0);
    expect(r.gwpKg).toBe(0);
  });
});

describe("computeImpacts", () => {
  it("returns the EcoLogits min/max range in Wh and g", () => {
    const entry = { total: { min: 440, max: 440 }, active: { min: 44, max: 132 }, tps: 46.2, ttft: 0.53 };
    const defaults = { zone: "WOR", pue: { min: 1.09, max: 1.14 }, unknown_params: { total: { min: 8, max: 440 }, active: { min: 8, max: 110 } } };
    const imp = computeImpacts(entry, 500, { gwp: 0.3844 }, defaults);
    close(imp.energyWh[0], 0.7007863545514568);
    close(imp.energyWh[1], 1.1899213268139133);
    close(imp.gwpG[0], 0.29894610495653523);
    close(imp.gwpG[1], 0.4869695882942235);
  });
});
