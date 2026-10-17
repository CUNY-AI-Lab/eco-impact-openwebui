import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { impactsSingle } from "../src/impacts.js";

interface GoldenCase {
  name: string; active_params_b: number; total_params_b: number; output_tokens: number;
  mix_gwp: number; pue: number; tps: number | null; ttft: number | null;
  energy_kwh: number; gwp_kg: number;
}
interface Golden { relative_tolerance: number; cases: GoldenCase[] }

const golden = JSON.parse(
  readFileSync(new URL("../../../fixtures/ecologits-golden.json", import.meta.url), "utf8"),
) as Golden; // SAFETY: fixture is checked in beside this test and validated by the Python suite.

const close = (actual: number, expected: number): void => {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * golden.relative_tolerance);
};

describe("impactsSingle", () => {
  for (const c of golden.cases) {
    it(`reproduces EcoLogits for ${c.name}`, () => {
      const result = impactsSingle({
        activeParamsB: c.active_params_b, totalParamsB: c.total_params_b, outputTokens: c.output_tokens,
        mixGwp: c.mix_gwp, pue: c.pue, ...(c.tps === null ? {} : { tps: c.tps }), ...(c.ttft === null ? {} : { ttft: c.ttft }),
      });
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
