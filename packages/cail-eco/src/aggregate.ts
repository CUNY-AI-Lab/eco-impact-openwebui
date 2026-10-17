import { computeImpacts } from "./impacts.js";
import { genericEntry, isSkip, resolveEntry, zoneFor, type Registry } from "./registry.js";

export const TOOLS = ["sandbox", "agent-studio", "site-studio", "media-tools", "workbench", "api-key", "unattributed"] as const;
export type Tool = (typeof TOOLS)[number];
export const isTool = (value: unknown): value is Tool => typeof value === "string" && (TOOLS as readonly string[]).includes(value);

export interface UsageRow { tool: Tool; model: string; requests: number; output_tokens: number }
/** Integer transport range: `gwp_ug` is micrograms CO2e, `energy_mwh` is milliwatt-hours. */
export interface IntegerRange { min: number; max: number }
export interface CarbonTotals { requests: number; output_tokens: number; gwp_ug: IntegerRange; energy_mwh: IntegerRange; generic_models: number }
export interface CarbonToolRow extends CarbonTotals { tool: Tool }
export interface CarbonEstimate { totals: CarbonTotals; tools: CarbonToolRow[] }

export const zeroTotals = (): CarbonTotals => ({ requests: 0, output_tokens: 0, gwp_ug: { min: 0, max: 0 }, energy_mwh: { min: 0, max: 0 }, generic_models: 0 });

export function addTotals(a: CarbonTotals, b: CarbonTotals): CarbonTotals {
  return {
    requests: a.requests + b.requests, output_tokens: a.output_tokens + b.output_tokens,
    gwp_ug: { min: a.gwp_ug.min + b.gwp_ug.min, max: a.gwp_ug.max + b.gwp_ug.max },
    energy_mwh: { min: a.energy_mwh.min + b.energy_mwh.min, max: a.energy_mwh.max + b.energy_mwh.max },
    generic_models: a.generic_models + b.generic_models,
  };
}

function rowTotals(row: UsageRow, registry: Registry): CarbonTotals {
  const base = { ...zeroTotals(), requests: row.requests, output_tokens: row.output_tokens };
  const resolved = resolveEntry([row.model], registry);
  if (resolved !== null && isSkip(resolved)) return base;
  const generic = resolved === null;
  const entry = resolved ?? genericEntry(registry);
  if (row.output_tokens <= 0) return { ...base, generic_models: generic ? 1 : 0 };
  const zone = zoneFor(row.model, registry, resolved);
  const impact = computeImpacts(entry, row.output_tokens, zone, registry.defaults);
  return {
    ...base,
    gwp_ug: { min: Math.round(impact.gwpG[0] * 1e6), max: Math.round(impact.gwpG[1] * 1e6) },
    energy_mwh: { min: Math.round(impact.energyWh[0] * 1e3), max: Math.round(impact.energyWh[1] * 1e3) },
    generic_models: generic ? 1 : 0,
  };
}

export function estimateRows(rows: readonly UsageRow[], registry: Registry): CarbonEstimate {
  const byTool = new Map<Tool, CarbonTotals>();
  for (const row of rows) byTool.set(row.tool, addTotals(byTool.get(row.tool) ?? zeroTotals(), rowTotals(row, registry)));
  const tools = [...byTool.entries()].map(([tool, totals]) => ({ tool, ...totals })).sort((a, b) => b.gwp_ug.max - a.gwp_ug.max);
  const totals = tools.reduce<CarbonTotals>((acc, t) => addTotals(acc, t), zeroTotals());
  return { totals, tools };
}
