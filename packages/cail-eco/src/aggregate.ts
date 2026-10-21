import { computeImpacts } from "./impacts.js";
import { isString } from "./json.js";
import { genericEntry, isSkip, resolveEntry, zoneFor, type ModelEntry, type Registry } from "./registry.js";

export const TOOLS = ["sandbox", "agent-studio", "site-studio", "media-tools", "workbench", "api-key", "unattributed"] as const;
export type Tool = (typeof TOOLS)[number];
export const isTool = <Value>(value: Value): value is Value & Tool => TOOLS.some((tool) => tool === value);

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
  // The chain is affine in latency and ttft is paid once per request, so N requests with the same model equal
  // one evaluation over the summed tokens with ttft * N. Copy the entry; the registry entry stays untouched.
  const scaled: ModelEntry = entry.ttft === undefined ? entry : { ...entry, ttft: entry.ttft * row.requests };
  const impact = computeImpacts(scaled, row.output_tokens, zone, registry.defaults);
  return {
    ...base,
    gwp_ug: { min: Math.round(impact.gwpG[0] * 1e6), max: Math.round(impact.gwpG[1] * 1e6) },
    energy_mwh: { min: Math.round(impact.energyWh[0] * 1e3), max: Math.round(impact.energyWh[1] * 1e3) },
    generic_models: generic ? 1 : 0,
  };
}

/** `Number.isSafeInteger` is false for every non-number, so no separate number check is needed. */
const isCount = (v: number): boolean => Number.isSafeInteger(v) && v >= 0;

/** Throws a RangeError naming the first invalid field; callers are expected to validate rows before this point. */
function assertRow(row: UsageRow): void {
  if (!isTool(row.tool)) throw new RangeError(`tool must be one of the CAIL tools, got ${JSON.stringify(row.tool)}`);
  if (!isString(row.model)) throw new RangeError(`model must be a string, got ${JSON.stringify(row.model)}`);
  if (!isCount(row.requests)) throw new RangeError(`requests must be a safe nonnegative integer, got ${String(row.requests)}`);
  if (!isCount(row.output_tokens)) throw new RangeError(`output_tokens must be a safe nonnegative integer, got ${String(row.output_tokens)}`);
}

/** Validates every row up front (RangeError on bad input), then sums per-tool totals, largest gwp_ug.max first. */
export function estimateRows(rows: readonly UsageRow[], registry: Registry): CarbonEstimate {
  for (const row of rows) assertRow(row);
  const byTool = new Map<Tool, CarbonTotals>();
  for (const row of rows) byTool.set(row.tool, addTotals(byTool.get(row.tool) ?? zeroTotals(), rowTotals(row, registry)));
  const tools = [...byTool.entries()].map(([tool, totals]) => ({ tool, ...totals })).sort((a, b) => b.gwp_ug.max - a.gwp_ug.max);
  const totals = tools.reduce<CarbonTotals>((acc, t) => addTotals(acc, t), zeroTotals());
  return { totals, tools };
}
