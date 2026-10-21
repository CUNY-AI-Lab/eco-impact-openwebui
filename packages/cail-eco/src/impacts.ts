// Impact math vendored from mlco2/ecologits, commit
// 42154236c3b275346e8b97b04f49cd19877316e6, file ecologits/impacts/llm.py (MPL-2.0).
// Keep identical to eco_impact_filter.py; fixtures/ecologits-golden.json pins both.

import type { ModelEntry, RegistryDefaults, Zone } from "./registry.js";

export const MODEL_QUANTIZATION_BITS = 16;
export const GPU_ENERGY_ALPHA = 1.1665273170451914e-6;
export const GPU_ENERGY_BETA = -0.011205921025579175;
export const GPU_ENERGY_GAMMA = 4.052928146734005e-5;
export const LATENCY_ALPHA = 0.0006785088094353663;
export const LATENCY_BETA = 0.0003119310311688259;
export const LATENCY_GAMMA = 0.019473717579473387;
export const GPU_MEMORY = 80;
export const GPU_EMBODIED_IMPACT_GWP = 273;
export const SERVER_GPUS = 8;
export const SERVER_POWER = 1.2;
export const SERVER_EMBODIED_IMPACT_GWP = 5700;
export const HARDWARE_LIFESPAN = 3 * 365 * 24 * 60 * 60;
export const BATCH_SIZE = 64;

export interface ImpactInput {
  activeParamsB: number;
  totalParamsB: number;
  outputTokens: number;
  mixGwp: number;
  pue: number;
  tps?: number;
  ttft?: number;
  requestLatency?: number;
}

export interface ImpactResult { energyKwh: number; gwpKg: number }

export function impactsSingle(input: ImpactInput): ImpactResult {
  const { activeParamsB, totalParamsB, outputTokens, mixGwp, pue } = input;
  const gpuEnergyPerTokenKwh =
    (GPU_ENERGY_ALPHA * Math.exp(GPU_ENERGY_BETA * BATCH_SIZE) * activeParamsB + GPU_ENERGY_GAMMA) / 1000;
  const gpuEnergyKwh = outputTokens * gpuEnergyPerTokenKwh;
  const latencyPerToken =
    input.tps !== undefined && input.tps > 0
      ? 1 / input.tps
      : LATENCY_ALPHA * activeParamsB + LATENCY_BETA * BATCH_SIZE + LATENCY_GAMMA;
  const gpuLatency = outputTokens * latencyPerToken + (input.ttft ?? 0);
  const latency = Math.min(input.requestLatency ?? Number.POSITIVE_INFINITY, gpuLatency);
  const requiredMemoryGb = (1.2 * totalParamsB * MODEL_QUANTIZATION_BITS) / 8;
  const gpuCount = 2 ** Math.ceil(Math.log2(Math.max(1, Math.ceil(requiredMemoryGb / GPU_MEMORY))));
  const serverEnergyKwh = (latency / 3600) * SERVER_POWER * (gpuCount / SERVER_GPUS) / BATCH_SIZE;
  const itEnergyKwh = serverEnergyKwh + gpuCount * gpuEnergyKwh;
  const energyKwh = pue * itEnergyKwh;
  const usageGwpKg = energyKwh * mixGwp;
  const embodiedUnitGwp = (gpuCount / SERVER_GPUS) * SERVER_EMBODIED_IMPACT_GWP + gpuCount * GPU_EMBODIED_IMPACT_GWP;
  const embodiedGwpKg = (latency * embodiedUnitGwp) / (HARDWARE_LIFESPAN * BATCH_SIZE);
  return { energyKwh, gwpKg: usageGwpKg + embodiedGwpKg };
}

export interface ImpactRange { energyWh: [number, number]; gwpG: [number, number] }

/** Mirrors Python `compute_impacts`: min/max impacts for a registry entry, in Wh and g CO2e. */
export function computeImpacts(entry: ModelEntry, outputTokens: number, zone: Zone, defaults: RegistryDefaults): ImpactRange {
  const pue = defaults.pue;
  const extra: Pick<ImpactInput, "tps" | "ttft"> = {};
  if (entry.tps !== undefined) extra.tps = entry.tps;
  if (entry.ttft !== undefined) extra.ttft = entry.ttft;
  const lo = impactsSingle({ activeParamsB: entry.active.min, totalParamsB: entry.total.min, outputTokens, mixGwp: zone.gwp, pue: pue.min, ...extra });
  const hi = impactsSingle({ activeParamsB: entry.active.max, totalParamsB: entry.total.max, outputTokens, mixGwp: zone.gwp, pue: pue.max, ...extra });
  return { energyWh: [lo.energyKwh * 1000, hi.energyKwh * 1000], gwpG: [lo.gwpKg * 1000, hi.gwpKg * 1000] };
}
