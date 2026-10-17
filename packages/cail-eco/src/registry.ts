export interface Range { min: number; max: number }
export interface Zone { gwp: number; adpe?: number; pe?: number; wue?: number }
export interface RegistryDefaults { zone: string; pue: Range; unknown_params: { total: Range; active: Range } }
export interface ModelEntry { total: Range; active: Range; tps?: number; ttft?: number; zone?: string; confidence?: string; source?: string; provider?: string }
export interface SkipEntry { skip: true; source?: string }
export type RegistryEntry = ModelEntry | SkipEntry;
export interface Pattern extends ModelEntry { match: string }
export interface Registry {
  version: 2; source: string; generated: string; defaults: RegistryDefaults;
  zones: Record<string, Zone>; zone_prefixes: Array<{ prefix: string; zone: string }>;
  provider_zones: Record<string, string>; ids: Record<string, unknown>; unresolved_ids: string[]; patterns: unknown[];
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isRange = (v: unknown): v is Range => isObject(v) && typeof v.min === "number" && typeof v.max === "number";
const positiveRange = (v: unknown): v is Range => isRange(v) && v.min > 0 && v.min <= v.max && Number.isFinite(v.max);

export function isSkip(entry: RegistryEntry): entry is SkipEntry { return "skip" in entry && entry.skip === true; }

/** Mirrors Python `_usable`: a skip marker, or positive numeric total/active ranges. */
function usable(value: unknown): RegistryEntry | null {
  if (!isObject(value)) return null;
  if (value.skip === true) return { skip: true };
  if (!positiveRange(value.total) || !positiveRange(value.active)) return null;
  const entry: ModelEntry = { total: value.total, active: value.active };
  if (typeof value.tps === "number") entry.tps = value.tps;
  if (typeof value.ttft === "number") entry.ttft = value.ttft;
  if (typeof value.zone === "string") entry.zone = value.zone;
  if (typeof value.confidence === "string") entry.confidence = value.confidence;
  return entry;
}

export function isRegistry(value: unknown): value is Registry {
  return isObject(value) && value.version === 2 && isObject(value.ids) && isObject(value.zones)
    && isObject(value.defaults) && isRange(value.defaults.pue);
}

/** Mirrors Python `find_id_entry`: exact key, else the longest key that is a whole trailing segment after '.' or '/'. */
function findIdKey(modelId: string, ids: Record<string, unknown>): string | null {
  const mid = modelId.toLowerCase().trim();
  if (mid in ids) return mid;
  let best: string | null = null;
  for (const key of Object.keys(ids)) {
    if ((mid.endsWith(`.${key}`) || mid.endsWith(`/${key}`)) && (best === null || key.length > best.length)) best = key;
  }
  return best;
}

const normalize = (modelId: string): string => modelId.toLowerCase().trim().replaceAll(".", "-");

function findPattern(modelId: string, patterns: unknown[]): unknown {
  const norm = normalize(modelId);
  let best: { match: string; entry: unknown } | null = null;
  for (const p of patterns) {
    if (!isObject(p) || typeof p.match !== "string" || p.match === "") continue;
    if (norm.includes(p.match) && (best === null || p.match.length > best.match.length)) best = { match: p.match, entry: p };
  }
  return best?.entry;
}

/** Mirrors Python `resolve_entry`. Returns null for unknown, unresolved or malformed ids (generic estimate). */
export function resolveEntry(modelIds: readonly string[], registry: Registry): RegistryEntry | null {
  for (const mid of modelIds) {
    const key = findIdKey(mid, registry.ids);
    if (key !== null) return usable(registry.ids[key]);
  }
  const unresolved: Record<string, unknown> = Object.fromEntries(registry.unresolved_ids.map((id) => [id, true]));
  if (modelIds.some((mid) => findIdKey(mid, unresolved) !== null)) return null;
  for (const mid of modelIds) {
    const entry = findPattern(mid, registry.patterns);
    if (entry !== undefined) return usable(entry);
  }
  return null;
}

export function genericEntry(registry: Registry): ModelEntry {
  return { total: registry.defaults.unknown_params.total, active: registry.defaults.unknown_params.active };
}

/** Mirrors Python `zone_for`. */
export function zoneFor(modelId: string, registry: Registry, entry: RegistryEntry | null): Zone {
  // SAFETY: the `in` check on the same key guarantees the lookup is defined.
  if (entry !== null && !isSkip(entry) && entry.zone !== undefined && entry.zone in registry.zones) return registry.zones[entry.zone]!;
  const mid = modelId.toLowerCase().trim();
  let zoneName = registry.defaults.zone;
  for (const zp of registry.zone_prefixes) { if (mid.startsWith(zp.prefix)) { zoneName = zp.zone; break; } }
  return registry.zones[zoneName] ?? registry.zones.WOR ?? { gwp: 0.45829 };
}
