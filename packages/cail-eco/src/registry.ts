import { isJsonObject, isNumber, isString, type JsonObject, type JsonValue } from "./json.js";

export type Range = { min: number; max: number };
export type Zone = { gwp: number; adpe?: number; pe?: number; wue?: number };
export type RegistryDefaults = { zone: string; pue: Range; unknown_params: { total: Range; active: Range } };
export type ModelEntry = { total: Range; active: Range; tps?: number; ttft?: number; zone?: string; confidence?: string; source?: string; provider?: string };
export type SkipEntry = { skip: true; source?: string };
export type RegistryEntry = ModelEntry | SkipEntry;
export type Registry = {
  version: 2; source: string; generated: string; defaults: RegistryDefaults;
  zones: Record<string, Zone>; zone_prefixes: Array<{ prefix: string; zone: string }>;
  provider_zones: Record<string, string>; ids: Record<string, JsonValue>; unresolved_ids: string[]; patterns: JsonValue[];
};

const isRange = (v: JsonValue | undefined): v is Range => isJsonObject(v) && isNumber(v.min) && isNumber(v.max);
const positiveRange = (v: JsonValue | undefined): v is Range => isRange(v) && v.min > 0 && v.min <= v.max && Number.isFinite(v.max);
/** Own-property lookup, so inherited names such as `constructor` or `__proto__` never match. */
const own = <T>(record: Record<string, T>, key: string): T | undefined => (Object.hasOwn(record, key) ? record[key] : undefined);

export function isSkip(entry: RegistryEntry): entry is SkipEntry { return "skip" in entry && entry.skip === true; }

/** Mirrors Python `_usable`: a skip marker, or positive numeric total/active ranges. */
function usable(value: JsonValue | undefined): RegistryEntry | null {
  if (!isJsonObject(value)) return null;
  if (value.skip === true) return { skip: true };
  if (!positiveRange(value.total) || !positiveRange(value.active)) return null;
  const entry: ModelEntry = { total: value.total, active: value.active };
  if (isNumber(value.tps)) entry.tps = value.tps;
  if (isNumber(value.ttft)) entry.ttft = value.ttft;
  if (isString(value.zone)) entry.zone = value.zone;
  if (isString(value.confidence)) entry.confidence = value.confidence;
  return entry;
}

const isStringArray = (v: JsonValue | undefined): v is string[] => Array.isArray(v) && v.every(isString);
const isZone = (v: JsonValue | undefined): v is Zone => isJsonObject(v) && isNumber(v.gwp) && Number.isFinite(v.gwp) && v.gwp >= 0;
const isZonePrefix = (v: JsonValue | undefined): v is { prefix: string; zone: string } => isJsonObject(v) && isString(v.prefix) && isString(v.zone);

function isDefaults(v: JsonValue | undefined): v is RegistryDefaults {
  return isJsonObject(v) && isString(v.zone) && positiveRange(v.pue)
    && isJsonObject(v.unknown_params) && positiveRange(v.unknown_params.total) && positiveRange(v.unknown_params.active);
}

/**
 * Accepts only a version-2 registry whose every field the lookup consumes is present and well-typed,
 * so a malformed fetch is rejected (and the caller falls back) instead of throwing during lookup.
 * Individual `ids` and `patterns` entries stay unchecked here: `resolveEntry` treats a malformed one as unknown.
 */
export function isRegistry<Value>(value: Value): value is Value & Registry {
  return isJsonObject(value) && value.version === 2 && isString(value.source) && isString(value.generated)
    && isDefaults(value.defaults)
    && isJsonObject(value.zones) && Object.values(value.zones).every(isZone)
    && Array.isArray(value.zone_prefixes) && value.zone_prefixes.every(isZonePrefix)
    && isJsonObject(value.provider_zones) && Object.values(value.provider_zones).every(isString)
    && isJsonObject(value.ids) && isStringArray(value.unresolved_ids) && Array.isArray(value.patterns);
}

/** Mirrors Python `find_id_entry`: exact key, else the longest key that is a whole trailing segment after '.' or '/'. */
function findIdKey(modelId: string, ids: JsonObject): string | null {
  const mid = modelId.toLowerCase().trim();
  if (Object.hasOwn(ids, mid)) return mid;
  let best: string | null = null;
  for (const key of Object.keys(ids)) {
    if ((mid.endsWith(`.${key}`) || mid.endsWith(`/${key}`)) && (best === null || key.length > best.length)) best = key;
  }
  return best;
}

const normalize = (modelId: string): string => modelId.toLowerCase().trim().replaceAll(".", "-");

function findPattern(modelId: string, patterns: readonly JsonValue[]): JsonObject | undefined {
  const norm = normalize(modelId);
  let best: { match: string; entry: JsonObject } | null = null;
  for (const p of patterns) {
    if (!isJsonObject(p) || !isString(p.match) || p.match === "") continue;
    if (norm.includes(p.match) && (best === null || p.match.length > best.match.length)) best = { match: p.match, entry: p };
  }
  return best?.entry;
}

/** Mirrors Python `resolve_entry`. Returns null for unknown, unresolved or malformed ids (generic estimate). */
export function resolveEntry(modelIds: readonly string[], registry: Registry): RegistryEntry | null {
  for (const mid of modelIds) {
    const key = findIdKey(mid, registry.ids);
    if (key === null) continue;
    const value = registry.ids[key];
    // Python `find_id_entry` returns None for a null value and `resolve_entry` moves on to the next candidate.
    if (value === null) continue;
    return usable(value);
  }
  const unresolved: Record<string, boolean> = Object.fromEntries(registry.unresolved_ids.map((id) => [id, true]));
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
  const entryZone = entry !== null && !isSkip(entry) && entry.zone !== undefined ? own(registry.zones, entry.zone) : undefined;
  if (entryZone !== undefined) return entryZone;
  const mid = modelId.toLowerCase().trim();
  let zoneName = registry.defaults.zone;
  for (const zp of registry.zone_prefixes) { if (mid.startsWith(zp.prefix)) { zoneName = zp.zone; break; } }
  return own(registry.zones, zoneName) ?? own(registry.zones, "WOR") ?? { gwp: 0.45829 };
}
