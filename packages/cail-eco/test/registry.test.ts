import { describe, expect, it } from "vitest";
import { EMBEDDED_REGISTRY } from "../src/embedded-registry.js";
import { genericEntry, isRegistry, resolveEntry, zoneFor, type Registry } from "../src/registry.js";

const registry: Registry = {
  version: 2, source: "test", generated: "2026-10-12",
  defaults: { zone: "WOR", pue: { min: 1.09, max: 1.2 }, unknown_params: { total: { min: 8, max: 440 }, active: { min: 8, max: 110 } } },
  zones: { WOR: { gwp: 0.45829 }, USA: { gwp: 0.3844 } },
  zone_prefixes: [{ prefix: "us.", zone: "USA" }],
  provider_zones: { "workers-ai": "USA" },
  ids: {
    "glm-4.7": { total: { min: 355, max: 355 }, active: { min: 32, max: 32 }, zone: "USA", confidence: "published" },
    whisper: { skip: true },
    broken: { total: { min: 0, max: 1 }, active: { min: 1, max: 1 } },
  },
  unresolved_ids: ["glm-5.3"],
  patterns: [{ match: "glm-5", total: { min: 1, max: 1 }, active: { min: 1, max: 1 } }],
};

describe("resolveEntry", () => {
  it("exact id beats legacy pattern", () => {
    expect(resolveEntry(["glm-4.7"], registry)).toMatchObject({ zone: "USA" });
  });
  it("accepts a connection prefix ending in . or / and never bleeds versions", () => {
    expect(resolveEntry(["gateway.glm-4.7"], registry)).toMatchObject({ zone: "USA" });
    expect(resolveEntry(["openai/glm-4.7"], registry)).toMatchObject({ zone: "USA" });
    expect(resolveEntry(["glm-4.7.1"], registry)).toBeNull();
    expect(resolveEntry(["xglm-4.7"], registry)).toBeNull();
  });
  it("unresolved gateway id never falls back to patterns", () => {
    expect(resolveEntry(["glm-5.3"], registry)).toBeNull();
  });
  it("skip entries resolve to the skip marker", () => {
    expect(resolveEntry(["whisper"], registry)).toEqual({ skip: true });
  });
  it("malformed entry is unknown", () => {
    expect(resolveEntry(["broken"], registry)).toBeNull();
  });
  it("generic entry uses unknown_params", () => {
    expect(genericEntry(registry)).toEqual({ total: { min: 8, max: 440 }, active: { min: 8, max: 110 } });
  });
});

describe("zoneFor", () => {
  it("entry zone beats prefix and default", () => {
    expect(zoneFor("glm-4.7", registry, resolveEntry(["glm-4.7"], registry)).gwp).toBe(0.3844);
    expect(zoneFor("us.anything", registry, null).gwp).toBe(0.3844);
    expect(zoneFor("anything", registry, null).gwp).toBe(0.45829);
  });
});

describe("isRegistry", () => {
  it("accepts the shape and rejects version 1 or missing ids", () => {
    expect(isRegistry(registry)).toBe(true);
    expect(isRegistry({ ...registry, version: 1 })).toBe(false);
    expect(isRegistry({ ...registry, ids: undefined })).toBe(false);
  });
});

describe("EMBEDDED_REGISTRY", () => {
  it("embedded registry passes isRegistry", () => {
    expect(isRegistry(EMBEDDED_REGISTRY)).toBe(true);
  });
});
