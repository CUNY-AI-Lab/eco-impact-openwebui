# @cuny-ai-lab/cail-eco

A TypeScript port of the EcoLogits inference-impact chain and the CAIL model
registry lookup that the Open WebUI Eco Impact filter (`eco_impact_filter.py`,
at the root of this repository) runs in Python. CAIL services use it to turn
token counts into energy and CO2e estimates with the same numbers the filter
shows in chat. The first consumer is the CAIL Admission Worker, which estimates
each person's cumulative footprint from Cloudflare AI Gateway token aggregates.

The impact math is vendored from [EcoLogits](https://ecologits.ai)
(mlco2/ecologits, commit `42154236c3b275346e8b97b04f49cd19877316e6`,
`ecologits/impacts/llm.py`) and is licensed MPL-2.0, like the rest of this
repository. Cite EcoLogits, not this package.

## Parity with the Python filter

`fixtures/ecologits-golden.json` at the repository root holds golden values
computed with the real EcoLogits library. Both test suites load it:
`test_eco_impact_filter.py::test_golden_fixture_shared_with_typescript` in
Python and `test/impacts.test.ts` here, at a relative tolerance of `1e-9`. If
either implementation drifts, CI fails. When EcoLogits changes its methodology
(the sync job's `VENDORED_LLM_BLOB` guard trips), re-vendor `src/impacts.ts`
together with `eco_impact_filter.py` and refresh the fixture.

## API

```ts
import { EMBEDDED_REGISTRY, estimateRows, formatRange, isRegistry } from "@cuny-ai-lab/cail-eco";

const estimate = estimateRows(
  [{ tool: "sandbox", model: "glm-4.7", requests: 12, output_tokens: 9_400 }],
  EMBEDDED_REGISTRY,
);
estimate.totals.gwp_ug; // { min, max } micrograms CO2e
estimate.tools;         // one row per tool, largest gwp_ug.max first
formatRange(estimate.totals.gwp_ug.min / 1e6, estimate.totals.gwp_ug.max / 1e6, "g");
```

- `impactsSingle(input)` runs one pass of the EcoLogits chain and returns
  `{ energyKwh, gwpKg }`.
- `computeImpacts(entry, outputTokens, zone, defaults)` returns the min/max
  range in Wh and g, like the Python `compute_impacts`.
- `resolveEntry`, `zoneFor`, `genericEntry` and `isRegistry` mirror the
  filter's registry lookup: exact ids first, connection prefixes ending in `.`
  or `/`, no fallback to look-alike patterns for unresolved Gateway ids, and a
  generic estimate for unknown or malformed entries.
- `estimateRows(rows, registry)` aggregates `{ tool, model, requests,
  output_tokens }` rows per tool. Skip entries (speech, embeddings) add requests
  but no impact. Rows estimated from the generic entry are counted in
  `generic_models`.
- `addTotals`, `formatRange` and `comparison` combine totals and render them.

The tool enum is fixed: `sandbox`, `agent-studio`, `site-studio`,
`media-tools`, `workbench`, `api-key`, `unattributed`.

## Integer units

Aggregates cross service boundaries as integers, so totals can be summed and
stored without floating-point drift:

- `gwp_ug`: micrograms CO2e
- `energy_mwh`: milliwatt-hours

Each is a `{ min, max }` pair of safe nonnegative integers with
`min <= max`. Each row is rounded once, then rows are summed. Convert back for
display (`gwp_ug / 1e6` grams, `energy_mwh / 1e3` Wh) and pass the result to
`formatRange`, which scales to mg/mWh or kg/kWh as needed.

## Embedded registry

`src/embedded-registry.ts` is a generated snapshot of the weekly registry
published to this repository's `registry` branch
(`https://raw.githubusercontent.com/CUNY-AI-Lab/eco-impact-openwebui/registry/eco_models.json`,
version 2). Consumers should fetch the live registry, check it with
`isRegistry`, and fall back to `EMBEDDED_REGISTRY` when the fetch fails.
Refresh the snapshot with:

```bash
bun run embed-registry
```

## Development

```bash
cd packages/cail-eco
bun install
bun run check   # typecheck + vitest
bun run build   # emits dist/
```

## Distribution

Until npm trusted publishing is configured for `@cuny-ai-lab/cail-eco`,
consumers vendor a copy of `src/` and record the commit they copied from.
After that, a GitHub release tagged `cail-eco-v<version>` publishes the
package through `.github/workflows/publish-cail-eco.yml`.
