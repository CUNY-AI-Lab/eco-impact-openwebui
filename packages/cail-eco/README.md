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
Python and `test/impacts.test.ts` here, at a relative tolerance of `1e-9`. The
shared fixture pins the impact chain (`impactsSingle`), so drift in either
language fails CI. Registry lookup and formatting are not pinned by a shared
fixture: their parity rests on TypeScript tests that mirror the Python suite's
cases, and a change to one side has to be mirrored on the other by hand.

When EcoLogits changes its methodology (the sync job's `VENDORED_LLM_BLOB`
guard trips), re-vendor `src/impacts.ts` together with `eco_impact_filter.py`
and refresh the fixture.

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
- `resolveEntry`, `zoneFor` and `genericEntry` mirror the filter's registry
  lookup: exact ids first, connection prefixes ending in `.` or `/`, no
  fallback to look-alike patterns for unresolved Gateway ids, and a generic
  estimate for unknown or malformed entries. An entry is malformed when its
  ranges are not positive with `min <= max`, or when it carries a `tps` that is
  not a finite number above 0 or a `ttft` that is not a finite number at or
  above 0 (either would make the latency term negative or infinite).
- `isRegistry` accepts only a version-2 registry in which every field the
  lookup reads is present and well-typed, so a malformed fetch is rejected
  rather than failing later.
- `estimateRows(rows, registry)` aggregates `{ tool, model, requests,
  output_tokens }` rows per tool. Skip entries (speech, embeddings) add requests
  but no impact. Rows estimated from the generic entry are counted in
  `generic_models`. It throws a `RangeError` naming the field when a row's
  `tool` is outside the enum, `model` is not a string, or `requests` or
  `output_tokens` is not a safe nonnegative integer. Validate rows first.
  It does not guard against results beyond `Number.MAX_SAFE_INTEGER` for
  absurd inputs (for example an `output_tokens` near the safe-integer limit);
  consumers validate rows with their own schema before calling it, as
  Admission does.
- `addTotals`, `formatRange` and `comparison` combine totals and render them.
  `formatRange` picks the display unit after rounding, so `999.6 g` prints
  `~1.00 kg`, `0.09996 g` prints `~0.10 g` rather than `100.0 mg`, and a zero
  range prints `0 g` or `0 Wh`. The Python filter's `_fmt_range` has no kg
  tier and never sees a zero estimate, so it can still print `100.0 mg` at that
  boundary; the TypeScript behavior matches the Model Access page.

The tool enum is fixed: `sandbox`, `agent-studio`, `site-studio`,
`media-tools`, `workbench`, `api-key`, `unattributed`.

## Per-request latency

Some registry entries carry a measured time to first token (`ttft`, seconds),
which the EcoLogits chain adds once to each request's latency. Latency enters
the server-energy and embodied terms linearly, so N requests with the same
model equal one evaluation over their summed output tokens with `ttft * N`.
`estimateRows` does exactly that: for a `ttft` entry it evaluates a copy of the
entry with `ttft` multiplied by the row's `requests`, which matches summing the
filter's per-message estimates to within floating-point error.

Every request in a row with output tokens pays the per-request term, including
requests that produced zero output tokens (the Python filter skips those
messages). An aggregate cannot tell those requests apart, so the estimate runs
slightly high for them, and that overcount is deliberate. A row whose
`output_tokens` is zero still estimates zero. Entries without `ttft` have no
per-request term, so this does not affect them.

## Integer units

Aggregates cross service boundaries as integers, so totals can be summed and
stored without floating-point drift:

- `gwp_ug`: micrograms CO2e
- `energy_mwh`: milliwatt-hours

Each is a `{ min, max }` pair of safe nonnegative integers with
`min <= max`. Each row is rounded once, then rows are summed. Rounding a
row once rather than rounding each request separately can shift a total by up
to half a unit per request; comparing a row against the summed per-request
floats rounded once, the two agree to within 1 µg (or 1 mWh). Convert back for
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
bun run check   # oxlint anti-slop rules + typecheck (src and tests) + vitest
bun run build   # emits dist/
```

`bun run lint` applies the CUNY AI Lab fleet's anti-slop oxlint rules
(`oxlint.config.ts`, plugin copied verbatim from `cail-tools-admission` into
`tools/oxlint/anti-slop/`), so a vendored copy of `src/` passes the consuming
repository's lint unchanged.

## Distribution

Until npm trusted publishing is configured for `@cuny-ai-lab/cail-eco`,
consumers vendor a copy of `src/` and record the commit they copied from.
After that, a GitHub release tagged `cail-eco-v<version>` publishes the
package through `.github/workflows/publish-cail-eco.yml`.
