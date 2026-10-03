# Gateway-sourced weekly registry — design record

Date: 2026-10-03 · Status: implemented on branch `gateway-weekly-registry`

## Problem

CAIL now serves Open WebUI through CAIL Gateway (Workers AI + Bedrock Mantle behind
Cloudflare AI Gateway). Gateway admits models automatically from provider catalogs, so
new models appear without anyone touching this filter. The registry was refreshed by
hand (`sync-eco-models.py --owui-url` → review → push to EFS per environment), matched
models by longest substring (so `glm-5.3` silently used `glm-5`'s numbers), and chose
the electricity zone from Bedrock-era id prefixes that Gateway ids no longer carry.

## Decisions

| Question | Decision |
|---|---|
| Where automation runs | Weekly GitHub Action in this repo (Mon 06:00 UTC + manual dispatch). |
| How the filter gets the registry | Filter fetches `eco_models.json` from this repo's `registry` branch (raw URL), at most every 12 h, in the background; never blocks a reply. |
| Electricity zone | US grid for every Gateway provider (Mantle is us-east-1; Workers AI runs at the nearest Cloudflare GPU PoP to NYC users). Stored per provider so it can change. |
| Publishing | Job commits straight to the `registry` data branch when validation passes. `main` is protected (required `pytest` check) and the repo doesn't let Actions open PRs, so the data lives on its own branch, and a held build goes to `registry-review` with a compare link in the tracking issue. |

## Weekly job (`scripts/sync-eco-models.py --gateway-url …`)

Inputs: Gateway public `GET /v1/catalog`; EcoLogits data at its latest release tag;
Hugging Face `config.json` + `api/models/<repo>/revision/<rev>` for each catalog row that
links one; `scripts/eco-extra-patterns.json` (curated, always wins).

Per catalog row (`task` = Automatic Speech Recognition and other non-chat rows are
written with `"skip": true`), the first rule that resolves wins:

1. **Published curated/EcoLogits exact name** — a pattern equal to one of the row's
   names (id, `upstream_model`, `model_group` basename, HF repo basename), also after
   stripping variant suffixes (`-instruct`, `-it`, `-chat`, `-fp8`, `-fast`, `-lora`,
   `-hf`, …). Exact only: no substring or prefix matching for Gateway rows.
2. **Name hint** — an `aNb` token (e.g. `qwen3-vl-235b-a22b`) gives active params;
   total comes from HF safetensors, the catalog `size.label`, or the `NNNb` token.
   Confidence `published`.
3. **Hugging Face config** — total from safetensors; active = total − inactive routed
   expert weights (`3·hidden·moe_intermediate·(E−k)` per MoE layer). The formula
   over-counts by roughly 5–25% (shared experts, MTP layers), so active is stored as
   `[0.8·a, a]`. Dense configs (no expert keys) use active = total. Confidence `derived`.
4. **Estimated curated/EcoLogits exact name.** Estimates were often written before a
   model's weights were released (the first live run found `qwen3-coder-next` at
   600B/50B and `glm-4.7-flash` at 120B; the published figures are 80B/3B and 31B/3B),
   so they rank below Hugging Face data.
5. Otherwise **unresolved**: listed in `unresolved_ids` so the filter shows the flagged
   generic estimate instead of a look-alike legacy pattern, and in the tracking issue.

The catalog `size.label` alone is never trusted as "dense": several MoE architectures
(Kimi, Nemotron, Mistral Large 3) do not say MoE in their name.

Curated entries may set `"exact_only": true` (e.g. Mantle's bare `v3.2`) to stay out of
the legacy substring patterns. Output adds an `ids` map (exact Gateway id / upstream id / model_group → entry with
`zone`). `patterns` and `zone_prefixes` stay for legacy direct-provider connections.

**Gate (else open a PR instead of committing):** schema valid; no id that was covered
last week loses coverage; no active-param max moves by more than 3×; methodology guard —
if `ecologits/impacts/llm.py` at the new EcoLogits tag differs from the vendored blob
(`aef8473e…`), keep the previous EcoLogits ref and open an issue.

## Filter (v1.1.0)

- New valves: `registry_url` (default: this repo's raw `main` URL; empty disables),
  `registry_refresh_hours` (12).
- Fetch runs via `asyncio.to_thread` with a 5 s timeout; a good fetch replaces the
  in-memory registry and is written atomically to `registry_path`, which doubles as the
  warm cache across restarts. Failures keep the last good copy. Fallback chain: memory →
  `registry_path` → embedded.
- Lookup: exact `ids` (also matching a connection prefix such as `gateway.<id>`), for
  both the message's model id and the model card's `base_model_id`; ids in
  `unresolved_ids` go straight to generic; then legacy substring `patterns`; then generic.
- Zone: entry `zone` → legacy `zone_prefixes` → default.
- Entries with `skip` emit nothing.

## Out of scope

Gateway contract changes (e.g. exposing the serving provider per response); deploying
the new function version to dev/prod Open WebUI (operator step, dev first).

## First live run (2026-10-03)

60 catalog rows: 55 resolved (35 published, 11 derived from Hugging Face, 9 estimated),
3 skipped (Whisper), 2 unresolved (`glm-5.3`, `glm-5.3-flash`: no published parameters
or Hugging Face weights yet). A second run against the first output was byte-identical,
so quiet weeks produce no commit.
