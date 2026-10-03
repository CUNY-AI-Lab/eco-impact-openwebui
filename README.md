# Eco Impact Estimator for Open WebUI

A single-file [Open WebUI](https://github.com/open-webui/open-webui) **Filter
function** that shows an estimated **energy use and CO₂e footprint under every
model response**, for every model, as a small status line:

> 🌱 0.27–0.29 g CO₂e · 0.48–0.53 Wh · 🌬️ ~1.4 breaths

It applies the physically-grounded methodology of
[**EcoLogits**](https://ecologits.ai) (parameter-count → GPU/server energy →
data-center overhead → electricity-mix CO₂e, plus embodied hardware impact) to
whatever models your instance serves — OpenRouter, AWS Bedrock, Cloudflare
Workers AI, local models, etc. Estimates are shown as **min–max ranges** to
reflect the genuine uncertainty in these numbers.

This project builds directly on EcoLogits' research and data. See
[Attribution & license](#attribution--license) — it matters here, and it is
short.

---

## Why

Most "carbon" add-ons hard-code a single grams-per-token number per model.
That is easy but wrong: energy per token depends on the model's *active*
parameter count, the serving hardware, batching, data-center efficiency, and
the local electricity mix. EcoLogits models all of that. This filter packages
that methodology into a drop-in Open WebUI function so any operator can give
users honest, per-model awareness of what a request costs — without a database,
a sidecar, or a source-code patch.

## What it does

- **Works for every model**, globally, with no per-model setup.
- **Uses real token counts** from the response when the provider reports them,
  and falls back to a length-based estimate when it doesn't.
- **Never interferes with chat.** It only *reads* the response and emits a
  status event; it never modifies message content and never blocks a reply. Any
  internal error is swallowed and the response passes through untouched.
- **Shows uncertainty honestly** — min–max ranges, and a `⚠️ generic estimate`
  marker when a model isn't in the registry (it still shows a wide-range guess
  rather than nothing).
- **Self-contained.** One `.py` file, standard library + `pydantic` (already an
  Open WebUI dependency). Nothing to `pip install` into the image, no schema
  changes — it survives Open WebUI upgrades untouched. Its only network call is
  the background registry fetch (one small HTTPS GET every 12 h, which you can
  turn off).

## Install (2 minutes)

1. In Open WebUI: **Admin Panel → Functions → ➕ (New Function)**.
2. Paste the entire contents of [`eco_impact_filter.py`](eco_impact_filter.py)
   and **Save**.
3. Toggle the function **enabled**, then open its **⋮ menu → Global** so it
   applies to every model.

That's it — send a message and you'll see the status line. Out of the box it
uses a small **embedded registry** covering common model families, so it works
before you do anything else.

## The model registry

The filter estimates from a model's parameter counts, which it reads from a JSON
registry. Out of the box it uses a small **embedded registry** covering common
model families, so it works before you do anything else.

### Weekly registry (CUNY AI Lab default)

A GitHub Action in this repo ([`weekly-registry.yml`](.github/workflows/weekly-registry.yml))
rebuilds the registry every Monday and publishes it to the
[`registry` branch](../../tree/registry). The filter fetches it from there every
12 hours in the background and caches the last good copy at `registry_path`, so
**models added to CAIL Gateway pick up parameters without anyone touching Open
WebUI**. The job:

1. reads every model CAIL Gateway serves from its public `GET /v1/catalog`;
2. resolves each model's total and active parameters, in this order:
   published numbers from the curated list or EcoLogits (exact name only), an
   `NNNb-aNb` size in the model name, the model's Hugging Face config (MoE
   active parameters are computed from the expert counts and shown as a range),
   and only then an older *estimated* guess;
3. writes exact-id entries with the electricity zone of the serving provider
   (US grid for Workers AI and Bedrock Mantle);
4. checks the result against last week's (no served model may lose coverage, no
   active-parameter count may jump more than 3×, and EcoLogits' methodology must
   still match the math vendored here) and publishes only if it passes;
5. keeps one open issue labelled `eco-registry` listing models it could not
   resolve, with a stub to fill in.

Those unresolved models keep working with the flagged generic estimate. To fix
one, add an entry to [`scripts/eco-extra-patterns.json`](scripts/eco-extra-patterns.json)
on `main` (use `"exact_only": true` for ids too generic for substring matching)
and re-run the workflow from the Actions tab. If Hugging Face repos you serve are
gated, add an `HF_TOKEN` repository secret so the job can read their configs.

### Your own instance

Point the filter's `registry_url` valve at your own copy, or set it empty and
build a file locally:

```bash
# From an OpenAI-compatible catalog shaped like CAIL Gateway's /v1/catalog:
python3 scripts/sync-eco-models.py --gateway-url https://gateway.example.com -o eco_models.json
# Or a coverage report against an Open WebUI instance (admin API key):
export OWUI_API_KEY=<your key>
python3 scripts/sync-eco-models.py --owui-url https://your-openwebui.example.com -o eco_models.json
```

Then copy `eco_models.json` to the filter's `registry_path` inside your
container (see [`scripts/install-registry.example.sh`](scripts/install-registry.example.sh));
the filter re-reads it whenever the file changes. A ready-to-use example is in
[`examples/eco_models.json`](examples/eco_models.json).

## Configuration (valves)

Set in the function's settings in the Admin UI:

| Valve | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master on/off. |
| `registry_url` | this repo's `registry` branch | Where to fetch the registry. Empty = use `registry_path` only. |
| `registry_refresh_hours` | `12` | How often to re-fetch `registry_url` (failed fetches retry after 1 h). |
| `registry_path` | `/app/backend/data/eco_models.json` | Local registry file; also caches the last good fetch across restarts. |
| `show_energy` | `true` | Show Wh alongside CO₂e. |
| `show_comparison` | `true` | Show a real-world equivalence (breaths / walking / driving). |
| `debug_logging` | `false` | Log calculations to the server console. |

Per-user: each user can hide the line via their own `enabled` UserValve.

Model cards (custom models built on a base model) are estimated from their base
model. Speech-to-text models get no status line.

## How the estimate is computed

Vendored directly from EcoLogits (see [Attribution](#attribution--license)):
energy per request is derived from the model's **active** parameter count and
output token count through a GPU-energy regression, plus server overhead and
data-center PUE; CO₂e is that energy times the **electricity-mix** carbon
intensity of the zone the registry assigns the model (the serving provider's
zone for Gateway models; world average by default; legacy direct-provider ids
can be mapped by prefix, e.g. AWS Bedrock `us.*` ids → US grid). Embodied
(hardware-manufacturing) impact is amortized in as well. For models whose exact
architecture isn't public, EcoLogits provides estimated parameter ranges, which
is why closed models show a wider min–max band.

These are **estimates for awareness**, not audited measurements. Treat the
ranges as order-of-magnitude guidance.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install pydantic pytest
.venv/bin/python -m pytest -v
```

The tests include golden values computed with the real EcoLogits library, so
the vendored math is verified to reproduce upstream outputs exactly.

## Attribution & license

This project is a thin, operator-facing wrapper around the work of
**[EcoLogits](https://ecologits.ai)** by
**[GenAI Impact](https://genai-impact.org)**
([mlco2/ecologits](https://github.com/mlco2/ecologits)). The
environmental-impact methodology, its coefficients, and the model/electricity
data all come from them. If you use these estimates in research, **cite
EcoLogits**, not this repo.

Because EcoLogits is licensed under the **Mozilla Public License 2.0
(MPL-2.0)** and this project vendors its methodology and redistributes its data,
**this project is also released under MPL-2.0** (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE) for exactly what is derived from where).

Original contributions here — the Open WebUI Filter integration, the status-line
UX, the model-matching/registry/fallback logic, and the sync tooling — are by
the **CUNY AI Lab** and are likewise MPL-2.0.

Not affiliated with or endorsed by EcoLogits/GenAI Impact or Open WebUI.
