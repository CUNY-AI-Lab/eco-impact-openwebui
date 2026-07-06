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
  changes — it survives Open WebUI upgrades untouched.

## Install (2 minutes)

1. In Open WebUI: **Admin Panel → Functions → ➕ (New Function)**.
2. Paste the entire contents of [`eco_impact_filter.py`](eco_impact_filter.py)
   and **Save**.
3. Toggle the function **enabled**, then open its **⋮ menu → Global** so it
   applies to every model.

That's it — send a message and you'll see the status line. Out of the box it
uses a small **embedded registry** covering common model families, so it works
before you do anything else.

## Sharpen the numbers: the model registry

The filter estimates from a model's parameter counts. It reads them from a JSON
registry file inside the container (default
`/app/backend/data/eco_models.json`, which persists across restarts) and
**auto-reloads it whenever the file changes** — no restart required. If the file
is missing, the embedded fallback is used.

Build a registry tailored to *your* served models with the included tool:

```bash
# Point it at your instance; needs an Open WebUI API key with model-list access.
export OWUI_API_KEY=<your key>
python3 scripts/sync-eco-models.py --owui-url https://your-openwebui.example.com -o eco_models.json
```

`sync-eco-models.py`:

1. downloads EcoLogits' current model-parameter and electricity-mix data,
2. fetches your instance's live model list,
3. matches them and prints a **coverage report**, explicitly flagging any served
   model it couldn't match (with a ready-to-fill stub), and
4. merges your manual additions from
   [`scripts/eco-extra-patterns.json`](scripts/eco-extra-patterns.json) — where
   you can add parameter counts for models EcoLogits doesn't cover (this repo
   ships a starter set of ~180 curated entries: Qwen, DeepSeek, GLM, Kimi,
   Llama, Nova, Gemma, and more).

Then copy `eco_models.json` to the filter's `registry_path` inside your
container. How you copy a file in depends on your deployment (docker cp,
kubectl cp, a mounted volume, …); see
[`scripts/install-registry.example.sh`](scripts/install-registry.example.sh)
for the common shapes. **When you add new models to your instance, just re-run
the sync, review the coverage report, and copy the file over.** Unmatched models
keep working via the flagged generic estimate, so nothing breaks if you forget —
the sync just tightens the numbers.

A ready-to-use example registry is in
[`examples/eco_models.json`](examples/eco_models.json).

## Configuration (valves)

Set in the function's settings in the Admin UI:

| Valve | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master on/off. |
| `registry_path` | `/app/backend/data/eco_models.json` | Where to read the model registry. |
| `show_energy` | `true` | Show Wh alongside CO₂e. |
| `show_comparison` | `true` | Show a real-world equivalence (breaths / walking / driving). |
| `debug_logging` | `false` | Log calculations to the server console. |

Per-user: each user can hide the line via their own `enabled` UserValve.

## How the estimate is computed

Vendored directly from EcoLogits (see [Attribution](#attribution--license)):
energy per request is derived from the model's **active** parameter count and
output token count through a GPU-energy regression, plus server overhead and
data-center PUE; CO₂e is that energy times the **electricity-mix** carbon
intensity (world average by default; the registry can map provider id prefixes
to region mixes, e.g. AWS Bedrock `us.*` ids → US grid). Embodied
(hardware-manufacturing) impact is amortized in as well. For models whose exact
architecture isn't public, EcoLogits provides estimated parameter ranges, which
is why closed models show a wider min–max band.

These are **estimates for awareness**, not audited measurements. Treat the
ranges as order-of-magnitude guidance.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install pydantic pytest
.venv/bin/python -m pytest test_eco_impact_filter.py -v
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
