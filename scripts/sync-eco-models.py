#!/usr/bin/env python3
"""
sync-eco-models.py — build eco_models.json for the Eco Impact Estimator filter.

Fetches EcoLogits' model-parameter registry and electricity mixes from GitHub,
builds per-model entries for every model CAIL Gateway serves (exact ids, with
parameters from curated data, EcoLogits, model-name hints, or the model's
Hugging Face config), validates the result against last week's registry, and
writes the merged registry plus a Markdown report.

Usage:
    # What the weekly GitHub Action runs (public catalog, no key needed):
    python3 scripts/sync-eco-models.py --gateway-url https://tools.ailab.gc.cuny.edu \
        --previous registry/eco_models.json -o registry/eco_models.json --report report.md

    # Legacy: coverage report against an Open WebUI instance (needs an admin key):
    export OWUI_API_KEY=<key>
    python3 scripts/sync-eco-models.py --owui-url https://your-openwebui.example.com -o eco_models.json

    # Offline / testing:
    python3 scripts/sync-eco-models.py --ecologits-dir /path/to/ecologits-clone \
        --models-json models_fixture.json -o /tmp/eco_models.json

Manual additions for unresolved models go in scripts/eco-extra-patterns.json
(same entry shape as "patterns" in the output) — they are merged on every run
and win over every automatic source.

Exit codes: 0 = registry written and passed validation; 3 = registry written but
validation flagged changes that need a human (the workflow opens a PR instead of
committing); anything else = failure.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

PINNED_REF = "42154236c3b275346e8b97b04f49cd19877316e6"
RAW_URL = "https://raw.githubusercontent.com/mlco2/ecologits/{ref}/ecologits/data/{name}"
GH_API = "https://api.github.com/repos/mlco2/ecologits"
# Git blob sha of ecologits/impacts/llm.py that eco_impact_filter.py vendors. If a
# newer EcoLogits release changes that file, the vendored math is stale: keep the
# old data ref and flag it for a human instead of mixing new data with old math.
VENDORED_LLM_BLOB = "aef8473e3cab495999b1ed3c8489be30c378effa"

EXIT_NEEDS_REVIEW = 3
# A weekly change in a model's max active parameters beyond this factor is held
# for review rather than published.
MAX_ACTIVE_CHANGE = 3.0
# Hold the build if fewer than this share of last run's covered ids are still in
# the catalog (an empty or truncated catalog response looks like mass retirement).
MIN_CATALOG_KEPT = 0.5

# Electricity zone per Gateway provider. Mantle runs in AWS us-east-1; Workers AI
# serves from the nearest Cloudflare GPU location, which for CUNY users is the US.
PROVIDER_ZONES = {"bedrock-mantle": "USA", "workers-ai": "USA", "openrouter": "USA"}

# Catalog tasks that are not text generation get a "skip" entry (no status line).
SKIP_TASKS = {"automatic speech recognition", "text embeddings", "translation",
              "text-to-image", "text to speech", "image classification"}

# Trailing name tokens that do not change a model's weights.
VARIANT_SUFFIXES = ("instruct", "it", "chat", "fp8", "fast", "lora", "hf", "awq",
                    "int4", "int8", "bf16", "fp16", "preview", "latest")

# Providers earlier in this list win pattern-name collisions.
PROVIDER_PRIORITY = ["openai", "anthropic", "google_genai", "mistralai", "cohere",
                     "huggingface_hub"]

DEFAULTS = {
    "zone": "WOR",
    "pue": {"min": 1.09, "max": 1.2},
    "unknown_params": {"total": {"min": 8, "max": 440}, "active": {"min": 8, "max": 110}},
}

ZONE_PREFIXES = [
    {"prefix": "us.", "zone": "USA"},
    {"prefix": "amazon.", "zone": "USA"},
    {"prefix": "anthropic.", "zone": "USA"},
    {"prefix": "meta.", "zone": "USA"},
    {"prefix": "mistral.", "zone": "USA"},
]


def http_json(url, headers=None):
    # Default urllib UA (Python-urllib/x.y) gets 403'd by the WAF in front of OWUI.
    hdrs = {"User-Agent": "eco-models-sync/1.1"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def normalize(s):
    return s.lower().strip().replace(".", "-")


def pattern_for(name):
    """Derive a match pattern from an EcoLogits model name.

    HF names like 'meta-llama/Meta-Llama-3.1-8B-Instruct' become
    'llama-3-1-8b-instruct' so they substring-match OpenRouter/Bedrock ids.
    """
    base = normalize(name.split("/")[-1])
    if base.startswith("meta-llama"):
        base = base[len("meta-"):]
    return base


def norm_minmax(v):
    if isinstance(v, (int, float)):
        return {"min": float(v), "max": float(v)}
    if isinstance(v, dict) and "min" in v and "max" in v:
        return {"min": float(v["min"]), "max": float(v["max"])}
    raise ValueError(f"unsupported parameter value: {v!r}")


def norm_params(arch):
    """Normalize EcoLogits' three architecture shapes to (total, active) min/max."""
    p = arch.get("parameters")
    if arch.get("type") == "moe":
        if isinstance(p, dict) and "total" in p:
            total, active = p["total"], p["active"]
        else:
            total, active = p, arch.get("active_parameters", p)
    else:
        total = active = p
    return norm_minmax(total), norm_minmax(active)


def build_patterns(eco_models):
    by_name = {}
    patterns = {}
    prio = {p: i for i, p in enumerate(PROVIDER_PRIORITY)}
    ordered = sorted(eco_models["models"], key=lambda m: prio.get(m["provider"], 99))
    skipped = 0
    for m in ordered:
        try:
            total, active = norm_params(m["architecture"])
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        pat = pattern_for(m["name"])
        entry = {
            "match": pat,
            "total": total,
            "active": active,
            "confidence": "published" if not m.get("warnings") else "estimated",
        }
        dep = m.get("deployment") or {}
        if dep.get("tps"):
            entry["tps"] = dep["tps"]
        if dep.get("ttft"):
            entry["ttft"] = dep["ttft"]
        by_name[(m["provider"], m["name"])] = entry
        if pat not in patterns:
            patterns[pat] = entry
    for a in eco_models.get("aliases", []):
        target = by_name.get((a["provider"], a["alias"]))
        if target is None:
            continue
        pat = pattern_for(a["name"])
        if pat not in patterns:
            patterns[pat] = {**target, "match": pat}
    if skipped:
        print(f"note: skipped {skipped} EcoLogits entries with unparseable architecture")
    return patterns


def find_match(model_id, patterns):
    norm = normalize(model_id)
    best = None
    for pat in patterns:
        if pat and pat in norm and (best is None or len(pat) > len(best)):
            best = pat
    return best


# ============================================================
# EcoLogits ref selection
# ============================================================

def _gh_headers():
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def choose_ecologits_ref(requested, fetch=http_json, previous_ref=None):
    """Resolve 'latest' to EcoLogits' newest release tag, unless that release
    changed the impact methodology the filter vendors; then keep the ref the
    previous registry used (or the pin). Returns (ref, warning)."""
    if requested != "latest":
        return requested, None
    tag = fetch(f"{GH_API}/releases/latest", _gh_headers())["tag_name"]
    blob = fetch(f"{GH_API}/contents/ecologits/impacts/llm.py?ref={tag}",
                 _gh_headers())["sha"]
    if blob != VENDORED_LLM_BLOB:
        keep = previous_ref or PINNED_REF
        return keep, (
            f"EcoLogits {tag} changed ecologits/impacts/llm.py (blob {blob[:10]}, "
            f"vendored {VENDORED_LLM_BLOB[:10]}). Kept data at {keep}; "
            "re-vendor the math in eco_impact_filter.py, update VENDORED_LLM_BLOB, "
            "and refresh the golden tests.")
    return tag, None


# ============================================================
# Gateway rows -> exact-id entries
# ============================================================

HINT_RE = re.compile(r"(?:^|[-_/])(\d+(?:\.\d+)?)b-a(\d+(?:\.\d+)?)b(?:$|[-_])")
SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmbt])\s*$", re.I)
HF_RE = re.compile(r"huggingface\.co/(?:api/models/)?([^/]+/[^/]+)/(?:blob|tree|resolve|revision)/([0-9a-f]{7,40})")


def size_label_b(label):
    """'358B' -> 358.0, '1T' -> 1000.0 (billions); None if unparseable."""
    m = SIZE_RE.match(label or "")
    if not m:
        return None
    scale = {"k": 1e-6, "m": 1e-3, "b": 1.0, "t": 1000.0}[m.group(2).lower()]
    return float(m.group(1)) * scale


def row_names(row):
    """Names a catalog row is known by, lowercase, without vendor prefixes."""
    names = [row.get("id"), row.get("upstream_model"), row.get("model_group")]
    spec = row.get("specifications") or {}
    for src in ((spec.get("weights") or {}).get("source_url"),
                (spec.get("architecture") or {}).get("source_url")):
        m = HF_RE.search(src or "")
        if m:
            names.append(m.group(1))
    out = []
    for n in names:
        if not n:
            continue
        base = n.lower().strip().split("/")[-1]
        if base not in out:
            out.append(base)
    return out


def variant_names(name):
    """normalize(name) and successively shorter forms with variant suffixes
    removed: 'llama-3.1-8b-instruct-fp8' -> llama-3-1-8b-instruct-fp8,
    llama-3-1-8b-instruct, llama-3-1-8b."""
    n = normalize(name)
    out = [n]
    changed = True
    while changed:
        changed = False
        for suf in VARIANT_SUFFIXES:
            if n.endswith("-" + suf):
                n = n[: -len(suf) - 1]
                out.append(n)
                changed = True
    return out


def hf_ref(row):
    spec = row.get("specifications") or {}
    for src in ((spec.get("architecture") or {}).get("source_url"),
                (spec.get("weights") or {}).get("source_url"),
                (spec.get("size") or {}).get("source_url")):
        m = HF_RE.search(src or "")
        if m:
            return m.group(1), m.group(2)
    return None


def fetch_hf(repo, rev, token=None, fetch=http_json):
    """(total_params, config) from Hugging Face, or None when unavailable
    (gated without a token, missing safetensors metadata, network error)."""
    hdrs = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        meta = fetch(f"https://huggingface.co/api/models/{repo}/revision/{rev}", hdrs)
        cfg = fetch(f"https://huggingface.co/{repo}/resolve/{rev}/config.json", hdrs)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    total = (meta.get("safetensors") or {}).get("total")
    if not total:
        return None
    return float(total), cfg


EXPERT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts")
TOPK_KEYS = ("num_experts_per_tok", "experts_per_token", "num_experts_per_token",
             "top_k_experts", "moe_topk")


def active_from_config(total, cfg):
    """Active parameter count from an HF config.

    Returns (active, "dense"), (active, "moe"), or (None, reason) when the config
    looks MoE but can't be parsed. Inactive weight = routed experts not selected
    per token: 3 * hidden * expert_intermediate per expert per MoE layer. This
    ignores shared experts' exact size and MTP layers, so it over-counts active
    by roughly 5-25%; callers widen it into a range."""
    t = cfg.get("text_config") or cfg
    E = next((t[k] for k in EXPERT_KEYS if t.get(k)), None)
    k = next((t[k] for k in TOPK_KEYS if t.get(k)), None)
    if not E:
        if any("expert" in key or key.startswith("moe") for key in t):
            return None, "config mentions experts but no expert count"
        return total, "dense"
    if not k:
        return None, "MoE config without experts-per-token"
    H = t.get("hidden_size")
    inter = t.get("moe_intermediate_size") or t.get("expert_intermediate_size")
    if not inter and "intermediate_size" in t and not t.get("moe_intermediate_size"):
        inter = t.get("intermediate_size")
    L = t.get("num_hidden_layers")
    if not (H and inter and L):
        return None, "MoE config missing hidden/intermediate/layer sizes"
    moe_layers = L - int(t.get("first_k_dense_replace") or 0)
    moe_layers -= len(t.get("mlp_only_layers") or [])
    step = int(t.get("decoder_sparse_step") or 1)
    if step > 1:
        moe_layers = moe_layers // step
    inactive = moe_layers * (E - k) * 3 * H * inter
    active = total - inactive
    if active <= 0 or active > total:
        return None, f"derived active {active / 1e9:.1f}B out of range"
    return active, "moe"


def mm(lo, hi=None):
    hi = lo if hi is None else hi
    return {"min": round(float(lo), 3), "max": round(float(hi), 3)}


def resolve_row(row, extras, eco_patterns, hf_lookup):
    """(entry, None) or (None, reason) for one Gateway catalog row.

    Order: published curated/EcoLogits exact name, 'NNNb-aNb' name hint,
    Hugging Face config, then an "estimated" curated/EcoLogits exact name. Never
    substring/prefix matching: a new version must not inherit an older model's
    numbers."""
    names = row_names(row)

    def exact(confidence_ok):
        for source, table in (("curated", extras), ("ecologits", eco_patterns)):
            for n in names:
                for v in variant_names(n):
                    if v in table and confidence_ok(table[v].get("confidence")):
                        e = {k: table[v][k] for k in
                             ("total", "active", "confidence", "tps", "ttft") if k in table[v]}
                        e["source"] = f"{source}:{v}"
                        return e
        return None

    # Published numbers first; an "estimated" guess only when nothing better exists
    # (estimates were often written before a model's weights were released).
    e = exact(lambda c: c != "estimated")
    if e:
        return e, None

    spec = row.get("specifications") or {}
    ref = hf_ref(row)
    hf = hf_lookup(*ref) if ref else None

    for n in names:
        m = HINT_RE.search(n)
        if m:
            total = (hf[0] / 1e9) if hf else (
                size_label_b((spec.get("size") or {}).get("label")) or float(m.group(1)))
            return {"total": mm(total), "active": mm(float(m.group(2))),
                    "confidence": "published", "source": f"name:{n}"}, None

    why = "no curated entry, no size hint in name, no Hugging Face link"
    if ref is not None and hf is None:
        why = f"Hugging Face {ref[0]} unavailable (gated or no safetensors metadata)"
    elif hf is not None:
        total, cfg = hf
        active, kind = active_from_config(total, cfg)
        if active is not None:
            tb, ab = total / 1e9, active / 1e9
            return {"total": mm(tb),
                    "active": mm(ab) if kind == "dense" else mm(0.8 * ab, ab),
                    "confidence": "derived",
                    "source": f"hf:{ref[0]}@{ref[1][:7]}"}, None
        why = f"Hugging Face {ref[0]}: {kind}"
    e = exact(lambda c: c == "estimated")
    if e:
        return e, None
    return None, why


def is_skip(row):
    return (row.get("task") or "").strip().lower() in SKIP_TASKS


def build_ids(catalog_rows, extras, eco_patterns, hf_lookup):
    """Exact-id map for every Gateway row, plus (resolved, unresolved) reports."""
    ids, resolved, unresolved = {}, [], []
    for row in sorted(catalog_rows, key=lambda r: r["id"]):
        keys = [k.lower() for k in (row.get("id"), row.get("upstream_model"),
                                    row.get("model_group")) if k]
        if is_skip(row):
            entry, why = {"skip": True, "source": f"task:{row.get('task')}"}, None
        else:
            entry, why = resolve_row(row, extras, eco_patterns, hf_lookup)
        if entry is None:
            unresolved.append({"id": row["id"], "keys": keys, "provider": row.get("provider"),
                               "reason": why, "hf": hf_ref(row),
                               "size": ((row.get("specifications") or {}).get("size") or {}).get("label")})
            continue
        if not entry.get("skip"):
            entry["zone"] = PROVIDER_ZONES.get(row.get("provider"), "WOR")
            entry["provider"] = row.get("provider")
            resolved.append((row["id"], entry))
        for k in keys:
            ids.setdefault(k, entry)
    return ids, resolved, unresolved


# ============================================================
# Validation gate
# ============================================================

def validate(registry, previous, catalog_ids):
    """Problems that should stop an unattended publish. Returns a list of str."""
    problems, invalid = [], set()
    for key, e in registry.get("ids", {}).items():
        if e.get("skip"):
            continue
        try:
            t, a = e["total"], e["active"]
            assert 0 < t["min"] <= t["max"] and 0 < a["min"] <= a["max"]
            assert a["max"] <= t["max"] * 1.001
            assert e.get("zone") in registry["zones"]
        except (AssertionError, KeyError, TypeError):
            problems.append(f"`{key}`: invalid entry {json.dumps(e)}")
            invalid.add(key)
    if not previous:
        return problems
    old_ids = {k: v for k, v in (previous.get("ids") or {}).items() if not v.get("skip")}
    live = {c.lower() for c in catalog_ids}
    # An empty or truncated catalog response must not wipe last week's coverage.
    if old_ids and len(live & set(old_ids)) < MIN_CATALOG_KEPT * len(old_ids):
        problems.append(f"catalog shrank: only {len(live & set(old_ids))} of "
                        f"{len(old_ids)} previously covered ids are still served")
    for key, old in old_ids.items():
        if key not in live or key in invalid:
            continue
        new = registry["ids"].get(key)
        if new is None:
            problems.append(f"`{key}` was covered last run but is now unresolved")
            continue
        if new.get("skip"):
            continue
        ratio = new["active"]["max"] / max(old["active"]["max"], 1e-9)
        if ratio > MAX_ACTIVE_CHANGE or ratio < 1 / MAX_ACTIVE_CHANGE:
            problems.append(f"`{key}` active params moved {old['active']['max']}B -> "
                            f"{new['active']['max']}B ({old.get('source')} -> {new.get('source')})")
    return problems


def same_content(a, b):
    strip = lambda r: {k: v for k, v in (r or {}).items() if k != "generated"}
    return strip(a) == strip(b)


def render_report(resolved, unresolved, problems, warning, source, extras_path):
    lines = ["# Eco registry weekly sync", "",
             f"EcoLogits data: `{source}`", ""]
    by_conf = {}
    for _, e in resolved:
        by_conf[e["confidence"]] = by_conf.get(e["confidence"], 0) + 1
    lines.append(f"Resolved {len(resolved)} Gateway models: " + ", ".join(
        f"{n} {c}" for c, n in sorted(by_conf.items())) + ".")
    if warning:
        lines += ["", "## ⚠️ Methodology changed upstream", "", warning]
    if problems:
        lines += ["", "## Held for review", "",
                  "The registry was not published automatically because:", ""]
        lines += [f"- {p}" for p in problems]
    if unresolved:
        lines += ["", f"## Unresolved ({len(unresolved)})", "",
                  "These models show the flagged generic estimate until they get an "
                  f"entry in `{extras_path}`.", "",
                  "| Model | Provider | Size | Why |", "|---|---|---|---|"]
        for u in unresolved:
            hf = f"[{u['hf'][0]}](https://huggingface.co/{u['hf'][0]})" if u["hf"] else ""
            lines.append(f"| `{u['id']}` | {u['provider']} | {u['size'] or ''} | "
                         f"{u['reason']} {hf} |")
        stub = [{"match": normalize(u["id"]), "total": {"min": 0, "max": 0},
                 "active": {"min": 0, "max": 0}, "confidence": "published",
                 "note": "fill from the model card"} for u in unresolved]
        lines += ["", "Stub to fill in and add:", "", "```json",
                  json.dumps(stub, indent=1), "```"]
    else:
        lines += ["", "No unresolved models."]
    return "\n".join(lines) + "\n"


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway-url", help="CAIL Gateway base URL; reads its public "
                    "/v1/catalog and writes exact-id entries for every served model")
    ap.add_argument("--catalog-json", help="Offline: a saved /v1/catalog response")
    ap.add_argument("--previous", help="Last published registry, for the validation gate")
    ap.add_argument("--report", help="Write a Markdown report (used as the issue body)")
    ap.add_argument("--hf-token-env", default="HF_TOKEN",
                    help="Env var with a Hugging Face token for gated repos (optional)")
    ap.add_argument("--no-hf", action="store_true", help="Don't call Hugging Face")
    ap.add_argument("--owui-url", help="Legacy: Open WebUI base URL for a coverage report")
    ap.add_argument("--api-key-env", default="OWUI_API_KEY",
                    help="Env var holding the OWUI API key (default OWUI_API_KEY)")
    ap.add_argument("--models-json", help="Offline: file with OWUI /api/models JSON "
                    "or a plain JSON list of model ids")
    ap.add_argument("--ecologits-dir", help="Offline: local ecologits clone directory")
    ap.add_argument("--ref", default="latest",
                    help="ecologits git ref, or 'latest' release (default) guarded by "
                    "the vendored-methodology check")
    ap.add_argument("--extra", default="scripts/eco-extra-patterns.json",
                    help="Manual pattern additions (merged last, win collisions)")
    ap.add_argument("-o", "--output", default="eco_models.json")
    args = ap.parse_args()

    previous = None
    if args.previous and os.path.exists(args.previous):
        previous = load_json(args.previous)

    # 1. EcoLogits data
    warning = None
    if args.ecologits_dir:
        d = args.ecologits_dir
        eco_models = load_json(os.path.join(d, "ecologits/data/models.json"))
        mixes = load_json(os.path.join(d, "ecologits/data/electricity_mixes.json"))
        source = f"mlco2/ecologits@local({d})"
    else:
        prev_src = (previous or {}).get("source", "")
        prev_ref = prev_src.split("@", 1)[1] if prev_src.startswith(
            "mlco2/ecologits@") and "(" not in prev_src else None
        ref, warning = choose_ecologits_ref(args.ref, previous_ref=prev_ref)
        eco_models = http_json(RAW_URL.format(ref=ref, name="models.json"))
        mixes = http_json(RAW_URL.format(ref=ref, name="electricity_mixes.json"))
        source = f"mlco2/ecologits@{ref}"
    print(f"EcoLogits: {len(eco_models['models'])} models, "
          f"{len(mixes['electricity_mixes'])} electricity zones ({source})")
    if warning:
        print(f"WARNING: {warning}")

    eco_patterns = build_patterns(eco_models)
    patterns = dict(eco_patterns)

    # 2. Manual extras (win collisions)
    extras = {}
    if args.extra and os.path.exists(args.extra):
        for e in load_json(args.extra):
            e["match"] = normalize(e["match"])
            extras[e["match"]] = e
        # exact_only entries are too generic for the legacy substring matcher.
        patterns.update({k: {f: v for f, v in e.items() if f != "exact_only"}
                         for k, e in extras.items() if not e.get("exact_only")})
        print(f"Merged {len(extras)} manual patterns from {args.extra}")

    registry = {
        "version": 2,
        "source": source,
        "generated": time.strftime("%Y-%m-%d"),
        "defaults": DEFAULTS,
        "zones": {z["name"]: {k: z[k] for k in ("gwp", "adpe", "pe", "wue")}
                  for z in mixes["electricity_mixes"]},
        "zone_prefixes": ZONE_PREFIXES,
        "provider_zones": PROVIDER_ZONES,
        "ids": {},
        "unresolved_ids": [],
        "patterns": sorted(patterns.values(), key=lambda e: e["match"]),
    }

    # 3. Gateway catalog -> exact ids
    rows = None
    if args.catalog_json:
        rows = load_json(args.catalog_json)["data"]
    elif args.gateway_url:
        rows = http_json(args.gateway_url.rstrip("/") + "/v1/catalog")["data"]
    problems, resolved, unresolved = [], [], []
    if rows is not None:
        token = os.environ.get(args.hf_token_env)
        cache = {}

        def hf_lookup(repo, rev):
            if args.no_hf:
                return None
            if (repo, rev) not in cache:
                cache[(repo, rev)] = fetch_hf(repo, rev, token)
            return cache[(repo, rev)]

        ids, resolved, unresolved = build_ids(rows, extras, eco_patterns, hf_lookup)
        registry["ids"] = dict(sorted(ids.items()))
        registry["unresolved_ids"] = sorted({k for u in unresolved for k in u["keys"]}
                                            - set(ids))
        problems = validate(registry, previous, [k for r in rows for k in
                                                (r.get("id"), r.get("upstream_model"),
                                                 r.get("model_group")) if k])
        print(f"\nGateway catalog: {len(rows)} rows, {len(resolved)} resolved, "
              f"{len(unresolved)} unresolved, "
              f"{sum(1 for e in ids.values() if e.get('skip'))} skip keys")
        for mid, e in resolved:
            print(f"  ✓ {mid:40} {e['active']['min']}-{e['active']['max']}B active / "
                  f"{e['total']['max']}B total  [{e['confidence']}, {e['source']}]")
        for u in unresolved:
            print(f"  ✗ {u['id']:40} {u['reason']}")
        for p in problems:
            print(f"  ! {p}")

    if same_content(registry, previous):
        registry["generated"] = previous.get("generated", registry["generated"])
    with open(args.output, "w") as fh:
        json.dump(registry, fh, indent=1)
        fh.write("\n")
    print(f"Wrote {args.output}: {len(registry['ids'])} ids, "
          f"{len(registry['patterns'])} patterns, {len(registry['zones'])} zones")

    if args.report:
        with open(args.report, "w") as fh:
            fh.write(render_report(resolved, unresolved, problems, warning, source,
                                   args.extra))

    # 4. Legacy coverage report against an Open WebUI model list
    model_ids = None
    if args.models_json:
        raw = load_json(args.models_json)
        model_ids = raw if isinstance(raw, list) else [m["id"] for m in raw.get("data", [])]
    elif args.owui_url:
        key = os.environ.get(args.api_key_env)
        if not key:
            sys.exit(f"error: {args.api_key_env} not set (needed for --owui-url)")
        data = http_json(args.owui_url.rstrip("/") + "/api/models",
                         headers={"Authorization": f"Bearer {key}"})
        model_ids = [m["id"] for m in data.get("data", [])]
    if model_ids is not None:
        print(f"\nOpen WebUI coverage ({len(model_ids)} models served):")
        unmatched = []
        for mid in sorted(model_ids):
            if mid.lower() in registry["ids"]:
                print(f"  ✓ {mid}  ->  exact id")
                continue
            m = find_match(mid, patterns)
            if m:
                print(f"  ~ {mid}  ->  pattern {m}")
            else:
                unmatched.append(mid)
                print(f"  ✗ {mid}  ->  UNMATCHED (will use generic estimate)")
        print(f"\nMatched {len(model_ids) - len(unmatched)}/{len(model_ids)}.")

    if problems:
        sys.exit(EXIT_NEEDS_REVIEW)


if __name__ == "__main__":
    main()
