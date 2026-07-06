#!/usr/bin/env python3
"""
sync-eco-models.py — build eco_models.json for the Eco Impact Estimator filter.

Fetches EcoLogits' model-parameter registry and electricity mixes from GitHub,
fetches the live Open WebUI model list, matches them, prints a coverage report
(flagging served models with no registry entry), and writes the merged registry.

Usage:
    # Full run against dev (needs an OWUI admin API key):
    export OWUI_API_KEY=<key>
    python3 scripts/sync-eco-models.py --owui-url https://your-openwebui.example.com -o eco_models.json

    # Offline / testing:
    python3 scripts/sync-eco-models.py --ecologits-dir /path/to/ecologits-clone \
        --models-json models_fixture.json -o /tmp/eco_models.json

Then push to EFS:  ./scripts/push-eco-models.sh dev eco_models.json

Manual additions for unmatched models go in scripts/eco-extra-patterns.json
(same entry shape as "patterns" in the output) — they are merged on every run.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

PINNED_REF = "42154236c3b275346e8b97b04f49cd19877316e6"
RAW_URL = "https://raw.githubusercontent.com/genai-impact/ecologits/{ref}/ecologits/data/{name}"

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
    hdrs = {"User-Agent": "eco-models-sync/1.0"}
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owui-url", help="Open WebUI base URL for the live model list")
    ap.add_argument("--api-key-env", default="OWUI_API_KEY",
                    help="Env var holding the OWUI API key (default OWUI_API_KEY)")
    ap.add_argument("--models-json", help="Offline: file with OWUI /api/models JSON "
                    "or a plain JSON list of model ids")
    ap.add_argument("--ecologits-dir", help="Offline: local ecologits clone directory")
    ap.add_argument("--ref", default=PINNED_REF, help="ecologits git ref (default: pinned)")
    ap.add_argument("--extra", default="scripts/eco-extra-patterns.json",
                    help="Manual pattern additions (merged last, win collisions)")
    ap.add_argument("-o", "--output", default="eco_models.json")
    args = ap.parse_args()

    # 1. EcoLogits data
    if args.ecologits_dir:
        d = args.ecologits_dir
        eco_models = json.load(open(os.path.join(d, "ecologits/data/models.json")))
        mixes = json.load(open(os.path.join(d, "ecologits/data/electricity_mixes.json")))
        source = f"genai-impact/ecologits@local({d})"
    else:
        eco_models = http_json(RAW_URL.format(ref=args.ref, name="models.json"))
        mixes = http_json(RAW_URL.format(ref=args.ref, name="electricity_mixes.json"))
        source = f"genai-impact/ecologits@{args.ref}"
    print(f"EcoLogits: {len(eco_models['models'])} models, "
          f"{len(mixes['electricity_mixes'])} electricity zones ({source})")

    patterns = build_patterns(eco_models)

    # 2. Manual extras (win collisions)
    if args.extra and os.path.exists(args.extra):
        extras = json.load(open(args.extra))
        for e in extras:
            e["match"] = normalize(e["match"])
            patterns[e["match"]] = e
        if extras:
            print(f"Merged {len(extras)} manual patterns from {args.extra}")

    registry = {
        "version": 1,
        "source": source,
        "generated": time.strftime("%Y-%m-%d"),
        "defaults": DEFAULTS,
        "zones": {z["name"]: {k: z[k] for k in ("gwp", "adpe", "pe", "wue")}
                  for z in mixes["electricity_mixes"]},
        "zone_prefixes": ZONE_PREFIXES,
        "patterns": sorted(patterns.values(), key=lambda e: e["match"]),
    }
    with open(args.output, "w") as fh:
        json.dump(registry, fh, indent=1)
    print(f"Wrote {args.output}: {len(registry['patterns'])} patterns, "
          f"{len(registry['zones'])} zones")

    # 3. Coverage report against the live/offline OWUI model list
    model_ids = None
    if args.models_json:
        raw = json.load(open(args.models_json))
        model_ids = raw if isinstance(raw, list) else [m["id"] for m in raw.get("data", [])]
    elif args.owui_url:
        key = os.environ.get(args.api_key_env)
        if not key:
            sys.exit(f"error: {args.api_key_env} not set (needed for --owui-url)")
        data = http_json(args.owui_url.rstrip("/") + "/api/models",
                         headers={"Authorization": f"Bearer {key}"})
        model_ids = [m["id"] for m in data.get("data", [])]

    if model_ids is None:
        print("\nNo OWUI model list given (--owui-url or --models-json) — "
              "skipping coverage report.")
        return

    print(f"\nCoverage report ({len(model_ids)} models served):")
    unmatched = []
    for mid in sorted(model_ids):
        m = find_match(mid, patterns)
        if m:
            e = patterns[m]
            print(f"  ✓ {mid}  ->  {m}  "
                  f"[{e['total']['min']}-{e['total']['max']}B total, {e['confidence']}]")
        else:
            unmatched.append(mid)
            print(f"  ✗ {mid}  ->  UNMATCHED (will use generic estimate)")
    print(f"\nMatched {len(model_ids) - len(unmatched)}/{len(model_ids)}.")
    if unmatched:
        print(f"\nTo cover the {len(unmatched)} unmatched model(s), add entries to "
              f"{args.extra} and re-run. Stub:")
        stub = [{"match": normalize(u), "total": {"min": 8, "max": 440},
                 "active": {"min": 8, "max": 110}, "confidence": "estimated"}
                for u in unmatched]
        print(json.dumps(stub, indent=1))
        print("Tip: shorten each \"match\" to a family prefix (e.g. \"nova-micro\") so future variants match too, and replace the placeholder parameter ranges with real values from the model card.")


if __name__ == "__main__":
    main()
