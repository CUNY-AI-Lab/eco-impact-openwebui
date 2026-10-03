"""
title: Eco Impact Estimator
author: CUNY AI Lab
version: 1.1.0
license: MPL-2.0
description: Per-message energy and CO2e estimates for every model, shown as a status line. Methodology and constants vendored from EcoLogits (https://ecologits.ai).
"""

# This file is licensed under the Mozilla Public License 2.0 (MPL-2.0) because
# the impact math below is derived from EcoLogits (https://ecologits.ai),
# mlco2/ecologits, which is MPL-2.0. See LICENSE and NOTICE.
#
# Impact math vendored from mlco2/ecologits, commit
# 42154236c3b275346e8b97b04f49cd19877316e6, file ecologits/impacts/llm.py.
# Re-sync constants when refreshing the registry (scripts/sync-eco-models.py).

import asyncio
import json
import math
import os
import tempfile
import time
import urllib.request
from typing import Any, Optional

from pydantic import BaseModel, Field

MODEL_QUANTIZATION_BITS = 16

GPU_ENERGY_ALPHA = 1.1665273170451914e-06
GPU_ENERGY_BETA = -0.011205921025579175
GPU_ENERGY_GAMMA = 4.052928146734005e-05

LATENCY_ALPHA = 0.0006785088094353663
LATENCY_BETA = 0.0003119310311688259
LATENCY_GAMMA = 0.019473717579473387

GPU_MEMORY = 80  # GB
GPU_EMBODIED_IMPACT_GWP = 273  # kgCO2eq per GPU

SERVER_GPUS = 8
SERVER_POWER = 1.2  # kW
SERVER_EMBODIED_IMPACT_GWP = 5700  # kgCO2eq per server

HARDWARE_LIFESPAN = 3 * 365 * 24 * 60 * 60  # seconds
BATCH_SIZE = 64


def _impacts_single(
    active_params_b: float,
    total_params_b: float,
    output_tokens: float,
    mix_gwp: float,
    pue: float,
    tps: Optional[float] = None,
    ttft: Optional[float] = None,
    request_latency: float = math.inf,
) -> "tuple[float, float]":
    """One evaluation of the EcoLogits impact chain (use twice for min/max ranges).

    Returns (energy_kwh, gwp_kg) where gwp includes usage + embodied phases.
    """
    gpu_energy_per_token_kwh = (
        GPU_ENERGY_ALPHA * math.exp(GPU_ENERGY_BETA * BATCH_SIZE) * active_params_b
        + GPU_ENERGY_GAMMA
    ) / 1000
    gpu_energy_kwh = output_tokens * gpu_energy_per_token_kwh

    if tps:
        latency_per_token = 1 / tps
    else:
        latency_per_token = (
            LATENCY_ALPHA * active_params_b + LATENCY_BETA * BATCH_SIZE + LATENCY_GAMMA
        )
    gpu_latency = output_tokens * latency_per_token + (ttft or 0)
    latency = min(request_latency, gpu_latency)

    required_memory_gb = 1.2 * total_params_b * MODEL_QUANTIZATION_BITS / 8
    gpu_count = 2 ** math.ceil(math.log2(max(1, math.ceil(required_memory_gb / GPU_MEMORY))))

    server_energy_kwh = (latency / 3600) * SERVER_POWER * (gpu_count / SERVER_GPUS) / BATCH_SIZE
    it_energy_kwh = server_energy_kwh + gpu_count * gpu_energy_kwh
    energy_kwh = pue * it_energy_kwh

    usage_gwp_kg = energy_kwh * mix_gwp
    embodied_unit_gwp = (
        (gpu_count / SERVER_GPUS) * SERVER_EMBODIED_IMPACT_GWP
        + gpu_count * GPU_EMBODIED_IMPACT_GWP
    )
    embodied_gwp_kg = latency * embodied_unit_gwp / (HARDWARE_LIFESPAN * BATCH_SIZE)

    return energy_kwh, usage_gwp_kg + embodied_gwp_kg


# ============================================================
# Model registry
# ============================================================
# The live registry is published weekly to this repo's `registry` branch by a
# GitHub Action (scripts/sync-eco-models.py) and fetched from REGISTRY_URL; the last good copy
# is cached at registry_path. This embedded copy is only the fallback when
# neither has ever been available. Parameter data from EcoLogits models.json at the pinned
# commit; entries whose architecture is not publicly released are estimates.

REGISTRY_URL = (
    "https://raw.githubusercontent.com/CUNY-AI-Lab/eco-impact-openwebui/"
    "registry/eco_models.json"
)
FETCH_RETRY_SECONDS = 3600  # after a failed fetch

DEFAULT_REGISTRY = {
    "version": 1,
    "source": "embedded-fallback mlco2/ecologits@4215423",
    "generated": "2026-07-06",
    "defaults": {
        "zone": "WOR",
        "pue": {"min": 1.09, "max": 1.2},
        "unknown_params": {
            "total": {"min": 8, "max": 440},
            "active": {"min": 8, "max": 110},
        },
    },
    "zones": {
        "WOR": {"gwp": 0.45829, "adpe": 1.349e-08, "pe": 2.5871, "wue": 3.908},
        "USA": {"gwp": 0.3844, "adpe": 9.855e-08, "pe": 9.6884, "wue": 3.1321},
    },
    "zone_prefixes": [
        {"prefix": "us.", "zone": "USA"},
        {"prefix": "amazon.", "zone": "USA"},
        {"prefix": "anthropic.", "zone": "USA"},
        {"prefix": "meta.", "zone": "USA"},
        {"prefix": "mistral.", "zone": "USA"},
    ],
    "patterns": [
        {"match": "gpt-oss-120b", "total": {"min": 117, "max": 117},
         "active": {"min": 5.1, "max": 5.1}, "confidence": "published"},
        {"match": "gpt-oss-20b", "total": {"min": 21, "max": 21},
         "active": {"min": 3.6, "max": 3.6}, "confidence": "published"},
        {"match": "gpt-4o-mini", "total": {"min": 8, "max": 28},
         "active": {"min": 8, "max": 28}, "tps": 33.0, "ttft": 0.6,
         "confidence": "estimated"},
        {"match": "gpt-4o", "total": {"min": 440, "max": 440},
         "active": {"min": 44, "max": 132}, "tps": 46.2, "ttft": 0.53,
         "confidence": "estimated"},
        {"match": "gpt-4-1-mini", "total": {"min": 40, "max": 112},
         "active": {"min": 40, "max": 112}, "tps": 34.4, "ttft": 0.69,
         "confidence": "estimated"},
        {"match": "gpt-4-1", "total": {"min": 352, "max": 352},
         "active": {"min": 35, "max": 106}, "tps": 42.5, "ttft": 0.63,
         "confidence": "estimated"},
        {"match": "gpt-5", "total": {"min": 300, "max": 300},
         "active": {"min": 30, "max": 90}, "tps": 53.6, "ttft": 1.57,
         "confidence": "estimated"},
        {"match": "claude-opus-4", "total": {"min": 2000, "max": 2000},
         "active": {"min": 200, "max": 600}, "tps": 12.1, "ttft": 1.74,
         "confidence": "estimated"},
        {"match": "claude-sonnet-4", "total": {"min": 440, "max": 440},
         "active": {"min": 44, "max": 132}, "tps": 32.2, "ttft": 1.33,
         "confidence": "estimated"},
        {"match": "claude-haiku-4", "total": {"min": 10, "max": 35},
         "active": {"min": 10, "max": 35}, "tps": 71.1, "ttft": 0.67,
         "confidence": "estimated"},
        {"match": "llama-3-1-8b", "total": {"min": 8.03, "max": 8.03},
         "active": {"min": 8.03, "max": 8.03}, "confidence": "published"},
        {"match": "llama-3-3-70b", "total": {"min": 70.6, "max": 70.6},
         "active": {"min": 70.6, "max": 70.6}, "confidence": "published"},
        {"match": "gemini-2-5-pro", "total": {"min": 2000, "max": 2000},
         "active": {"min": 200, "max": 600}, "tps": 89.6, "ttft": 2.28,
         "confidence": "estimated"},
        {"match": "gemini-2-5-flash", "total": {"min": 440, "max": 440},
         "active": {"min": 44, "max": 132}, "tps": 64.6, "ttft": 1.1,
         "confidence": "estimated"},
        {"match": "mistral-small", "total": {"min": 119, "max": 119},
         "active": {"min": 8, "max": 8}, "tps": 98.8, "ttft": 0.29,
         "confidence": "published"},
    ],
}


def _normalize(model_id: str) -> str:
    """Normalize a model id for pattern matching: lowercase, dots -> dashes.

    Keeps ':' and '/' so vendor prefixes stay harmless substrings
    (matching is substring-based).
    """
    return model_id.lower().strip().replace(".", "-")


def find_entry(model_id: str, registry: dict) -> Optional[dict]:
    """Return the registry pattern entry whose match string is the LONGEST
    substring of the normalized model id, or None."""
    norm = _normalize(model_id)
    best = None
    for p in registry.get("patterns", []):
        m = p.get("match", "")
        if m and m in norm and (best is None or len(m) > len(best["match"])):
            best = p
    return best


def find_id_entry(model_id: str, registry: dict) -> Optional[dict]:
    """Exact lookup in the registry's "ids" map (Gateway ids, upstream ids, model
    groups). Also matches when the id carries a connection prefix ending in '.'
    or '/' (e.g. 'gateway.glm-4.7'). Never matches a different version: the key
    has to be the whole id or a whole trailing segment."""
    ids = registry.get("ids") or {}
    mid = model_id.lower().strip()
    if mid in ids:
        return ids[mid]
    best = None
    for key in ids:
        if (mid.endswith("." + key) or mid.endswith("/" + key)) and (
                best is None or len(key) > len(best)):
            best = key
    return ids[best] if best else None


def resolve_entry(model_ids: "list[str]", registry: dict) -> Optional[dict]:
    """Registry entry for the first candidate id that resolves: exact ids first
    (for every candidate), then legacy substring patterns. Gateway models the
    weekly sync could not resolve are listed in "unresolved_ids" and get the
    flagged generic estimate, never a look-alike pattern (glm-5.3 is not glm-5)."""
    for mid in model_ids:
        e = find_id_entry(mid, registry)
        if e is not None:
            return e if _usable(e) else None
    unresolved = {"ids": {u: True for u in registry.get("unresolved_ids") or []}}
    if any(find_id_entry(mid, unresolved) for mid in model_ids):
        return None
    for mid in model_ids:
        e = find_entry(mid, registry)
        if e is not None:
            return e if _usable(e) else None
    return None


def _usable(entry: Any) -> bool:
    """A skip marker, or an entry with positive numeric total/active ranges. A
    malformed entry is treated as unknown (generic estimate), not as an error."""
    if not isinstance(entry, dict):
        return False
    if entry.get("skip"):
        return True
    try:
        return all(0 < float(entry[k]["min"]) <= float(entry[k]["max"])
                   for k in ("total", "active"))
    except (KeyError, TypeError, ValueError):
        return False


def zone_for(model_id: str, registry: dict, entry: Optional[dict] = None) -> dict:
    """Electricity mix for a request: the entry's own zone (Gateway rows carry
    one per provider), else a raw-id prefix match for legacy direct-provider ids
    (e.g. Bedrock 'us.' ids run in us-east-1), else the registry default."""
    zones = registry.get("zones", {})
    if entry and entry.get("zone") in zones:
        return zones[entry["zone"]]
    mid = model_id.lower().strip()
    zone_name = registry.get("defaults", {}).get("zone", "WOR")
    for zp in registry.get("zone_prefixes", []):
        if mid.startswith(zp["prefix"]):
            zone_name = zp["zone"]
            break
    return zones.get(zone_name) or zones.get("WOR") or DEFAULT_REGISTRY["zones"]["WOR"]


def _valid_registry(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("defaults"), dict)
        and isinstance(data.get("zones"), dict) and bool(data["zones"])
        and (bool(data.get("patterns")) or bool(data.get("ids")))
    )


def _write_cache(path: str, data: dict) -> float:
    """Atomically replace path with data; returns the new mtime. The temp name is
    unique so workers and replicas sharing the EFS directory never write the
    same temp file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                               prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return os.path.getmtime(path)


def _fetch_json(url: str, timeout: float = 5.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "eco-impact-filter/1.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def compute_impacts(entry: dict, output_tokens: int, zone: dict, defaults: dict) -> dict:
    """Min/max impacts for a registry entry. Returns Wh and g CO2e ranges."""
    pue = defaults.get("pue") or {"min": 1.2, "max": 1.2}
    tps, ttft = entry.get("tps"), entry.get("ttft")
    e_lo, g_lo = _impacts_single(
        entry["active"]["min"], entry["total"]["min"], output_tokens,
        mix_gwp=zone["gwp"], pue=pue["min"], tps=tps, ttft=ttft)
    e_hi, g_hi = _impacts_single(
        entry["active"]["max"], entry["total"]["max"], output_tokens,
        mix_gwp=zone["gwp"], pue=pue["max"], tps=tps, ttft=ttft)
    return {
        "energy_wh": (e_lo * 1000.0, e_hi * 1000.0),
        "gwp_g": (g_lo * 1000.0, g_hi * 1000.0),
    }


# ============================================================
# Formatting
# ============================================================

# CO2-equivalence tiers: (threshold_g, grams_per_unit, singular, plural, emoji)
CO2_COMPARISONS = [
    (0.0, 0.2, "breath", "breaths", "🌬️"),
    (1.0, 0.05, "m of walking", "m of walking", "🚶"),
    (10.0, 120.0, "km of driving", "km of driving", "🚗"),
]


def _fmt_num(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_range(lo: float, hi: float, unit: str) -> str:
    """Render 'lo–hi unit', collapsing to '~x unit' when the range is tight.
    Scales g->mg / Wh->mWh for small values."""
    if unit == "g" and hi < 0.1:
        lo, hi, unit = lo * 1000, hi * 1000, "mg"
    if unit == "Wh" and hi < 0.1:
        lo, hi, unit = lo * 1000, hi * 1000, "mWh"
    if lo <= 0 or hi <= lo * 1.02:
        return f"~{_fmt_num(hi)} {unit}"
    return f"{_fmt_num(lo)}–{_fmt_num(hi)} {unit}"


def _comparison(co2_g_mid: float) -> str:
    selected = CO2_COMPARISONS[0]
    for tier in CO2_COMPARISONS:
        if co2_g_mid >= tier[0]:
            selected = tier
    _, per_unit, singular, plural, emoji = selected
    value = co2_g_mid / per_unit
    label = singular if round(value, 1) == 1.0 else plural
    return f"{emoji} ~{_fmt_num(value)} {label}"


def format_status(imp: dict, unknown: bool, show_energy: bool, show_comparison: bool) -> str:
    g_lo, g_hi = imp["gwp_g"]
    parts = [f"🌱 {_fmt_range(g_lo, g_hi, 'g')} CO₂e"]
    if show_energy:
        e_lo, e_hi = imp["energy_wh"]
        parts.append(_fmt_range(e_lo, e_hi, "Wh"))
    if show_comparison:
        parts.append(_comparison((g_lo + g_hi) / 2))
    if unknown:
        parts.append("⚠️ generic estimate (model not in registry)")
    return " · ".join(parts)


# ============================================================
# Filter
# ============================================================


class Filter:
    class Valves(BaseModel):
        enabled: bool = Field(default=True, description="Master on/off switch.")
        registry_url: str = Field(
            default=REGISTRY_URL,
            description="Where to fetch the weekly registry. Empty disables "
            "fetching and uses registry_path only.",
        )
        registry_refresh_hours: float = Field(
            default=12, description="How often to re-fetch registry_url.")
        registry_path: str = Field(
            default="/app/backend/data/eco_models.json",
            description="Local registry JSON; also caches the last good fetch so "
            "restarts start warm. Embedded registry is the last resort.",
        )
        show_energy: bool = Field(default=True, description="Show Wh alongside CO2e.")
        show_comparison: bool = Field(
            default=True, description="Show a real-world equivalence (breaths etc.).")
        debug_logging: bool = Field(default=False, description="Log to server console.")

    class UserValves(BaseModel):
        enabled: bool = Field(default=True, description="Show eco impact estimates.")

    def __init__(self):
        self.valves = self.Valves()
        self._reg_mtime: Optional[float] = None
        self._reg_data: Optional[dict] = None
        self._next_fetch = 0.0  # time.monotonic() when a fetch is next due
        self._refresh_task: Optional["asyncio.Task"] = None

    def _registry(self) -> dict:
        """Load the EFS registry with an mtime cache; embedded fallback."""
        path = self.valves.registry_path
        try:
            mtime = os.path.getmtime(path)
            if mtime != self._reg_mtime:
                with open(path) as fh:
                    data = json.load(fh)
                if _valid_registry(data):
                    self._reg_data = data
                self._reg_mtime = mtime  # only reached on successful parse; a corrupt file keeps the last good copy and retries next message
        except Exception as e:
            if self.valves.debug_logging:
                print(f"[EcoImpact] registry load failed ({e}); using fallback")
        return self._reg_data or DEFAULT_REGISTRY

    def _refresh_due(self) -> bool:
        if not self.valves.registry_url:
            return False
        running = self._refresh_task is not None and not self._refresh_task.done()
        return not running and time.monotonic() >= self._next_fetch

    async def _refresh(self) -> None:
        """Fetch registry_url; on success swap it in and cache it to
        registry_path. Any failure keeps the last good copy."""
        url = self.valves.registry_url
        if not url:
            return
        ok = False
        try:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, _fetch_json, url)
            if not _valid_registry(data):
                raise ValueError("fetched registry failed validation")
            self._reg_data = data
            ok = True
            self._reg_mtime = await loop.run_in_executor(
                None, _write_cache, self.valves.registry_path, data)
        except Exception as e:
            if self.valves.debug_logging:
                print(f"[EcoImpact] registry {'cache write' if ok else 'fetch'} failed: {e}")
        finally:
            hours = max(0.1, float(self.valves.registry_refresh_hours or 12))
            self._next_fetch = time.monotonic() + (hours * 3600 if ok else FETCH_RETRY_SECONDS)

    def _get_model_id(self, body: dict) -> str:
        model = body.get("model")
        if isinstance(model, dict):
            model = model.get("id")
        if not model:
            model = (body.get("metadata") or {}).get("model")
            if isinstance(model, dict):
                model = model.get("id")
        return model or "__unknown__"

    def _get_output_tokens(self, body: dict) -> int:
        usage = body.get("usage") or {}
        # output_tokens FIRST: as of OWU v0.11.0 completion_tokens carries only
        # the most recent model call, while output_tokens stays cumulative across
        # every call made for the response (tool loops, sub-agents). Carbon has to
        # count all of them. Pre-0.11 the two keys were equal, so this is a no-op
        # on older versions.
        ct = usage.get("output_tokens") or usage.get("completion_tokens")
        if ct:
            return int(ct)
        for msg in reversed(body.get("messages") or []):
            if msg.get("role") == "assistant":
                mu = msg.get("usage") or {}
                ct = mu.get("output_tokens") or mu.get("completion_tokens")
                if ct:
                    return int(ct)
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict))
                return max(1, len(content) // 4)
        return 0

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Any = None,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        try:
            if not self.valves.enabled or __event_emitter__ is None:
                return body
            uv = (__user__ or {}).get("valves")
            if uv is not None and not getattr(uv, "enabled", True):
                return body

            output_tokens = self._get_output_tokens(body)
            if output_tokens <= 0:
                return body

            reg = self._registry()
            if self._refresh_due():
                # Background only: this reply uses the registry already loaded.
                self._refresh_task = asyncio.create_task(self._refresh())

            model_id = self._get_model_id(body)
            candidates = [model_id]
            base = ((__model__ or {}).get("info") or {}).get("base_model_id")
            if base:
                candidates.append(base)
            entry = resolve_entry(candidates, reg)
            if entry is not None and entry.get("skip"):
                return body
            unknown = entry is None
            if unknown:
                up = reg["defaults"]["unknown_params"]
                entry = {"match": "", "total": up["total"], "active": up["active"],
                         "confidence": "unknown"}

            imp = compute_impacts(entry, output_tokens,
                                  zone_for(base or model_id, reg, entry),
                                  reg.get("defaults", {}))
            line = format_status(imp, unknown, self.valves.show_energy,
                                 self.valves.show_comparison)
            if self.valves.debug_logging:
                print(f"[EcoImpact] {model_id} tokens={output_tokens} -> {line}")
            await __event_emitter__(
                {"type": "status", "data": {"description": line, "done": True}})
        except Exception as e:
            if self.valves.debug_logging:
                print(f"[EcoImpact] outlet error: {e}")
        return body


if __name__ == "__main__":
    # Smoke check (full suite: pytest custom_pipes/test_eco_impact_filter.py)
    imp = compute_impacts(
        DEFAULT_REGISTRY["patterns"][0], 500,
        DEFAULT_REGISTRY["zones"]["WOR"], DEFAULT_REGISTRY["defaults"])
    print("gpt-oss-120b / 500 tokens ->",
          format_status(imp, False, True, True))
