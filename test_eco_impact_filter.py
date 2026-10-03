"""Tests for eco_impact_filter. Run: .venv/bin/python -m pytest custom_pipes/test_eco_impact_filter.py -v

Golden values computed with the real EcoLogits library at commit
42154236c3b275346e8b97b04f49cd19877316e6 (compute_llm_impacts).
"""
import math

import pytest

from eco_impact_filter import _impacts_single

REL = 1e-9


def test_golden_dense_8b():
    # Dense 8B model, 200 output tokens, WOR mix (gwp 0.45829), PUE 1.2, no tps/ttft.
    energy_kwh, gwp_kg = _impacts_single(
        active_params_b=8.0, total_params_b=8.0, output_tokens=200,
        mix_gwp=0.45829, pue=1.2,
    )
    assert energy_kwh == pytest.approx(1.7830516910617744e-05, rel=REL)
    assert gwp_kg == pytest.approx(9.632008989816002e-06, rel=REL)


def test_golden_moe_range_extremes():
    # gpt-4o-like MoE: total 440B, active 44-132B, 500 tokens, USA mix (gwp 0.3844),
    # PUE range 1.09-1.14, tps=46.2, ttft=0.53. Two extreme passes must reproduce
    # EcoLogits' RangeValue result exactly.
    e_lo, g_lo = _impacts_single(44.0, 440.0, 500, mix_gwp=0.3844, pue=1.09,
                                 tps=46.2, ttft=0.53)
    e_hi, g_hi = _impacts_single(132.0, 440.0, 500, mix_gwp=0.3844, pue=1.14,
                                 tps=46.2, ttft=0.53)
    assert e_lo == pytest.approx(0.0007007863545514568, rel=REL)
    assert e_hi == pytest.approx(0.0011899213268139132, rel=REL)
    assert g_lo == pytest.approx(0.00029894610495653524, rel=REL)
    assert g_hi == pytest.approx(0.0004869695882942235, rel=REL)


def test_monotonic_in_tokens():
    e1, g1 = _impacts_single(8.0, 8.0, 100, mix_gwp=0.45829, pue=1.2)
    e2, g2 = _impacts_single(8.0, 8.0, 1000, mix_gwp=0.45829, pue=1.2)
    assert e2 > e1 and g2 > g1


def test_tiny_model_gpu_count_floor():
    # 1B params -> memory 2.4 GB -> 1 GPU, must not crash on log2(1)=0.
    e, g = _impacts_single(1.0, 1.0, 10, mix_gwp=0.45829, pue=1.2)
    assert e > 0 and g > 0


import json


def test_normalize():
    from eco_impact_filter import _normalize
    assert _normalize("anthropic/Claude-Sonnet-4.5") == "anthropic/claude-sonnet-4-5"
    assert _normalize("us.amazon.nova-micro-v1:0") == "us-amazon-nova-micro-v1:0"


def test_find_entry_longest_match():
    from eco_impact_filter import DEFAULT_REGISTRY, find_entry
    e = find_entry("openai/gpt-4o-mini-2024-07-18", DEFAULT_REGISTRY)
    assert e is not None and e["match"] == "gpt-4o-mini"  # beats "gpt-4o"
    e2 = find_entry("@cf/openai/gpt-oss-120b", DEFAULT_REGISTRY)
    assert e2 is not None and e2["match"] == "gpt-oss-120b"
    assert e2["confidence"] == "published"
    assert find_entry("totally-unknown-model-9000", DEFAULT_REGISTRY) is None


def test_zone_for_bedrock_prefix():
    from eco_impact_filter import DEFAULT_REGISTRY, zone_for
    assert zone_for("us.amazon.nova-micro-v1:0", DEFAULT_REGISTRY)["gwp"] == 0.3844   # USA
    assert zone_for("openai/gpt-4o", DEFAULT_REGISTRY)["gwp"] == 0.45829              # WOR


def test_compute_impacts_range():
    from eco_impact_filter import DEFAULT_REGISTRY, compute_impacts
    entry = {"match": "gpt-4o", "total": {"min": 440, "max": 440},
             "active": {"min": 44, "max": 132}, "tps": 46.2, "ttft": 0.53}
    zone = DEFAULT_REGISTRY["zones"]["USA"]
    imp = compute_impacts(entry, 500, zone, {"pue": {"min": 1.09, "max": 1.14}})
    # Same golden values as test_golden_moe_range_extremes, in Wh / g.
    assert imp["energy_wh"][0] == pytest.approx(0.7007863545514568, rel=REL)
    assert imp["energy_wh"][1] == pytest.approx(1.1899213268139132, rel=REL)
    assert imp["gwp_g"][0] == pytest.approx(0.29894610495653524, rel=REL)
    assert imp["gwp_g"][1] == pytest.approx(0.4869695882942235, rel=REL)


def test_registry_load_cache_and_fallback(tmp_path):
    from eco_impact_filter import DEFAULT_REGISTRY, Filter
    f = Filter()
    # Missing file -> embedded fallback
    f.valves.registry_path = str(tmp_path / "nope.json")
    assert f._registry() is DEFAULT_REGISTRY
    # Valid file -> loaded
    p = tmp_path / "eco_models.json"
    reg = {"version": 1, "defaults": DEFAULT_REGISTRY["defaults"],
           "zones": DEFAULT_REGISTRY["zones"], "zone_prefixes": [],
           "patterns": [{"match": "zzz-test-model", "total": {"min": 1, "max": 1},
                         "active": {"min": 1, "max": 1}, "confidence": "published"}]}
    p.write_text(json.dumps(reg))
    f.valves.registry_path = str(p)
    loaded = f._registry()
    assert loaded["patterns"][0]["match"] == "zzz-test-model"
    # Corrupt file -> keeps last good copy
    p.write_text("{not json")
    os_utime_bump(p)
    assert f._registry()["patterns"][0]["match"] == "zzz-test-model"


def os_utime_bump(p):
    import os
    st = os.stat(p)
    os.utime(p, (st.st_atime, st.st_mtime + 10))


import asyncio
from types import SimpleNamespace


def _mk_body(model="openai/gpt-4o", completion_tokens=500):
    return {
        "model": model,
        "usage": {"prompt_tokens": 100, "completion_tokens": completion_tokens},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ],
    }


def _run_outlet(f, body, user=None):
    events = []

    async def emitter(ev):
        events.append(ev)

    async def go():
        return await f.outlet(body, __event_emitter__=emitter, __user__=user)

    out = asyncio.run(go())
    return out, events


def test_format_status_range_and_units():
    from eco_impact_filter import format_status
    imp = {"energy_wh": (0.7007, 1.1899), "gwp_g": (0.2989, 0.4870)}
    line = format_status(imp, unknown=False, show_energy=True, show_comparison=True)
    assert "CO₂e" in line and "Wh" in line and "🌱" in line
    assert "–" in line  # range rendering
    line2 = format_status(imp, unknown=True, show_energy=False, show_comparison=False)
    assert "generic estimate" in line2 and "Wh" not in line2


def test_outlet_emits_status_and_returns_body_unchanged():
    from eco_impact_filter import Filter
    f = Filter()
    f.valves.registry_path = "/nonexistent/eco_models.json"  # force embedded fallback
    body = _mk_body()
    before = json.dumps(body, sort_keys=True)
    out, events = _run_outlet(f, body)
    assert json.dumps(out, sort_keys=True) == before  # body untouched
    assert len(events) == 1
    assert events[0]["type"] == "status"
    assert events[0]["data"]["done"] is True
    assert "CO₂e" in events[0]["data"]["description"]


def test_outlet_unknown_model_flagged():
    from eco_impact_filter import Filter
    f = Filter()
    f.valves.registry_path = "/nonexistent/eco_models.json"
    out, events = _run_outlet(f, _mk_body(model="mystery-model-x"))
    assert "generic estimate" in events[0]["data"]["description"]


def test_outlet_token_fallback_from_content():
    from eco_impact_filter import Filter
    f = Filter()
    f.valves.registry_path = "/nonexistent/eco_models.json"
    body = _mk_body()
    del body["usage"]
    out, events = _run_outlet(f, body)
    assert len(events) == 1  # estimated from content length instead


def test_outlet_user_valve_off():
    from eco_impact_filter import Filter
    f = Filter()
    user = {"valves": SimpleNamespace(enabled=False)}
    out, events = _run_outlet(f, _mk_body(), user=user)
    assert events == []


def test_outlet_never_raises():
    from eco_impact_filter import Filter
    f = Filter()

    async def bad_emitter(ev):
        raise RuntimeError("boom")

    async def go():
        return await f.outlet(_mk_body(), __event_emitter__=bad_emitter)

    out = asyncio.run(go())
    assert out["model"] == "openai/gpt-4o"  # body still returned


# ============================================================
# v1.1.0: Gateway ids, per-entry zones, remote registry refresh
# ============================================================

GW_REG = {
    "version": 2,
    "defaults": {"zone": "WOR", "pue": {"min": 1.09, "max": 1.2},
                 "unknown_params": {"total": {"min": 8, "max": 440},
                                    "active": {"min": 8, "max": 110}}},
    "zones": {"WOR": {"gwp": 0.45829}, "USA": {"gwp": 0.3844}},
    "zone_prefixes": [],
    "ids": {
        "glm-4.7": {"total": {"min": 358, "max": 358}, "active": {"min": 31, "max": 39},
                    "confidence": "derived", "zone": "USA"},
        "glm-4.7-flash": {"total": {"min": 31, "max": 31}, "active": {"min": 4, "max": 5},
                          "confidence": "derived", "zone": "USA"},
        "zhipuai/glm-4.7": {"total": {"min": 358, "max": 358},
                            "active": {"min": 31, "max": 39},
                            "confidence": "derived", "zone": "USA"},
        "whisper": {"skip": True},
    },
    "patterns": [{"match": "glm-4", "total": {"min": 9, "max": 9},
                  "active": {"min": 9, "max": 9}, "confidence": "published"}],
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test may reach GitHub; remote fetches fail unless a test overrides this."""
    import eco_impact_filter

    def boom(url, timeout=5.0):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(eco_impact_filter, "_fetch_json", boom)


def test_find_id_entry_exact_beats_legacy_pattern():
    from eco_impact_filter import find_id_entry, resolve_entry
    assert find_id_entry("glm-4.7", GW_REG)["active"]["max"] == 39
    assert find_id_entry("glm-4.7-flash", GW_REG)["active"]["max"] == 5
    assert find_id_entry("GLM-4.7", GW_REG) is not None
    # exact ids win over the legacy substring "glm-4"
    assert resolve_entry(["glm-4.7"], GW_REG)["confidence"] == "derived"
    # unknown id falls back to the legacy pattern
    assert resolve_entry(["openrouter/glm-4-9b"], GW_REG)["match"] == "glm-4"


def test_find_id_entry_connection_prefix_and_no_version_bleed():
    from eco_impact_filter import find_id_entry
    assert find_id_entry("gateway.glm-4.7", GW_REG)["active"]["max"] == 39
    assert find_id_entry("cail/glm-4.7-flash", GW_REG)["active"]["max"] == 5
    # a newer version must NOT silently reuse an older model's numbers
    assert find_id_entry("glm-4.7.1", GW_REG) is None
    assert find_id_entry("glm-4.70", GW_REG) is None
    # the prefix has to end at a separator, not mid-name
    assert find_id_entry("superglm-4.7", GW_REG) is None


def test_unresolved_gateway_id_never_falls_back_to_patterns():
    from eco_impact_filter import resolve_entry
    reg = dict(GW_REG, unresolved_ids=["glm-4.9"])
    assert resolve_entry(["openrouter/glm-4.9"], GW_REG)["match"] == "glm-4"
    assert resolve_entry(["glm-4.9"], reg) is None
    assert resolve_entry(["gateway.glm-4.9"], reg) is None


def test_resolve_entry_uses_model_card_base_model():
    from eco_impact_filter import resolve_entry
    e = resolve_entry(["course-card-uuid-123", "zhipuai/glm-4.7"], GW_REG)
    assert e is not None and e["active"]["max"] == 39
    assert resolve_entry(["course-card-uuid-123"], GW_REG) is None


def test_zone_from_entry_beats_default():
    from eco_impact_filter import zone_for
    entry = GW_REG["ids"]["glm-4.7"]
    assert zone_for("glm-4.7", GW_REG, entry)["gwp"] == 0.3844
    assert zone_for("glm-4.7", GW_REG, None)["gwp"] == 0.45829


def _filter_with(tmp_path, reg=None):
    from eco_impact_filter import Filter
    f = Filter()
    p = tmp_path / "eco_models.json"
    if reg is not None:
        p.write_text(json.dumps(reg))
    f.valves.registry_path = str(p)
    return f


def test_outlet_gateway_model_uses_ids_and_card_base(tmp_path):
    f = _filter_with(tmp_path, GW_REG)
    out, events = _run_outlet(f, _mk_body(model="glm-4.7"))
    assert "generic estimate" not in events[0]["data"]["description"]
    # model card whose base is a Gateway model
    events = []

    async def emitter(ev):
        events.append(ev)

    asyncio.run(f.outlet(_mk_body(model="my-course-card"), __event_emitter__=emitter,
                         __model__={"id": "my-course-card",
                                    "info": {"base_model_id": "glm-4.7"}}))
    assert "generic estimate" not in events[0]["data"]["description"]


def test_outlet_skip_entry_emits_nothing(tmp_path):
    f = _filter_with(tmp_path, GW_REG)
    out, events = _run_outlet(f, _mk_body(model="whisper"))
    assert events == []


def test_remote_refresh_success_updates_and_caches(tmp_path, monkeypatch):
    import eco_impact_filter
    f = _filter_with(tmp_path)  # no file on disk yet -> embedded
    calls = []

    def fake_fetch(url, timeout=5.0):
        calls.append(url)
        return GW_REG

    monkeypatch.setattr(eco_impact_filter, "_fetch_json", fake_fetch)
    f.valves.registry_url = "https://example.test/eco_models.json"
    asyncio.run(f._refresh())
    assert calls == ["https://example.test/eco_models.json"]
    assert "glm-4.7" in f._registry()["ids"]
    # written to registry_path as the warm cache for restarts
    assert json.loads((tmp_path / "eco_models.json").read_text())["ids"]
    # not due again until the refresh interval passes
    assert f._refresh_due() is False


def test_remote_refresh_failure_keeps_last_good(tmp_path, monkeypatch):
    import eco_impact_filter
    f = _filter_with(tmp_path, GW_REG)
    f.valves.registry_url = "https://example.test/eco_models.json"
    assert "ids" in f._registry()
    asyncio.run(f._refresh())  # autouse fixture makes the fetch fail
    assert "glm-4.7" in f._registry()["ids"]
    # an invalid payload is rejected too
    monkeypatch.setattr(eco_impact_filter, "_fetch_json",
                        lambda url, timeout=5.0: {"nope": True})
    asyncio.run(f._refresh())
    assert "glm-4.7" in f._registry()["ids"]


def test_remote_refresh_disabled_when_url_empty(tmp_path, monkeypatch):
    import eco_impact_filter
    f = _filter_with(tmp_path)
    f.valves.registry_url = ""
    called = []
    monkeypatch.setattr(eco_impact_filter, "_fetch_json",
                        lambda url, timeout=5.0: called.append(url))
    assert f._refresh_due() is False
    asyncio.run(f._refresh())
    assert called == []


def test_outlet_does_not_wait_for_remote_fetch(tmp_path, monkeypatch):
    import time
    import eco_impact_filter
    f = _filter_with(tmp_path, GW_REG)
    f.valves.registry_url = "https://example.test/eco_models.json"

    def slow_fetch(url, timeout=5.0):
        time.sleep(0.5)
        return GW_REG

    monkeypatch.setattr(eco_impact_filter, "_fetch_json", slow_fetch)
    events = []

    async def emitter(ev):
        events.append(ev)

    async def go():
        t0 = time.monotonic()
        await f.outlet(_mk_body(model="glm-4.7"), __event_emitter__=emitter)
        elapsed = time.monotonic() - t0
        if f._refresh_task is not None:
            await f._refresh_task
        return elapsed

    assert asyncio.run(go()) < 0.3
    assert len(events) == 1


def test_malformed_entry_treated_as_unknown(tmp_path):
    reg = json.loads(json.dumps(GW_REG))
    del reg["ids"]["glm-4.7"]["active"]
    f = _filter_with(tmp_path, reg)
    out, events = _run_outlet(f, _mk_body(model="glm-4.7"))
    assert len(events) == 1 and "generic estimate" in events[0]["data"]["description"]


def test_cache_write_uses_unique_temp_and_cleans_up(tmp_path, monkeypatch):
    import eco_impact_filter
    f = _filter_with(tmp_path)
    f.valves.registry_url = "https://example.test/eco_models.json"
    monkeypatch.setattr(eco_impact_filter, "_fetch_json", lambda url, timeout=5.0: GW_REG)
    (tmp_path / "eco_models.json.tmp").write_text("another worker's half-written file")
    asyncio.run(f._refresh())
    assert json.loads((tmp_path / "eco_models.json").read_text())["ids"]
    # did not clobber or reuse the shared-name temp file, and left no temp of its own
    assert (tmp_path / "eco_models.json.tmp").read_text() == "another worker's half-written file"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["eco_models.json", "eco_models.json.tmp"]
