"""Tests for scripts/sync-eco-models.py (offline: no GitHub, Gateway, or HF calls).

Run: .venv/bin/python -m pytest test_sync_eco_models.py -v
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "sync_eco_models", os.path.join(HERE, "scripts", "sync-eco-models.py"))
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def row(id, provider="workers-ai", task="Text Generation", group=None, hf=None,
        size=None, arch=None):
    spec = {}
    if hf:
        repo, rev = hf
        spec["architecture"] = {
            "name": arch or "X",
            "source_url": f"https://huggingface.co/{repo}/blob/{rev}/config.json"}
    if size:
        spec["size"] = {"label": size, "basis": "checkpoint"}
    return {"id": id, "upstream_model": id, "provider": provider, "task": task,
            "model_group": group, "specifications": spec or None}


PUB = {"total": {"min": 358, "max": 358}, "active": {"min": 32, "max": 32},
       "confidence": "published"}
EST = {"total": {"min": 400, "max": 600}, "active": {"min": 35, "max": 50},
       "confidence": "estimated"}

DENSE_CFG = {"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336}
MOE_CFG = {"hidden_size": 1000, "num_hidden_layers": 10, "moe_intermediate_size": 100,
           "n_routed_experts": 64, "num_experts_per_tok": 4}
# inactive = 10 layers * (64-4) experts * 3 * 1000 * 100 = 180M
MOE_INACTIVE = 10 * 60 * 3 * 1000 * 100


def no_hf(repo, rev):
    return None


# ---------- helpers ----------

def test_variant_names_strips_weight_neutral_suffixes():
    assert sync.variant_names("llama-3.1-8b-instruct-fp8") == [
        "llama-3-1-8b-instruct-fp8", "llama-3-1-8b-instruct", "llama-3-1-8b"]
    assert sync.variant_names("glm-5.3") == ["glm-5-3"]


def test_size_label_b():
    assert sync.size_label_b("358B") == 358.0
    assert sync.size_label_b("1T") == 1000.0
    assert sync.size_label_b("79.7B") == 79.7
    assert sync.size_label_b(None) is None
    assert sync.size_label_b("big") is None


def test_row_names_include_group_and_hf_repo():
    r = row("glm-4.7", group="zhipuai/glm-4.7", hf=("zai-org/GLM-4.7", "602d01e"))
    assert sync.row_names(r) == ["glm-4.7"]  # id, group basename, HF repo all dedupe
    r2 = row("v3.2", hf=("deepseek-ai/DeepSeek-V3.2", "abcdef1"))
    assert "deepseek-v3.2" in sync.row_names(r2)


# ---------- HF config -> active params ----------

def test_active_dense():
    assert sync.active_from_config(8e9, DENSE_CFG) == (8e9, "dense")


def test_active_moe_standard_keys():
    active, kind = sync.active_from_config(1e9, MOE_CFG)
    assert kind == "moe" and active == pytest.approx(1e9 - MOE_INACTIVE)


def test_active_moe_first_dense_layers_and_text_config():
    cfg = {"text_config": dict(MOE_CFG, first_k_dense_replace=2)}
    active, kind = sync.active_from_config(1e9, cfg)
    assert active == pytest.approx(1e9 - 8 * 60 * 3 * 1000 * 100)


def test_active_moe_gemma4_style_keys():
    cfg = {"hidden_size": 1000, "num_hidden_layers": 10, "moe_intermediate_size": 100,
           "intermediate_size": 400, "num_experts": 64, "top_k_experts": 4,
           "enable_moe_block": True}
    active, kind = sync.active_from_config(1e9, cfg)
    assert kind == "moe" and active == pytest.approx(1e9 - MOE_INACTIVE)


def test_active_unparseable_moe_is_not_called_dense():
    active, why = sync.active_from_config(1e9, {"hidden_size": 1, "moe_layer_freq": 1})
    assert active is None and "expert" in why
    active, why = sync.active_from_config(1e9, {"num_experts": 8, "hidden_size": 1})
    assert active is None


# ---------- resolution priority ----------

def test_published_curated_beats_hf():
    r = row("glm-4.7", hf=("zai-org/GLM-4.7", "602d01e"))
    hf = lambda repo, rev: (999e9, DENSE_CFG)
    e, why = sync.resolve_row(r, {"glm-4-7": PUB}, {}, hf)
    assert e["source"] == "curated:glm-4-7" and e["active"]["max"] == 32


def test_hf_beats_estimated_curated():
    r = row("qwen3-coder-next", hf=("Qwen/Qwen3-Coder-Next", "abc1234"))
    hf = lambda repo, rev: (1e9, MOE_CFG)
    e, why = sync.resolve_row(r, {"qwen3-coder-next": EST}, {}, hf)
    a = (1e9 - MOE_INACTIVE) / 1e9
    assert e["confidence"] == "derived"
    assert e["active"]["max"] == pytest.approx(a, abs=1e-3)
    assert e["active"]["min"] == pytest.approx(0.8 * a, abs=1e-3)
    assert e["source"].startswith("hf:Qwen/Qwen3-Coder-Next@")


def test_estimated_curated_used_when_nothing_better():
    r = row("qwen3-coder-next")  # no HF link
    e, why = sync.resolve_row(r, {"qwen3-coder-next": EST}, {}, no_hf)
    assert e["confidence"] == "estimated"


def test_name_hint_gives_active_params():
    r = row("nemotron-3-120b-a12b", size="120B")
    e, why = sync.resolve_row(r, {}, {}, no_hf)
    assert e["active"] == {"min": 12.0, "max": 12.0}
    assert e["total"] == {"min": 120.0, "max": 120.0}
    assert e["source"] == "name:nemotron-3-120b-a12b"


def test_no_version_bleed():
    # glm-5.3 must not inherit glm-5's numbers
    e, why = sync.resolve_row(row("glm-5.3"), {"glm-5": PUB}, {}, no_hf)
    assert e is None and "no curated entry" in why


def test_gated_hf_reports_reason():
    r = row("llama-3.2-11b-vision-instruct", hf=("meta-llama/Llama-3.2-11B-Vision", "abc1234"))
    e, why = sync.resolve_row(r, {}, {}, no_hf)
    assert e is None and "unavailable" in why


# ---------- build_ids ----------

def test_build_ids_keys_zone_and_skip():
    rows = [row("glm-4.7", provider="bedrock-mantle", group="zhipuai/glm-4.7"),
            row("whisper", task="Automatic Speech Recognition"),
            row("glm-5.3")]
    ids, resolved, unresolved = sync.build_ids(rows, {"glm-4-7": PUB}, {}, no_hf)
    assert ids["glm-4.7"]["zone"] == "USA" and ids["glm-4.7"]["provider"] == "bedrock-mantle"
    assert ids["zhipuai/glm-4.7"] is ids["glm-4.7"]
    assert ids["whisper"]["skip"] is True
    assert [u["id"] for u in unresolved] == ["glm-5.3"]
    assert [r for r, _ in resolved] == ["glm-4.7"]


# ---------- validation gate ----------

def _reg(ids):
    return {"zones": {"USA": {"gwp": 0.38}}, "ids": ids}


def _e(active_max, total=100):
    return {"total": {"min": total, "max": total},
            "active": {"min": active_max, "max": active_max}, "zone": "USA",
            "source": "x"}


def test_validate_ok_and_first_run():
    assert sync.validate(_reg({"a": _e(10)}), None, ["a"]) == []
    assert sync.validate(_reg({"a": _e(10)}), _reg({"a": _e(12)}), ["a"]) == []


def test_validate_flags_lost_coverage_only_for_live_models():
    prev = _reg({"a": _e(10), "gone": _e(10)})
    problems = sync.validate(_reg({}), prev, ["a"])
    assert len(problems) == 1 and "`a`" in problems[0]


def test_validate_flags_large_active_change():
    problems = sync.validate(_reg({"a": _e(40)}), _reg({"a": _e(10)}), ["a"])
    assert len(problems) == 1 and "moved" in problems[0]


def test_validate_flags_invalid_entry():
    bad = _e(200, total=100)  # active > total
    assert sync.validate(_reg({"a": bad}), None, ["a"])


def test_same_content_ignores_generated_date():
    a = {"generated": "2026-10-03", "ids": {"x": 1}}
    assert sync.same_content(a, dict(a, generated="2026-10-10"))
    assert not sync.same_content(a, {"generated": "2026-10-03", "ids": {"x": 2}})


# ---------- EcoLogits methodology guard ----------

def test_choose_ref_latest_when_methodology_unchanged():
    def fetch(url, headers=None):
        if url.endswith("/releases/latest"):
            return {"tag_name": "9.9.9"}
        return {"sha": sync.VENDORED_LLM_BLOB}
    assert sync.choose_ecologits_ref("latest", fetch) == ("9.9.9", None)


def test_choose_ref_keeps_pin_when_methodology_changed():
    def fetch(url, headers=None):
        if url.endswith("/releases/latest"):
            return {"tag_name": "9.9.9"}
        return {"sha": "0" * 40}
    ref, warning = sync.choose_ecologits_ref("latest", fetch)
    assert ref == sync.PINNED_REF and "re-vendor" in warning


def test_choose_ref_explicit_passthrough():
    assert sync.choose_ecologits_ref("abc", None) == ("abc", None)


# ---------- report ----------

def test_report_lists_unresolved_and_problems():
    md = sync.render_report(
        [("glm-4.7", dict(PUB, source="curated:glm-4-7"))],
        [{"id": "glm-5.3", "keys": ["glm-5.3"], "provider": "workers-ai", "reason": "no data", "hf": None,
          "size": None}],
        ["`a` moved"], None, "mlco2/ecologits@x", "scripts/eco-extra-patterns.json")
    assert "`glm-5.3`" in md and "Held for review" in md and '"match": "glm-5-3"' in md


# ---------- end to end (CLI, offline) ----------

@pytest.fixture
def eco_dir(tmp_path):
    d = tmp_path / "ecologits" / "ecologits" / "data"
    d.mkdir(parents=True)
    (d / "models.json").write_text(json.dumps({"aliases": [], "models": [
        {"provider": "huggingface_hub", "name": "openai/gpt-oss-120b",
         "architecture": {"type": "moe", "parameters": {"total": 117, "active": 5.1}}}]}))
    (d / "electricity_mixes.json").write_text(json.dumps({"electricity_mixes": [
        {"name": "WOR", "gwp": 0.45829, "adpe": 1, "pe": 1, "wue": 1},
        {"name": "USA", "gwp": 0.3844, "adpe": 1, "pe": 1, "wue": 1}]}))
    return tmp_path / "ecologits"


def _run_cli(tmp_path, eco_dir, rows, previous=None):
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps({"object": "list", "data": rows}))
    out = tmp_path / "eco_models.json"
    report = tmp_path / "report.md"
    args = [sys.executable, os.path.join(HERE, "scripts", "sync-eco-models.py"),
            "--catalog-json", str(cat), "--ecologits-dir", str(eco_dir), "--no-hf",
            "--extra", os.path.join(HERE, "scripts", "eco-extra-patterns.json"),
            "-o", str(out), "--report", str(report)]
    if previous is not None:
        prev = tmp_path / "previous.json"
        prev.write_text(json.dumps(previous))
        args += ["--previous", str(prev)]
    p = subprocess.run(args, capture_output=True, text=True)
    return p, out, report


def test_cli_end_to_end_and_filter_reads_output(tmp_path, eco_dir):
    rows = [row("gpt-oss-120b", group="openai/gpt-oss-120b"),
            row("glm-4.7", provider="bedrock-mantle"),
            row("whisper", task="Automatic Speech Recognition"),
            row("glm-5.3")]
    p, out, report = _run_cli(tmp_path, eco_dir, rows)
    assert p.returncode == 0, p.stdout + p.stderr
    reg = json.loads(out.read_text())
    assert reg["version"] == 2 and reg["ids"]["gpt-oss-120b"]["active"]["max"] == 5.1
    assert "`glm-5.3`" in report.read_text()
    assert reg["unresolved_ids"] == ["glm-5.3"]

    sys.path.insert(0, HERE)
    import eco_impact_filter as f
    assert f._valid_registry(reg)
    e = f.resolve_entry(["openai/gpt-oss-120b"], reg)
    imp = f.compute_impacts(e, 500, f.zone_for("gpt-oss-120b", reg, e), reg["defaults"])
    assert imp["gwp_g"][0] > 0
    assert f.resolve_entry(["whisper"], reg)["skip"] is True
    assert f.resolve_entry(["glm-5.3"], reg) is None  # not the legacy "glm-5" pattern


def test_cli_exits_3_and_still_writes_when_gate_fails(tmp_path, eco_dir):
    rows = [row("glm-5.3")]
    previous = {"generated": "2026-09-01", "ids": {"glm-5.3": _e(40)}}
    p, out, report = _run_cli(tmp_path, eco_dir, rows, previous)
    assert p.returncode == sync.EXIT_NEEDS_REVIEW, p.stdout + p.stderr
    assert out.exists() and "Held for review" in report.read_text()


def test_committed_registry_is_valid_for_the_filter():
    """A published registry must load in the filter and pass the gate's checks.
    The weekly workflow points ECO_REGISTRY at the freshly built file."""
    path = os.environ.get("ECO_REGISTRY") or os.path.join(HERE, "examples", "eco_models.json")
    reg = json.load(open(path))
    sys.path.insert(0, HERE)
    import eco_impact_filter as f
    assert f._valid_registry(reg)
    assert sync.validate(reg, None, list(reg["ids"])) == []


# ---------- review follow-ups ----------

def test_validate_malformed_live_entry_is_held_not_crash():
    bad = {"total": {"min": 1, "max": 1}, "zone": "USA"}  # no "active"
    problems = sync.validate(_reg({"a": bad}), _reg({"a": _e(10)}), ["a"])
    assert any("invalid entry" in p for p in problems)


def test_validate_holds_empty_or_shrunken_catalog():
    prev = _reg({k: _e(10) for k in "abcdefghij"})
    assert any("catalog" in p for p in sync.validate(_reg({}), prev, []))
    assert any("catalog" in p for p in sync.validate(_reg({"a": _e(10)}), prev, ["a", "b"]))
    # normal churn (a couple of models retired) is fine
    keep = {k: _e(10) for k in "abcdefgh"}
    assert sync.validate(_reg(keep), prev, list(keep)) == []


def test_choose_ref_falls_back_to_previous_ref_on_drift():
    def fetch(url, headers=None):
        if url.endswith("/releases/latest"):
            return {"tag_name": "9.9.9"}
        return {"sha": "0" * 40}
    ref, warning = sync.choose_ecologits_ref("latest", fetch, previous_ref="0.11.2")
    assert ref == "0.11.2" and "0.11.2" in warning
