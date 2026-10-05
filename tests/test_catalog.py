"""Тесты каталогизации/валидации метаданных (без сети)."""

from __future__ import annotations

import copy
import json
import os

import httpx
import pytest

import catalog
import metadata_refresh


DEEPSEEK_PAYLOAD = [
    {
        "id": "deepseek-flash",
        "object": "model",
        "owned_by": "deepseek",
        "name": "DeepSeek-V4.1-Flash",
        "context_window": 1048576,
        "max_output_tokens": 393216,
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "effort": {"supported_levels": ["low", "high", "max"], "default_level": "high"},
    },
    {
        "id": "deepseek-v4-pro",
        "object": "model",
        "owned_by": "deepseek",
        "name": "DeepSeek-V4-Pro",
        "context_window": 1048576,
        "max_output_tokens": 393216,
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "effort": {"supported_levels": ["low", "high", "max"], "default_level": "high"},
    },
]

NATIVE = [
    {
        "id": "deepseek-flash",
        "modelID": "deepseek-flash",
        "providerID": "deepseek",
        "package": "@opencode/ai/providers/deepseek",
        "compatibility": {"reasoningField": "reasoning_content"},
        "capabilities": {"tools": True, "input": ["text", "image"], "output": ["text"]},
        "variants": [
            {"id": "none", "body": {"thinking": {"type": "disabled"}}},
            {
                "id": "low",
                "settings": {"reasoningEffort": "low"},
                "body": {"thinking": {"type": "enabled"}},
            },
            {
                "id": "high",
                "settings": {"reasoningEffort": "high"},
                "body": {"thinking": {"type": "enabled"}},
            },
            {
                "id": "max",
                "settings": {"reasoningEffort": "max"},
                "body": {"thinking": {"type": "enabled"}},
            },
        ],
        "cost": [{"input": 0.15, "output": 0.6, "cache": {"read": 0.003, "write": 0}}],
    },
    {
        "id": "deepseek-v4-pro",
        "modelID": "deepseek-v4-pro",
        "providerID": "deepseek",
        "package": "@opencode/ai/providers/deepseek",
        "compatibility": {"reasoningField": "reasoning_content"},
        "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
        "cost": [{"input": 0.66, "output": 1.98, "cache": {"read": 0.022, "write": 0}}],
    },
]

OPENROUTER_PAYLOAD = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-5.5",
            "name": "Anthropic: Claude Sonnet 5.5",
            "context_length": 1000000,
            "architecture": {
                "input_modalities": ["text", "image", "file"],
                "output_modalities": ["text"],
            },
            "pricing": {
                "prompt": "0.000002",
                "completion": "0.00001",
                "input_cache_read": "0.0000002",
                "input_cache_write": "0.0000025",
            },
            "top_provider": {
                "context_length": 1000000,
                "max_completion_tokens": 128000,
            },
            "supported_parameters": ["tools", "reasoning", "reasoning_effort", "temperature"],
            "reasoning": {
                "mandatory": True,
                "supported_efforts": ["max", "xhigh", "high", "medium", "low"],
                "default_effort": "high",
            },
        },
        {
            "id": "openai/gpt-5.6-luna-pro",
            "name": "OpenAI: GPT-5.6 Luna Pro",
            "context_length": 1050000,
            "architecture": {
                "input_modalities": ["file", "image", "text"],
                "output_modalities": ["text"],
            },
            "pricing": {
                "prompt": "0.0000002",
                "completion": "0.0000012",
                "input_cache_read": "0.00000002",
                "input_cache_write": "0.00000025",
                "overrides": [
                    {
                        "min_prompt_tokens": 272000,
                        "prompt": "0.0000004",
                        "completion": "0.0000018",
                        "input_cache_read": "0.00000004",
                        "input_cache_write": "0.0000005",
                    }
                ],
            },
            "top_provider": {
                "context_length": 1050000,
                "max_completion_tokens": 128000,
            },
            "supported_parameters": ["tools", "reasoning", "reasoning_effort"],
            "reasoning": {
                "mandatory": False,
                "default_enabled": True,
                "supported_efforts": ["max", "xhigh", "high", "medium", "low", "none"],
                "default_effort": "medium",
            },
        },
        {
            "id": "openrouter/auto",
            "name": "Auto Router",
            "context_length": 2000000,
            "architecture": {
                "input_modalities": ["text", "image", "audio", "file", "video"],
                "output_modalities": ["text", "image"],
                "tokenizer": "Router",
            },
            "pricing": {"prompt": "-1", "completion": "-1"},
            "top_provider": {"context_length": None, "max_completion_tokens": None},
            "supported_parameters": ["tools", "reasoning", "reasoning_effort"],
        },
    ]
}

ALIASES = {
    "deepseek-flash": {
        "upstream": "deepseek",
        "model": "deepseek-flash",
        "name": "DeepSeek Flash (homelab)",
    },
    "deepseek-v4-pro": {
        "upstream": "deepseek",
        "model": "deepseek-v4-pro",
        "name": "DeepSeek V4 Pro (homelab)",
    },
    "claude-sonnet": {
        "upstream": "openrouter",
        "model": "anthropic/claude-sonnet-5.5",
        "name": "Claude Sonnet 5.5 (homelab)",
    },
    "gpt-luna": {
        "upstream": "openrouter",
        "model": "openai/gpt-5.6-luna-pro",
        "name": "GPT-5.6 Luna Pro (homelab)",
    },
    "auto": {"upstream": "openrouter", "model": "openrouter/auto", "name": "Auto"},
}


def build() -> dict:
    normalized = {
        "deepseek": catalog.normalize_upstream_models("deepseek", DEEPSEEK_PAYLOAD, NATIVE),
        "openrouter": catalog.normalize_upstream_models(
            "openrouter", OPENROUTER_PAYLOAD, NATIVE
        ),
    }
    cat = catalog.empty_catalog()
    cat["models"] = catalog.build_alias_entries(ALIASES, normalized)
    cat["generated_at"] = catalog.now_utc_iso()
    return cat


# --------------------------------------------------------------------------- #
def test_deepseek_flash_exact_metadata_and_native_fields() -> None:
    entry = build()["models"]["deepseek-flash"]
    assert entry["limit"] == {"context": 1048576, "output": 393216}
    assert entry["metadata_quality"] == "exact"
    assert entry["capabilities"] == {
        "tools": True,
        "input": ["text", "image"],
        "output": ["text"],
    }
    assert entry["reasoning"]["supported_efforts"] == ["low", "high", "max"]
    assert entry["reasoning"]["default_effort"] == "high"
    assert entry["package"] == "@opencode/ai/providers/deepseek"
    assert entry["compatibility"] == {"reasoningField": "reasoning_content"}
    assert entry["cost"][0]["input"] == 0.15
    assert entry["cost"][0]["cache"]["read"] == 0.003
    assert [v["id"] for v in entry["variants"]] == ["none", "low", "high", "max"]
    assert entry["variants"][2]["body"] == {"thinking": {"type": "enabled"}}


def test_deepseek_pro_is_text_only() -> None:
    entry = build()["models"]["deepseek-v4-pro"]
    assert entry["capabilities"]["input"] == ["text"]
    assert "image" not in entry["capabilities"]["input"]
    assert entry["owner"] == "deepseek"


def test_openrouter_claude_pricing_and_mandatory_reasoning() -> None:
    entry = build()["models"]["claude-sonnet"]
    assert entry["limit"] == {"context": 1000000, "output": 128000}
    assert entry["capabilities"]["tools"] is True
    assert entry["reasoning"]["mandatory"] is True
    assert entry["reasoning"]["default_effort"] == "high"
    assert entry["package"] == "@opencode/ai/providers/openrouter"
    assert round(entry["cost"][0]["input"], 10) == 2.0
    assert round(entry["cost"][0]["output"], 10) == 10.0
    assert round(entry["cost"][0]["cache"]["read"], 10) == 0.2
    assert entry["cost"][0]["cache"]["write"] == 2.5
    ids = [v["id"] for v in entry["variants"]]
    assert ids == ["max", "xhigh", "high", "medium", "low"]
    assert entry["variants"][0]["settings"] == {"reasoning": {"effort": "max"}}


def test_openrouter_luna_has_tier_prices_and_none_variant() -> None:
    entry = build()["models"]["gpt-luna"]
    cost = entry["cost"]
    assert round(cost[0]["input"], 10) == 0.2
    assert round(cost[0]["cache"]["read"], 10) == 0.02
    assert round(cost[0]["cache"]["write"], 10) == 0.25
    tiered = [c for c in cost if c.get("tier")]
    assert tiered and tiered[0]["tier"] == {"type": "context", "size": 272000}
    assert round(tiered[0]["input"], 10) == 0.4
    assert "none" in [v["id"] for v in entry["variants"]]


def test_auto_router_limits_and_prices_stay_unknown() -> None:
    entry = build()["models"]["auto"]
    assert entry["limit"]["context"] is None
    assert entry["limit"]["output"] is None
    assert entry["limit"]["context_declared"] == 2000000
    assert entry["cost"] is None
    assert entry["metadata_quality"] == "unknown"
    assert entry["recommended_disabled"] is True
    assert entry["variants"] is None


def test_negative_and_malformed_prices_are_unknown_not_zero() -> None:
    payload = [
        {"id": "x", "pricing": {"prompt": "-1", "completion": "-1"}, "top_provider": {}},
        {"id": "y", "pricing": {"prompt": "oops", "completion": "0"}, "top_provider": {}},
    ]
    entries = catalog.normalize_upstream_models("openrouter", {"data": payload})
    assert entries[0]["cost"] is None
    assert entries[1]["cost"] is None


def test_missing_metadata_does_not_become_precise_defaults() -> None:
    entries = catalog.normalize_upstream_models(
        "deepseek", [{"id": "mystery", "object": "model"}], []
    )
    entry = entries[0]
    assert entry["limit"] == {"context": None, "output": None}
    assert entry["metadata_quality"] == "unknown"
    assert entry["cost"] is None
    assert entry["package"] == "@opencode/ai/providers/deepseek"
    assert entry["variants"] is None


def test_catalog_roundtrip(tmp_path) -> None:
    path = tmp_path / "catalog.json"
    data = build()
    catalog.save_catalog(path, data)
    loaded = catalog.load_catalog(path)
    assert loaded["models"]["deepseek-flash"]["limit"] == {
        "context": 1048576,
        "output": 393216,
    }
    assert loaded["generated_at"] == data["generated_at"]
    # битый файл не валит шлюз
    path.write_text("{not json", encoding="utf-8")
    assert catalog.load_catalog(path)["models"] == {}


def test_budgets_separate_hard_output_from_request_budget() -> None:
    budgets = catalog.compute_budgets({"context": 1048576, "output": 393216})
    assert budgets["hard_output"] == 393216
    assert budgets["recommended_output"] < budgets["hard_output"]
    assert budgets["usable_input"] < 1048576


# --------------------------------------------------------------------------- #
# refresh / last-known-good
# --------------------------------------------------------------------------- #
def _config() -> dict:
    return {
        "upstreams": {
            "deepseek": {"base_url": "https://api.deepseek.com/v1"},
            "openrouter": {"base_url": "https://openrouter.ai/api/v1"},
        },
        "models": copy.deepcopy(ALIASES),
    }


def test_refresh_success_updates_and_timestamps() -> None:
    def fetch(name: str):
        if name == "deepseek":
            return DEEPSEEK_PAYLOAD
        return OPENROUTER_PAYLOAD

    cat, report = metadata_refresh.refresh_catalog(
        _config(), catalog.empty_catalog(), fetch=fetch, native_models=NATIVE
    )
    assert report.upstreams["deepseek"]["status"] == "ok"
    assert cat["models"]["deepseek-flash"]["limit"]["context"] == 1048576
    assert cat["generated_at"] is not None


def test_refresh_failure_preserves_last_known_good() -> None:
    existing = build()

    def fetch(name: str):
        raise RuntimeError("network down")

    cat, report = metadata_refresh.refresh_catalog(
        _config(), existing, fetch=fetch, native_models=NATIVE
    )
    assert report.upstreams["deepseek"]["status"] == "error"
    assert report.kept_last_known_good
    # точные значения не потеряны и не заменены дефолтами
    assert cat["models"]["deepseek-flash"]["limit"]["context"] == 1048576
    assert cat["models"]["claude-sonnet"]["cost"][0]["input"] == 2.0
    assert cat["generated_at"] == existing["generated_at"]


def test_refresh_empty_or_malformed_does_not_overwrite_precise() -> None:
    existing = build()

    def fetch(name: str):
        return []  # malformed/unknown, не точные данные

    cat, report = metadata_refresh.refresh_catalog(
        _config(), existing, fetch=fetch, native_models=NATIVE
    )
    assert report.upstreams["deepseek"]["status"] == "error"
    assert cat["models"]["deepseek-flash"]["limit"]["context"] == 1048576


def test_refresh_requires_configured_allowlist() -> None:
    called: list[str] = []

    def fetch(name: str):
        called.append(name)
        return DEEPSEEK_PAYLOAD

    metadata_refresh.refresh_catalog(
        {"upstreams": {"deepseek": {"base_url": "https://x"}}},
        catalog.empty_catalog(),
        fetch=fetch,
    )
    assert called == ["deepseek"]


def test_live_fetch_rejects_upstream_outside_allowlist() -> None:
    try:
        metadata_refresh.fetch_upstream_models(
            {"upstreams": {"deepseek": {"base_url": "https://x"}}}, "evil"
        )
    except ValueError as exc:
        assert "allowlist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------- #
# LKG field-level retention + route verification
# --------------------------------------------------------------------------- #
def test_refresh_incomplete_valid_metadata_retains_precise_fields() -> None:
    existing = build()

    def fetch(name: str):
        # валидные, но неполные данные: только id, без limit/cost/efforts/caps
        if name == "deepseek":
            return {"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]}
        return {"data": [{"id": "anthropic/claude-sonnet-5.5"}]}

    cat, report = metadata_refresh.refresh_catalog(
        _config(), existing, fetch=fetch, native_models=[]  # native не маскирует
    )
    flash = cat["models"]["deepseek-flash"]
    assert flash["limit"]["context"] == 1048576
    assert flash["limit"]["output"] == 393216
    assert round(flash["cost"][0]["input"], 10) == 0.15
    assert [v["id"] for v in flash["variants"]] == ["none", "low", "high", "max"]
    assert flash["capabilities"]["input"] == ["text", "image"]
    assert flash["reasoning"]["supported_efforts"] == ["low", "high", "max"]
    assert flash["stale"] is True
    assert flash["retained_fields"]
    assert flash["stale_fields_from"] == existing["models"]["deepseek-flash"]["fetched_at"]
    # качество по-прежнему exact: удержаны и limit, и capabilities
    assert flash["metadata_quality"] == "exact"
    assert "deepseek-flash" in report.models_updated


def test_refresh_route_change_does_not_retain_old_fields() -> None:
    existing = build()
    aliases = {
        "deepseek-flash": {"upstream": "deepseek", "model": "deepseek-v4-pro"},
    }

    def fetch(name: str):
        return DEEPSEEK_PAYLOAD

    config = {"upstreams": {"deepseek": {"base_url": "https://api.deepseek.com/v1"}},
              "models": aliases}
    cat, _ = metadata_refresh.refresh_catalog(
        config, existing, fetch=fetch, native_models=NATIVE
    )
    flash = cat["models"]["deepseek-flash"]
    assert flash["upstream_model"] == "deepseek-v4-pro"
    assert flash["retained_fields"] == []
    assert flash["stale"] is False
    assert round(flash["cost"][0]["input"], 10) == 0.66  # v4-pro, не старый 0.15


def test_refresh_prunes_aliases_no_longer_in_config() -> None:
    existing = build()
    aliases = {
        "deepseek-flash": {"upstream": "deepseek", "model": "deepseek-flash"},
    }

    def fetch(name: str):
        return DEEPSEEK_PAYLOAD if name == "deepseek" else OPENROUTER_PAYLOAD

    config = {"upstreams": {"deepseek": {"base_url": "https://x"}, "openrouter": {"base_url": "https://y"}},
              "models": aliases}
    cat, report = metadata_refresh.refresh_catalog(config, existing, fetch=fetch)
    assert set(cat["models"]) == {"deepseek-flash"}
    assert "auto" in report.models_pruned


# --------------------------------------------------------------------------- #
# Live metadata wins over stale native
# --------------------------------------------------------------------------- #
def test_live_efforts_win_over_stale_native_variants() -> None:
    entry = {"id": "deepseek-flash", "context_window": 1048576, "max_output_tokens": 393216,
             "input_modalities": ["text", "image"], "output_modalities": ["text"],
             "effort": {"supported_levels": ["low", "max"], "default_level": "low"}}
    native = [{
        "id": "deepseek-flash", "modelID": "deepseek-flash", "providerID": "deepseek",
        "package": "@opencode/ai/providers/deepseek",
        "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
        "variants": [
            {"id": "none", "body": {"thinking": {"type": "disabled"}}},
            {"id": "low", "settings": {"reasoningEffort": "low"}},
            {"id": "high", "settings": {"reasoningEffort": "high"}},
            {"id": "max", "settings": {"reasoningEffort": "max"}},
        ],
        "cost": [{"input": 0.15, "output": 0.6, "cache": {"read": 0.003, "write": 0}}],
    }]
    normalized = catalog.normalize_upstream_models("deepseek", [entry], native)[0]
    ids = [v["id"] for v in normalized["variants"]]
    assert ids == ["none", "low", "max"]  # без устаревшего high
    assert "high" not in ids


def test_live_openrouter_pricing_wins_over_native_cost_and_variants() -> None:
    native = [{
        "id": "anthropic/claude-sonnet-5.5",
        "modelID": "anthropic/claude-sonnet-5.5",
        "providerID": "openrouter",
        "package": "@opencode/ai/providers/openrouter",
        "variants": [{"id": "native-only", "settings": {"reasoning": {"effort": "high"}}}],
        "cost": [{"input": 99.0, "output": 99.0, "cache": {"read": 0, "write": 0}}],
    }]
    normalized = catalog.normalize_upstream_models(
        "openrouter", OPENROUTER_PAYLOAD, native
    )
    claude = next(e for e in normalized if e["id"] == "anthropic/claude-sonnet-5.5")
    assert round(claude["cost"][0]["input"], 10) == 2.0  # live, не 99
    assert "native-only" not in [v["id"] for v in claude["variants"]]


def test_default_effort_must_belong_to_supported_set() -> None:
    ds = catalog.normalize_upstream_models(
        "deepseek",
        [{"id": "m", "effort": {"supported_levels": ["low"], "default_level": "high"}}],
        [],
    )[0]
    assert ds["reasoning"]["default_effort"] is None

    or_entry = {
        "id": "z",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["tools"],
        "pricing": {"prompt": "0.000001", "completion": "0.000001"},
        "top_provider": {"context_length": 100, "max_completion_tokens": 10},
        "reasoning": {"supported_efforts": ["low"], "default_effort": "bogus"},
    }
    or_norm = catalog.normalize_upstream_models("openrouter", {"data": [or_entry]}, [])[0]
    assert or_norm["reasoning"]["default_effort"] is None


def test_malformed_capabilities_not_marked_exact() -> None:
    entry = {"id": "m", "context_window": 100, "max_output_tokens": 10}
    norm = catalog.normalize_upstream_models("deepseek", [entry], [])[0]
    assert norm["capabilities"] is None
    assert norm["metadata_quality"] == "partial"


def test_merge_entry_retains_only_when_route_unchanged() -> None:
    old = {
        "upstream": "deepseek",
        "upstream_model": "deepseek-flash",
        "limit": {"context": 1048576, "output": 393216},
        "cost": [{"input": 0.15, "output": 0.6, "cache": {"read": 0.003, "write": 0}}],
    }
    same = {
        "upstream": "deepseek",
        "upstream_model": "deepseek-flash",
        "limit": {"context": None, "output": None},
        "cost": None,
    }
    merged = catalog.merge_entry(old, same)
    assert merged["limit"]["context"] == 1048576
    assert merged["stale"] is True

    changed = {
        "upstream": "deepseek",
        "upstream_model": "deepseek-v4-pro",
        "limit": {"context": None, "output": None},
        "cost": None,
    }
    merged2 = catalog.merge_entry(old, changed)
    assert merged2["limit"]["context"] is None
    assert merged2["stale"] is False
    assert merged2["retained_fields"] == []


# --------------------------------------------------------------------------- #
# CLI / env file
# --------------------------------------------------------------------------- #
def test_live_fetch_uses_env_file_loading(tmp_path, monkeypatch) -> None:
    import gateway

    env_file = tmp_path / "env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=from-file\nEXISTING=file\n", encoding="utf-8"
    )
    monkeypatch.setenv("EXISTING", "env")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    gateway.load_env_file(env_file)
    assert os.environ["DEEPSEEK_API_KEY"] == "from-file"
    assert os.environ["EXISTING"] == "env"  # non-overriding

    captured: dict[str, str | None] = {}

    def handler(request):
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    config = {
        "upstreams": {
            "deepseek": {
                "base_url": "https://api.deepseek.com/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
            }
        }
    }
    payload = metadata_refresh.fetch_upstream_models(
        config, "deepseek", transport=httpx.MockTransport(handler)
    )
    assert payload == {"data": []}
    assert captured["auth"] == "Bearer from-file"


def test_cli_exposes_env_file_and_sync_config() -> None:
    import catalog_cli

    parser = catalog_cli.build_parser()
    refresh = parser.parse_args(
        ["refresh", "--catalog", "c.json", "--live", "--env-file", "/tmp/env"]
    )
    assert refresh.env_file == "/tmp/env"
    sync = parser.parse_args(
        ["sync", "--catalog", "c.json", "--opencode", "o.json", "--config", "gw.json"]
    )
    assert sync.config == "gw.json"
    budget = parser.parse_args(
        ["sync", "--catalog", "c.json", "--opencode", "o.json", "--request-output-budget", "64000"]
    )
    assert budget.request_output_budget == 64000


# --------------------------------------------------------------------------- #
# Provenance: fixture vs live; brand-new precision not stale
# --------------------------------------------------------------------------- #
def test_fixture_refresh_records_fixture_provenance() -> None:
    fixture_ts = "2026-10-03T06:16:42.201620+00:00"

    def fetch(name: str):
        return DEEPSEEK_PAYLOAD if name == "deepseek" else OPENROUTER_PAYLOAD

    cat, _ = metadata_refresh.refresh_catalog(
        _config(),
        catalog.empty_catalog(),
        fetch=fetch,
        source_kind="fixture",
        source_timestamps={"deepseek": fixture_ts, "openrouter": fixture_ts},
    )
    source = cat["sources"]["deepseek"]
    assert source["kind"] == "fixture" and source["type"] == "fixture"
    assert source["fetched_at"] == fixture_ts
    assert source["processed_at"] and source["processed_at"] != fixture_ts

    entry = cat["models"]["deepseek-flash"]
    assert entry["source_type"] == "fixture"
    assert entry["fetched_at"] == fixture_ts
    assert entry["processed_at"] == source["processed_at"]
    # brand-new точные данные не помечаются stale
    assert entry["stale"] is False
    assert entry["retained_fields"] == []


def test_live_refresh_records_live_kind() -> None:
    def fetch(name: str):
        return DEEPSEEK_PAYLOAD if name == "deepseek" else OPENROUTER_PAYLOAD

    cat, _ = metadata_refresh.refresh_catalog(
        _config(), catalog.empty_catalog(), fetch=fetch
    )
    assert cat["sources"]["deepseek"]["kind"] == "live"
    assert cat["models"]["deepseek-flash"]["source_type"] == "live"
    assert cat["models"]["deepseek-flash"]["stale"] is False


def test_catalog_matches_lead_native_runtime_proof() -> None:
    import pathlib

    proof_path = (
        pathlib.Path(__file__).resolve().parents[1] / "audit" / "native-runtime-check.json"
    )
    if not proof_path.exists():
        pytest.skip("native runtime proof not present")
    raw = proof_path.read_text(encoding="utf-8")
    start = raw.find("[")
    proof = json.JSONDecoder().raw_decode(raw[start:])[0]
    models = build()["models"]
    for expected in proof:
        entry = models[expected["id"]]
        assert entry["limit"] == expected["limit"], expected["id"]
        assert entry["package"] == expected["package"], expected["id"]
        assert [v["id"] for v in entry["variants"]] == expected["variants"], expected["id"]
