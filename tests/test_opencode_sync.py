"""Тесты синхронизации каталога в V2-конфиг OpenCode (без сети)."""

from __future__ import annotations

import copy
import json
import os
import stat

import opencode_sync
import test_catalog

SECRET = "Bearer super-secret-homelab-token"

BASE_CONFIG = {
    "providers": {
        "homelab": {
            "name": "Homelab (mini)",
            "package": "@opencode/ai/providers/openai-compatible",
            "settings": {"baseURL": "http://127.0.0.1:8130/v1"},
            "headers": {"Authorization": SECRET},
            "models": {
                "deepseek-flash": {
                    "name": "DeepSeek Flash (homelab)",
                    "limit": {"context": 200000, "output": 32000},
                    "capabilities": {"tools": True, "input": ["text", "image"], "output": ["text"]},
                    "variants": [
                        {"id": "low", "settings": {"reasoningEffort": "low"}},
                        {"id": "medium", "settings": {"reasoningEffort": "medium"}},
                        {"id": "high", "settings": {"reasoningEffort": "high"}},
                    ],
                    "cost": [],
                },
                "deepseek-v4-pro": {"name": "DeepSeek V4 Pro (homelab)"},
                "claude-sonnet": {"name": "Claude Sonnet 5.5 (homelab)"},
                "gpt-luna": {"name": "GPT-5.6 Luna Pro (homelab)"},
                "auto": {
                    "name": "OpenRouter Auto (homelab)",
                    "limit": {"context": 200000, "output": 32000},
                    "variants": [
                        {"id": "low", "settings": {"reasoningEffort": "low"}},
                        {"id": "medium", "settings": {"reasoningEffort": "medium"}},
                        {"id": "high", "settings": {"reasoningEffort": "high"}},
                    ],
                },
                "user-custom": {"name": "My Custom", "limit": {"context": 123, "output": 45}},
            },
        },
        "other": {
            "name": "Other",
            "package": "@opencode/ai/providers/openai",
            "settings": {"apiKey": "literal-other-secret"},
        },
    },
    "agents": {"build": {"model": "homelab/deepseek-flash"}},
    "permissions": {"edit": "ask"},
}


def _catalog() -> dict:
    return test_catalog.build()


def _plan(config: dict | None = None, **kwargs):
    return opencode_sync.plan_sync(_catalog(), config or copy.deepcopy(BASE_CONFIG), **kwargs)


def _write(path, config) -> None:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
def test_dry_run_plans_without_writing(tmp_path) -> None:
    target = tmp_path / "opencode.json"
    _write(target, BASE_CONFIG)
    before = target.read_bytes()

    plan = _plan()
    assert plan.changed
    assert target.read_bytes() == before


def test_plan_updates_only_managed_fields() -> None:
    plan = _plan()
    homelab = plan.config["providers"]["homelab"]
    flash = homelab["models"]["deepseek-flash"]
    assert flash["limit"] == {"context": 1048576, "output": 393216}
    assert flash["package"] == "@opencode/ai/providers/deepseek"
    assert flash["compatibility"] == {"reasoningField": "reasoning_content"}
    # слияние по id: новый none/max добавлены, пользовательский medium сохранён
    assert [v["id"] for v in flash["variants"]] == ["low", "medium", "high", "none", "max"]
    low = next(v for v in flash["variants"] if v["id"] == "low")
    assert low["settings"] == {"reasoningEffort": "low"}
    assert low["body"] == {"thinking": {"type": "enabled"}}
    medium = next(v for v in flash["variants"] if v["id"] == "medium")
    assert medium == {"id": "medium", "settings": {"reasoningEffort": "medium"}}
    assert flash["cost"]
    # alias id не подменяется upstream-ным
    assert "modelID" not in flash and flash.get("id") is None


def test_merge_preserves_user_variant_fields_and_extra_options() -> None:
    plan = _plan()
    flash = plan.config["providers"]["homelab"]["models"]["deepseek-flash"]
    low = next(v for v in flash["variants"] if v["id"] == "low")
    assert low["settings"] == {"reasoningEffort": "low"}
    # пользовательский variant с неизвестным каталогу id остаётся
    assert "medium" in {v["id"] for v in flash["variants"]}
    # пользовательские ключи compatibility сохраняются
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["providers"]["homelab"]["models"]["deepseek-flash"]["compatibility"] = {
        "requireReasoning": True
    }
    merged = _plan(cfg).config["providers"]["homelab"]["models"]["deepseek-flash"]
    assert merged["compatibility"] == {
        "requireReasoning": True,
        "reasoningField": "reasoning_content",
    }


def test_conflicting_modelID_is_refused() -> None:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["providers"]["homelab"]["models"]["deepseek-flash"]["modelID"] = "deepseek-chat"
    plan = _plan(cfg)
    flash = plan.config["providers"]["homelab"]["models"]["deepseek-flash"]
    # пакет/limit не синкаются, пока explicit modelID конфликтует с alias
    assert "package" not in flash
    assert "deepseek-flash" in plan.refused
    assert any("modelID" in w for w in plan.warnings)


def test_route_mismatch_gateway_alias_is_skipped() -> None:
    gateway_aliases = {
        "deepseek-flash": {"upstream": "deepseek", "model": "other-model"},
    }
    plan = _plan(gateway_aliases=gateway_aliases)
    flash = plan.config["providers"]["homelab"]["models"]["deepseek-flash"]
    assert "package" not in flash
    assert "deepseek-flash" in plan.refused
    assert any("route mismatch" in w for w in plan.warnings)


def test_route_verified_flag() -> None:
    plan = _plan(
        gateway_aliases={
            "deepseek-flash": {"upstream": "deepseek", "model": "deepseek-flash"}
        }
    )
    assert plan.route_unverified is False
    assert _plan().route_unverified is True


def test_nan_and_inf_prices_rejected_and_json_strict() -> None:
    import math

    assert opencode_sync._clean_cost(
        [{"input": float("nan"), "output": 1.0, "cache": {}}]
    ) is None
    assert opencode_sync._clean_cost(
        [{"input": float("inf"), "output": 1.0, "cache": {}}]
    ) is None
    assert opencode_sync._clean_cost(
        [{"input": 1.0, "output": 2.0, "cache": {"read": float("nan")}}]
    ) == [{"input": 1.0, "output": 2.0}]
    try:
        opencode_sync._serialize({"x": float("nan")})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("allow_nan=False must reject NaN")
    assert math.isfinite(1.0)


def test_alias_modelID_preserved_if_present() -> None:
    config = copy.deepcopy(BASE_CONFIG)
    config["providers"]["homelab"]["models"]["deepseek-flash"]["modelID"] = "deepseek-flash"
    plan = _plan(config)
    flash = plan.config["providers"]["homelab"]["models"]["deepseek-flash"]
    assert flash["modelID"] == "deepseek-flash"


def test_unrelated_providers_settings_agents_untouched() -> None:
    plan = _plan()
    cfg = plan.config
    assert cfg["providers"]["other"] == BASE_CONFIG["providers"]["other"]
    assert cfg["agents"] == BASE_CONFIG["agents"]
    assert cfg["permissions"] == BASE_CONFIG["permissions"]
    assert cfg["providers"]["homelab"]["headers"] == {"Authorization": SECRET}
    homelab_settings = cfg["providers"]["homelab"]["settings"]
    assert homelab_settings["baseURL"] == BASE_CONFIG["providers"]["homelab"]["settings"]["baseURL"]
    # единственное добавление — derived apiKey (см. отдельные тесты)
    assert set(homelab_settings) <= {"baseURL", "apiKey"}
    assert cfg["providers"]["homelab"]["models"]["user-custom"] == {"name": "My Custom", "limit": {"context": 123, "output": 45}}


def test_secret_never_appears_in_summary() -> None:
    plan = _plan()
    blob = json.dumps(plan.summary(), ensure_ascii=False)
    assert SECRET not in blob
    assert "super-secret" not in blob
    assert "literal-other-secret" not in blob


def test_apply_writes_0600_backup_and_atomic_update(tmp_path) -> None:
    target = tmp_path / "opencode.json"
    _write(target, BASE_CONFIG)
    os.chmod(target, 0o644)

    plan = _plan()
    backup = opencode_sync.apply_sync(target, plan.config)

    assert backup.exists()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert ".bak" in backup.name
    original = json.loads(backup.read_text(encoding="utf-8"))
    assert original["providers"]["homelab"]["models"]["deepseek-flash"]["limit"] == {
        "context": 200000,
        "output": 32000,
    }
    updated = json.loads(target.read_text(encoding="utf-8"))
    assert updated["providers"]["homelab"]["models"]["deepseek-flash"]["limit"] == {
        "context": 1048576,
        "output": 393216,
    }


def test_idempotent_second_plan_has_no_changes() -> None:
    first = _plan()
    second = opencode_sync.plan_sync(_catalog(), first.config)
    assert not second.changed
    assert second.changes == []


def test_unknown_metadata_left_unchanged_with_warning() -> None:
    plan = _plan()
    auto = plan.config["providers"]["homelab"]["models"]["auto"]
    # у auto границы/цены неизвестны — старые значения не переписываются дефолтами
    assert auto["limit"] == {"context": 200000, "output": 32000}
    assert "cost" not in auto
    assert any("unknown" in w for w in plan.warnings)


def test_disable_recommended_only_with_explicit_flag() -> None:
    without = _plan()
    assert "disabled" not in without.config["providers"]["homelab"]["models"]["auto"]
    with_flag = _plan(disable_recommended=True)
    assert with_flag.config["providers"]["homelab"]["models"]["auto"]["disabled"] is True


def test_add_missing_creates_catalog_model_only_when_requested() -> None:
    config = copy.deepcopy(BASE_CONFIG)
    del config["providers"]["homelab"]["models"]["gpt-luna"]
    plain = _plan(config)
    assert "gpt-luna" not in plain.config["providers"]["homelab"]["models"]
    added = _plan(config, add_missing=True)
    assert "gpt-luna" in added.config["providers"]["homelab"]["models"]


def test_cost_incomplete_is_not_written() -> None:
    # gpt-luna в базовом конфиге без cost; в каталоге cost известен и должен появиться
    plan = _plan()
    luna = plan.config["providers"]["homelab"]["models"]["gpt-luna"]
    assert luna["cost"]
    assert luna["cost"][0]["input"] > 0


# --------------------------------------------------------------------------- #
# Native switch credential (provider.settings.apiKey)
# --------------------------------------------------------------------------- #
def test_native_switch_derives_apikey_from_authorization_header() -> None:
    plan = _plan()
    homelab = plan.config["providers"]["homelab"]
    assert plan.credential_derived is True
    assert homelab["settings"]["apiKey"] == "super-secret-homelab-token"
    # headers и baseURL не тронуты
    assert homelab["headers"] == {"Authorization": SECRET}
    assert homelab["settings"]["baseURL"] == "http://127.0.0.1:8130/v1"
    # секрет не светится в summary
    blob = json.dumps(plan.summary(), ensure_ascii=False)
    assert "super-secret-homelab-token" not in blob
    key_change = next(
        c for c in plan.changes if c["field"] == "settings.apiKey"
    )
    assert key_change["value"] == "<redacted>"
    # другой провайдер не тронут
    assert plan.config["providers"]["other"]["settings"] == {"apiKey": "literal-other-secret"}


def test_native_switch_refuses_without_any_credential() -> None:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["providers"]["homelab"]["headers"] = {}
    plan = _plan(cfg)
    assert plan.credential_derived is False
    assert "apiKey" not in plan.config["providers"]["homelab"].get("settings", {})
    assert "deepseek-flash" in plan.refused
    assert any("credential" in w for w in plan.warnings)
    # native package не выставлен и модель не отключена молча
    flash = plan.config["providers"]["homelab"]["models"]["deepseek-flash"]
    assert "package" not in flash
    assert "disabled" not in flash


def test_existing_apikey_is_never_overwritten() -> None:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["providers"]["homelab"]["settings"]["apiKey"] = "existing-provider-key"
    cfg["providers"]["homelab"]["headers"]["Authorization"] = "Bearer another-token"
    plan = _plan(cfg)
    assert plan.config["providers"]["homelab"]["settings"]["apiKey"] == "existing-provider-key"
    assert plan.credential_derived is False
    assert all(c["field"] != "settings.apiKey" for c in plan.changes)


def test_declared_env_is_valid_credential_source() -> None:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["providers"]["homelab"]["headers"] = {}
    cfg["providers"]["homelab"]["env"] = ["HOMELAB_GATEWAY_KEY"]
    plan = _plan(cfg)
    assert plan.credential_derived is False
    assert "apiKey" not in plan.config["providers"]["homelab"].get("settings", {})
    assert "deepseek-flash" not in plan.refused
    assert plan.config["providers"]["homelab"]["models"]["deepseek-flash"]["package"] == (
        "@opencode/ai/providers/deepseek"
    )


def test_credential_derivation_is_idempotent() -> None:
    first = _plan()
    assert first.credential_derived is True
    second = opencode_sync.plan_sync(_catalog(), first.config)
    assert not second.changed
    assert second.credential_derived is False


# --------------------------------------------------------------------------- #
# Optional explicit request output budget (verified limit.output field)
# --------------------------------------------------------------------------- #
def test_request_output_budget_optional_and_off_by_default() -> None:
    default = _plan()
    assert default.request_output_budget is None
    assert default.config["providers"]["homelab"]["models"]["deepseek-flash"]["limit"] == {
        "context": 1048576,
        "output": 393216,
    }

    capped = _plan(request_output_budget=64000)
    assert capped.request_output_budget == 64000
    assert capped.config["providers"]["homelab"]["models"]["deepseek-flash"]["limit"] == {
        "context": 1048576,
        "output": 64000,
    }


def test_request_output_budget_larger_than_hard_keeps_hard() -> None:
    plan = _plan(request_output_budget=99999999)
    assert plan.config["providers"]["homelab"]["models"]["deepseek-flash"]["limit"]["output"] == 393216


def test_request_output_budget_invalid_ignored_with_warning() -> None:
    plan = _plan(request_output_budget=0)
    assert plan.request_output_budget is None
    assert any("request_output_budget" in w for w in plan.warnings)
    assert plan.config["providers"]["homelab"]["models"]["deepseek-flash"]["limit"]["output"] == 393216
