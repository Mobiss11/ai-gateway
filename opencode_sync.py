"""Синхронизация каталога метаданных в V2-конфиг OpenCode.

Scope синхронизации жёстко ограничен:

* только провайдер ``homelab`` (по умолчанию) и только модели, уже описанные
  в его ``models`` (плюс явный ``--add-missing``);
* управляемые поля: ``name``, ``package``, ``compatibility``, ``capabilities``,
  ``limit``, ``cost``, ``variants`` и (опционально) ``disabled: true``;
* слияние, а не замена: пользовательские variants/settings и лишние ключи
  сохраняются; ``modelID``/``id`` не подменяются;
* маршрут проверяется по gateway-алиасам: если alias перенаправлен, устаревшие
  метаданные не применяются;
* явный ``modelID``, отличный от alias, приводит к отказу от синка модели
  (иначе шлюз получит upstream-id и не сроутит alias);
* чужие провайдеры, агенты, настройки, заголовки и секреты не трогаются.

По умолчанию — dry-run. Применение только по явному флагу: защищённая
timestamped-копия 0600 + атомарная замена. Секреты никогда не печатаются.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANAGED_PROVIDER_DEFAULT = "homelab"

# Per-model native packages, для которых OpenCode нужен provider.settings.apiKey,
# иначе модель выпадает из /api/model.
NATIVE_MODEL_PACKAGES = frozenset(
    {
        "@opencode/ai/providers/deepseek",
        "@opencode/ai/providers/openrouter",
    }
)

MANAGED_MODEL_FIELDS = (
    "name",
    "package",
    "compatibility",
    "capabilities",
    "limit",
    "cost",
    "variants",
)

SENSITIVE_KEY_HINTS = (
    "authorization",
    "api_key",
    "apikey",
    "api-key",
    "token",
    "secret",
    "password",
    "credential",
)

_TOKEN_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9._-]{6,})")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def redact_text(value: str) -> str:
    value = _TOKEN_RE.sub(r"\1<redacted>", value)
    value = _KEY_RE.sub("<redacted>", value)
    return value


def redact_value(value: Any, key: str | None = None) -> Any:
    """Глубокая замена секретов для отображения. Не для записи конфига."""
    if key is not None and any(hint in key.lower() for hint in SENSITIVE_KEY_HINTS):
        return "<redacted>"
    if isinstance(value, dict):
        return {k: redact_value(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


@dataclass
class SyncPlan:
    provider: str
    changes: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    route_unverified: bool = False
    credential_derived: bool = False
    request_output_budget: int | None = None
    config: dict[str, Any] | None = None

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "route_verified": not self.route_unverified,
            "credential_derived": self.credential_derived,
            "request_output_budget": self.request_output_budget,
            "changed_models": sorted({c["model"] for c in self.changes}),
            "refused_models": sorted(self.refused),
            "changes": self.changes,
            "warnings": list(self.warnings),
        }


def _bearer_token_from_headers(headers: Any) -> str | None:
    """Case-insensitive ``Authorization: Bearer <token>`` из provider.headers."""
    if not isinstance(headers, dict):
        return None
    for key, value in headers.items():
        if not isinstance(key, str) or key.strip().lower() != "authorization":
            continue
        if not isinstance(value, str):
            continue
        match = re.match(r"(?i)^\s*bearer\s+(.+?)\s*$", value)
        if match and match.group(1):
            return match.group(1)
    return None


def _clean_capabilities(caps: Any) -> dict[str, Any] | None:
    if not isinstance(caps, dict):
        return None
    tools = caps.get("tools")
    inputs = [x for x in caps.get("input") or [] if isinstance(x, str) and x]
    outputs = [x for x in caps.get("output") or [] if isinstance(x, str) and x]
    if tools is None or not inputs or not outputs:
        return None
    return {"tools": bool(tools), "input": inputs, "output": outputs}


def _clean_limit(limit: Any) -> dict[str, int] | None:
    if not isinstance(limit, dict):
        return None
    context = limit.get("context")
    output = limit.get("output")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        return None
    if isinstance(output, bool) or not isinstance(output, int) or output <= 0:
        return None
    return {"context": context, "output": output}


def _is_price(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _clean_cost(cost: Any) -> list[dict[str, Any]] | None:
    if not isinstance(cost, list):
        return None
    result: list[dict[str, Any]] = []
    for tier in cost:
        if not isinstance(tier, dict):
            continue
        price_in = tier.get("input")
        price_out = tier.get("output")
        if not _is_price(price_in) or not _is_price(price_out):
            continue
        item: dict[str, Any] = {"input": price_in, "output": price_out}
        tier_meta = tier.get("tier")
        if (
            isinstance(tier_meta, dict)
            and tier_meta.get("type") == "context"
            and isinstance(tier_meta.get("size"), int)
            and tier_meta["size"] > 0
        ):
            item["tier"] = {"type": "context", "size": tier_meta["size"]}
        cache = tier.get("cache") if isinstance(tier.get("cache"), dict) else {}
        clean_cache = {
            k: cache[k] for k in ("read", "write") if _is_price(cache.get(k))
        }
        if clean_cache:
            item["cache"] = clean_cache
        result.append(item)
    return result or None


def _clean_variants(variants: Any) -> list[dict[str, Any]] | None:
    if not isinstance(variants, list):
        return None
    result: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        vid = variant.get("id")
        if not isinstance(vid, str) or not vid:
            continue
        clean: dict[str, Any] = {"id": vid}
        for key in ("settings", "body", "headers"):
            if isinstance(variant.get(key), dict) and variant[key]:
                clean[key] = copy.deepcopy(variant[key])
        result.append(clean)
    return result or None


def _merge_dict(base: Any, overlay: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивный merge overlay-поверх-base; base-ключи сохраняются."""
    result: dict[str, Any] = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_variants(existing: Any, managed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Слияние по id: пользовательские variants и их поля сохраняются."""
    existing_list = [
        v
        for v in (existing if isinstance(existing, list) else [])
        if isinstance(v, dict) and isinstance(v.get("id"), str) and v["id"]
    ]
    by_id = {v["id"]: copy.deepcopy(v) for v in existing_list}
    order = [v["id"] for v in existing_list]
    for variant in managed:
        vid = variant["id"]
        if vid in by_id:
            by_id[vid] = _merge_dict(by_id[vid], variant)
        else:
            by_id[vid] = copy.deepcopy(variant)
            order.append(vid)
    return [by_id[i] for i in order if i in by_id]


def _set_field(target: dict[str, Any], key: str, value: Any, model: str, changes: list) -> None:
    if value is None:
        return
    if target.get(key) == value:
        return
    changes.append({"model": model, "field": key, "value": redact_value(value)})
    target[key] = value


def plan_sync(
    catalog: dict[str, Any],
    config: dict[str, Any],
    *,
    provider: str = MANAGED_PROVIDER_DEFAULT,
    add_missing: bool = False,
    disable_recommended: bool = False,
    gateway_aliases: dict[str, Any] | None = None,
    request_output_budget: int | None = None,
) -> SyncPlan:
    """Планирует синк. ``gateway_aliases`` — ``config.models`` шлюза для проверки
    маршрута; без него синк помечается как route_unverified.

    ``request_output_budget`` — явный opt-in: ограничивает `limit.output`
    модели (verified-поле), не трогая прочие настройки. По умолчанию не задан.
    """
    plan = SyncPlan(provider=provider, config=copy.deepcopy(config))
    if request_output_budget is not None and (
        isinstance(request_output_budget, bool)
        or not isinstance(request_output_budget, int)
        or request_output_budget <= 0
    ):
        plan.warnings.append("request_output_budget must be a positive integer; ignored")
        request_output_budget = None
    plan.request_output_budget = request_output_budget
    providers = plan.config.get("providers")
    if not isinstance(providers, dict) or provider not in providers:
        plan.warnings.append(f"provider '{provider}' not found; nothing to sync")
        return plan
    provider_cfg = providers[provider]
    if not isinstance(provider_cfg, dict):
        plan.warnings.append(f"provider '{provider}' is not an object; nothing to sync")
        return plan
    models = provider_cfg.setdefault("models", {})
    if not isinstance(models, dict):
        plan.warnings.append(f"provider '{provider}'.models is not an object; nothing to sync")
        return plan

    settings = provider_cfg.get("settings")
    if not isinstance(settings, dict):
        settings = None
    existing_api_key = settings.get("apiKey") if settings is not None else None
    # Источники credential: settings.apiKey > declared env > Authorization header.
    credential_source: str | None = None
    if isinstance(existing_api_key, str) and existing_api_key:
        credential_source = "settings.apiKey"
    elif isinstance(provider_cfg.get("env"), list) and provider_cfg["env"]:
        credential_source = "env"
    header_token = _bearer_token_from_headers(provider_cfg.get("headers"))

    entries = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(entries, dict):
        plan.warnings.append("catalog has no models")
        return plan

    if not isinstance(gateway_aliases, dict):
        plan.route_unverified = True
        plan.warnings.append("gateway alias map not provided; route match not verified")

    for alias, entry in entries.items():
        if not isinstance(entry, dict):
            continue

        if isinstance(gateway_aliases, dict):
            spec = gateway_aliases.get(alias)
            if not isinstance(spec, dict):
                plan.warnings.append(f"model '{alias}': alias not in gateway config; skipped")
                plan.refused.append(alias)
                continue
            if (spec.get("upstream"), spec.get("model")) != (
                entry.get("upstream"),
                entry.get("upstream_model"),
            ):
                plan.warnings.append(
                    f"model '{alias}': catalog route mismatch "
                    f"(gateway -> {spec.get('upstream')}/{spec.get('model')}); skipped"
                )
                plan.refused.append(alias)
                continue

        if alias not in models:
            if not add_missing:
                continue
            models[alias] = {}
        target = models[alias]
        if not isinstance(target, dict):
            plan.warnings.append(f"model '{alias}' is not an object; skipped")
            continue

        model_id = target.get("modelID")
        if isinstance(model_id, str) and model_id and model_id != alias:
            plan.warnings.append(
                f"model '{alias}': explicit modelID '{model_id}' != alias; "
                f"refusing to sync (would misroute gateway)"
            )
            plan.refused.append(alias)
            continue

        # Нативный per-model пакет требует provider.settings.apiKey, иначе
        # OpenCode выбрасывает модель из /api/model.
        desired_package = entry.get("package")
        native_switch = (
            isinstance(desired_package, str)
            and desired_package in NATIVE_MODEL_PACKAGES
            and target.get("package") != desired_package
        )
        if native_switch and credential_source is None:
            if header_token:
                provider_cfg.setdefault("settings", {})["apiKey"] = header_token
                credential_source = "derived-from-Authorization"
                plan.credential_derived = True
                plan.changes.append(
                    {
                        "model": provider,
                        "field": "settings.apiKey",
                        "value": "<redacted>",
                    }
                )
            else:
                plan.warnings.append(
                    f"model '{alias}': cannot switch to native package "
                    f"'{desired_package}' — provider '{provider}' has no apiKey/env/"
                    "Authorization Bearer credential; refusing (model left as-is)"
                )
                plan.refused.append(alias)
                continue

        _set_field(
            target,
            "name",
            entry.get("name") if isinstance(entry.get("name"), str) else None,
            alias,
            plan.changes,
        )
        _set_field(
            target,
            "package",
            desired_package if isinstance(desired_package, str) else None,
            alias,
            plan.changes,
        )

        compat = entry.get("compatibility")
        if isinstance(compat, dict) and compat:
            _set_field(
                target,
                "compatibility",
                _merge_dict(target.get("compatibility"), compat),
                alias,
                plan.changes,
            )

        caps = _clean_capabilities(entry.get("capabilities"))
        if caps is None:
            if entry.get("capabilities") is not None:
                plan.warnings.append(
                    f"model '{alias}': capabilities incomplete/unknown; left unchanged"
                )
        else:
            _set_field(
                target,
                "capabilities",
                _merge_dict(target.get("capabilities"), caps),
                alias,
                plan.changes,
            )

        limit = _clean_limit(entry.get("limit"))
        if request_output_budget is not None:
            # Явный opt-in: ограничиваем verified-поле limit.output.
            base_limit = dict(limit) if limit else {}
            known_output = base_limit.get("output")
            base_limit["output"] = (
                request_output_budget
                if known_output is None
                else min(known_output, request_output_budget)
            )
            limit = base_limit
            _set_field(
                target,
                "limit",
                _merge_dict(target.get("limit"), limit),
                alias,
                plan.changes,
            )
        elif limit is None:
            plan.warnings.append(
                f"model '{alias}': limit unknown/incomplete; left unchanged (not precise)"
            )
        else:
            _set_field(
                target,
                "limit",
                _merge_dict(target.get("limit"), limit),
                alias,
                plan.changes,
            )

        cost = _clean_cost(entry.get("cost"))
        if cost is None and entry.get("cost"):
            plan.warnings.append(f"model '{alias}': cost incomplete; left unchanged")
        _set_field(target, "cost", cost, alias, plan.changes)

        variants = _clean_variants(entry.get("variants"))
        if variants is None:
            if entry.get("reasoning") is None and target.get("variants"):
                plan.warnings.append(
                    f"model '{alias}': reasoning metadata unknown; existing variants left unchanged"
                )
        else:
            _set_field(
                target,
                "variants",
                _merge_variants(target.get("variants"), variants),
                alias,
                plan.changes,
            )

        if disable_recommended and entry.get("recommended_disabled"):
            _set_field(target, "disabled", True, alias, plan.changes)

    return plan


def _serialize(config: dict[str, Any]) -> bytes:
    return (
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def apply_sync(
    target_path: str | os.PathLike[str],
    config: dict[str, Any],
) -> Path:
    """Атомарно записывает конфиг, предварительно создав backup 0600."""
    path = Path(target_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"OpenCode config not found: {path}")
    payload = _serialize(config)
    backup = path.with_name(f"{path.name}.{_timestamp()}.bak")
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as out, path.open("rb") as src:
            shutil.copyfileobj(src, out)
    except BaseException:
        try:
            backup.unlink()
        except OSError:
            pass
        raise
    try:
        os.chmod(backup, 0o600)
    except OSError:
        pass

    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return backup


def load_opencode_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"OpenCode config root must be a JSON object: {config_path}")
    return data
