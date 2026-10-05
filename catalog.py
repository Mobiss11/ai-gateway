"""Каталог метаданных моделей AI Gateway.

Единый проверяемый словарь (source + timestamp), который используют:

* ``/v1/models`` шлюза (обогащение ответа);
* CLI-экспорт метаданных;
* синхронизация конфига OpenCode (``opencode_sync.py``).

Принципы:

* отсутствующие/битые/отрицательные значения не превращаются в «точные»
  дефолты — они остаются ``None`` (unknown);
* отрицательные цены (маршрутизатор OpenRouter отдаёт ``-1``) — это unknown,
  а не ``0`` и не «2M бесплатно»;
* формат совместим с V2-конфигом OpenCode: ``limit``, ``capabilities``,
  ``variants``, ``cost``, ``compatibility``, ``package``.

Модуль не делает сетевых запросов. Живое обновление — в ``metadata_refresh``.
"""

from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CATALOG_VERSION = 1

# Проверенные (по CLI OpenCode 2.0.21) нативные пакеты провайдеров.
# Для них формы variants/reasoning совпадают с генераторами OpenCode.
KNOWN_PACKAGES: dict[str, str] = {
    "deepseek": "@opencode/ai/providers/deepseek",
    "openrouter": "@opencode/ai/providers/openrouter",
}

# Пакеты, у которых reasoning задаётся как ``settings.reasoning.effort``.
_OPENROUTER_REASONING = "@opencode/ai/providers/openrouter"
_DEEPSEEK_REASONING = "@opencode/ai/providers/deepseek"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Валидация значений
# --------------------------------------------------------------------------- #
def positive_int(value: Any) -> int | None:
    """Целое > 0 или None. bool/строки/NaN/inf не считаются числами."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    ivalue = int(value)
    return ivalue if ivalue > 0 and ivalue == value else None


def per_million(value: Any) -> float | None:
    """Цена за токен (строка/число) → USD за 1M токенов.

    Отрицательные и нечисловые значения → None (unknown), не 0.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if math.isnan(number) or math.isinf(number) or number < 0:
        return None
    return number * 1_000_000


def nonneg_number(value: Any) -> float | None:
    """Число >= 0 (native-цены OpenCode уже за 1M токенов)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if math.isnan(number) or math.isinf(number) or number < 0:
        return None
    return number


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _positive_tier(value: Any) -> int | None:
    return positive_int(value)


# --------------------------------------------------------------------------- #
# Нормализация upstream-метаданных
# --------------------------------------------------------------------------- #
def find_model(entries: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == model_id or entry.get("modelID") == model_id:
            return entry
    return None


def _find_native(
    native_models: list[dict[str, Any]], upstream: str, model_id: str
) -> dict[str, Any] | None:
    """Нативная запись OpenCode по (providerID, modelID/id)."""
    for entry in native_models:
        if not isinstance(entry, dict):
            continue
        if entry.get("providerID") != upstream:
            continue
        if entry.get("modelID") == model_id or entry.get("id") == model_id:
            return entry
    return None


def _is_router(entry: dict[str, Any]) -> bool:
    model_id = str(entry.get("id") or "")
    tokenizer = str((entry.get("architecture") or {}).get("tokenizer") or "").lower()
    pricing = entry.get("pricing") or {}
    prompt_price = pricing.get("prompt") if isinstance(pricing, dict) else None
    if "router" in tokenizer or "router" in model_id.lower():
        return True
    return prompt_price == "-1" or prompt_price == -1


def _normalize_compat(native: dict[str, Any] | None) -> dict[str, Any] | None:
    if not native:
        return None
    compat = native.get("compatibility")
    if isinstance(compat, dict) and compat:
        return copy.deepcopy(compat)
    return None


def _normalize_native_cost(cost: Any) -> list[dict[str, Any]] | None:
    if not isinstance(cost, list):
        return None
    result: list[dict[str, Any]] = []
    for tier in cost:
        if not isinstance(tier, dict):
            continue
        price_in = nonneg_number(tier.get("input"))
        price_out = nonneg_number(tier.get("output"))
        if price_in is None or price_out is None:
            continue
        cache = tier.get("cache") if isinstance(tier.get("cache"), dict) else {}
        base: dict[str, Any] = {
            "input": price_in,
            "output": price_out,
            "cache": {
                "read": nonneg_number(cache.get("read")),
                "write": nonneg_number(cache.get("write")),
            },
        }
        tier_meta = tier.get("tier")
        size = _positive_tier(tier_meta.get("size")) if isinstance(tier_meta, dict) else None
        if size is not None:
            base["tier"] = {"type": "context", "size": size}
        result.append(base)
    return result or None


def _normalize_openrouter_cost(pricing: Any) -> list[dict[str, Any]] | None:
    if not isinstance(pricing, dict):
        return None
    price_in = per_million(pricing.get("prompt"))
    price_out = per_million(pricing.get("completion"))
    tiers: list[dict[str, Any]] = []
    for override in pricing.get("overrides") or []:
        if not isinstance(override, dict):
            continue
        size = _positive_tier(override.get("min_prompt_tokens"))
        o_in = per_million(override.get("prompt"))
        o_out = per_million(override.get("completion"))
        if size is None or o_in is None or o_out is None:
            continue
        tiers.append(
            {
                "tier": {"type": "context", "size": size},
                "input": o_in,
                "output": o_out,
                "cache": {
                    "read": per_million(override.get("input_cache_read")),
                    "write": per_million(override.get("input_cache_write")),
                },
            }
        )
    if price_in is None or price_out is None:
        if not tiers:
            return None
        return tiers
    base = {
        "input": price_in,
        "output": price_out,
        "cache": {
            "read": per_million(pricing.get("input_cache_read")),
            "write": per_million(pricing.get("input_cache_write")),
        },
    }
    return [base, *tiers]


def _deepseek_variants(reasoning: dict[str, Any]) -> list[dict[str, Any]]:
    """Форма variants пакета ``@opencode/ai/providers/deepseek`` (CLI 2.0.21)."""
    levels = [x for x in reasoning.get("supported_efforts") or [] if isinstance(x, str)]
    variants: list[dict[str, Any]] = [
        {"id": "none", "body": {"thinking": {"type": "disabled"}}}
    ]
    for level in levels:
        if level == "none":
            continue
        variants.append(
            {
                "id": level,
                "settings": {"reasoningEffort": level},
                "body": {"thinking": {"type": "enabled"}},
            }
        )
    return variants


def _openrouter_variants(reasoning: dict[str, Any]) -> list[dict[str, Any]]:
    """Форма variants пакета ``@opencode/ai/providers/openrouter`` (CLI 2.0.21)."""
    levels = [x for x in reasoning.get("supported_efforts") or [] if isinstance(x, str)]
    variants: list[dict[str, Any]] = []
    for level in levels:
        variants.append({"id": level, "settings": {"reasoning": {"effort": level}}})
    return variants


def normalize_upstream_models(
    upstream: str,
    payload: Any,
    native_models: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Список нормализованных записей для одного upstream.

    Поддерживаются форматы: DeepSeek ``/v1/models`` (расширенный audited-набор
    c ``context_window``/``effort``) и OpenRouter ``/api/v1/models``. Формат
    определяется по полям записи, а не по имени upstream.
    """
    entries = _extract_entries(payload)
    native = native_models or []
    result: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id") or entry.get("modelID")
        if not isinstance(model_id, str) or not model_id:
            continue
        if "pricing" in entry or "top_provider" in entry or "architecture" in entry:
            result.append(_normalize_openrouter_entry(upstream, entry, native, model_id))
        else:
            result.append(_normalize_deepseek_entry(upstream, entry, native, model_id))
    return result


def _extract_entries(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
        models = payload.get("models")
        if isinstance(models, list):
            return models
        # dict-of-models форма
        return list(payload.values())
    if isinstance(payload, list):
        return payload
    return []


def _caps_complete(capabilities: Any) -> bool:
    if not isinstance(capabilities, dict):
        return False
    if capabilities.get("tools") is None:
        return False
    return bool(capabilities.get("input")) and bool(capabilities.get("output"))


def _quality(limit: dict[str, Any], capabilities: Any = None) -> str:
    """Качество записи. ``exact`` требует полных limit И capabilities."""
    known = sum(
        1 for key in ("context", "output") if positive_int(limit.get(key)) is not None
    )
    caps_ok = _caps_complete(capabilities)
    if known == 2 and caps_ok:
        return "exact"
    if known >= 1 or caps_ok:
        return "partial"
    return "unknown"


def _valid_default(default: Any, levels: list[str]) -> str | None:
    """default_effort допустим только если входит в supported_efforts."""
    if not isinstance(default, str) or not default:
        return None
    if levels and default not in levels:
        return None
    return default


def _provenance(upstream: str) -> dict[str, Any]:
    return {
        "source": upstream,
        "source_type": None,
        "fetched_at": None,
        "processed_at": None,
        "stale": False,
        "retained_fields": [],
    }


def _normalize_deepseek_entry(
    upstream: str,
    entry: dict[str, Any],
    native_models: list[dict[str, Any]],
    model_id: str,
) -> dict[str, Any]:
    native = _find_native(native_models, upstream, model_id)
    limit = {
        "context": positive_int(entry.get("context_window")),
        "output": positive_int(entry.get("max_output_tokens")),
    }
    effort = entry.get("effort") if isinstance(entry.get("effort"), dict) else {}
    levels = _string_list(effort.get("supported_levels"))
    reasoning: dict[str, Any] | None
    if levels or effort.get("default_level"):
        reasoning = {
            "supported_efforts": levels,
            "default_effort": _valid_default(effort.get("default_level"), levels),
            "mandatory": False,
        }
    else:
        reasoning = None

    native_caps = (native or {}).get("capabilities")
    input_modalities = _string_list(entry.get("input_modalities"))
    output_modalities = _string_list(entry.get("output_modalities"))
    capabilities: dict[str, Any] | None = None
    if input_modalities or output_modalities or isinstance(native_caps, dict):
        caps_source = native_caps if isinstance(native_caps, dict) else {}
        capabilities = {
            "tools": bool(caps_source.get("tools")) if caps_source else None,
            # live-модальности важнее устаревших native
            "input": input_modalities or list(caps_source.get("input") or []),
            "output": output_modalities or list(caps_source.get("output") or []),
        }

    package = (native or {}).get("package") or KNOWN_PACKAGES.get(upstream)
    compatibility = _normalize_compat(native)
    if compatibility is None and package == _DEEPSEEK_REASONING:
        compatibility = {"reasoningField": "reasoning_content"}

    # Живые цены (если появились) важнее native; иначе native fallback.
    cost = _normalize_openrouter_cost(entry.get("pricing"))
    if cost is None:
        cost = _normalize_native_cost((native or {}).get("cost"))

    variants: list[dict[str, Any]] | None = None
    native_variants = (native or {}).get("variants")
    if package == _DEEPSEEK_REASONING and reasoning and reasoning["supported_efforts"]:
        # live supported_efforts важнее устаревших native variants
        variants = _deepseek_variants(reasoning)
    elif isinstance(native_variants, list) and native_variants:
        variants = copy.deepcopy(native_variants)

    record = {
        "id": model_id,
        "upstream": upstream,
        "upstream_model": model_id,
        "name": entry.get("name") if isinstance(entry.get("name"), str) else None,
        "limit": limit,
        "capabilities": capabilities,
        "reasoning": reasoning,
        "cost": cost,
        "compatibility": compatibility,
        "package": package if isinstance(package, str) else None,
        "variants": variants,
        "owner": entry.get("owned_by") if isinstance(entry.get("owned_by"), str) else None,
        "metadata_quality": _quality(limit, capabilities),
        "recommended_disabled": False,
        "api_capabilities": copy.deepcopy(entry.get("api_capabilities"))
        if isinstance(entry.get("api_capabilities"), dict)
        else None,
    }
    record.update(_provenance(upstream))
    return record


def _normalize_openrouter_entry(
    upstream: str,
    entry: dict[str, Any],
    native_models: list[dict[str, Any]],
    model_id: str,
) -> dict[str, Any]:
    native = _find_native(native_models, upstream, model_id)
    router = _is_router(entry)
    top_provider = entry.get("top_provider") if isinstance(entry.get("top_provider"), dict) else {}
    declared_context = positive_int(entry.get("context_length"))
    if router:
        # Маршрутизатор: границы обслуживания неизвестны, 2M/0$ не гарантируем.
        limit = {"context": None, "output": None, "context_declared": declared_context}
    else:
        limit = {
            "context": positive_int(top_provider.get("context_length")) or declared_context,
            "output": positive_int(top_provider.get("max_completion_tokens")),
        }

    architecture = entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
    params = _string_list(entry.get("supported_parameters"))
    input_modalities = _string_list(architecture.get("input_modalities"))
    output_modalities = _string_list(architecture.get("output_modalities"))
    capabilities: dict[str, Any] | None = None
    if input_modalities or output_modalities or params:
        capabilities = {
            "tools": ("tools" in params) if params else None,
            "input": input_modalities,
            "output": output_modalities,
        }

    raw_reasoning = entry.get("reasoning") if isinstance(entry.get("reasoning"), dict) else {}
    levels = _string_list(raw_reasoning.get("supported_efforts"))
    reasoning: dict[str, Any] | None = None
    if levels or raw_reasoning.get("default_effort"):
        reasoning = {
            "supported_efforts": levels,
            "default_effort": _valid_default(raw_reasoning.get("default_effort"), levels),
            "mandatory": bool(raw_reasoning["mandatory"])
            if "mandatory" in raw_reasoning
            else None,
            "enabled_by_default": bool(raw_reasoning["default_enabled"])
            if "default_enabled" in raw_reasoning
            else None,
        }

    package = (native or {}).get("package") or KNOWN_PACKAGES.get(upstream)
    compatibility = _normalize_compat(native)
    # Живые цены OpenRouter важнее native; native — только fallback.
    cost = None if router else _normalize_openrouter_cost(entry.get("pricing"))
    if cost is None:
        cost = _normalize_native_cost((native or {}).get("cost"))

    variants: list[dict[str, Any]] | None = None
    native_variants = (native or {}).get("variants")
    if not router and package == _OPENROUTER_REASONING and reasoning and levels:
        # live supported_efforts важнее устаревших native variants
        variants = _openrouter_variants(reasoning)
    elif not router and isinstance(native_variants, list) and native_variants:
        variants = copy.deepcopy(native_variants)

    record = {
        "id": model_id,
        "upstream": upstream,
        "upstream_model": model_id,
        "name": entry.get("name") if isinstance(entry.get("name"), str) else None,
        "limit": limit,
        "capabilities": capabilities,
        "reasoning": reasoning,
        "cost": cost,
        "compatibility": compatibility,
        "package": package if isinstance(package, str) else None,
        "variants": variants,
        "owner": None,
        "metadata_quality": "unknown" if router else _quality(limit, capabilities),
        "recommended_disabled": router,
    }
    record.update(_provenance(upstream))
    return record


# --------------------------------------------------------------------------- #
# Каталог целиком
# --------------------------------------------------------------------------- #
def empty_catalog() -> dict[str, Any]:
    return {
        "version": CATALOG_VERSION,
        "generated_at": None,
        "sources": {},
        "models": {},
    }


def build_alias_entries(
    aliases: dict[str, Any],
    normalized: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """alias → запись каталога по ``upstream``/``model`` из конфига шлюза."""
    models: dict[str, dict[str, Any]] = {}
    for alias, spec in aliases.items():
        if not isinstance(spec, dict):
            continue
        upstream = spec.get("upstream")
        model = spec.get("model")
        if not isinstance(upstream, str) or not isinstance(model, str):
            continue
        entry = find_model(normalized.get(upstream, []), model)
        if entry is None:
            continue
        record = copy.deepcopy(entry)
        record["alias"] = alias
        record["name"] = spec.get("name") or record.get("name")
        record["gateway_name"] = spec.get("name")
        models[alias] = record
    return models


def load_catalog(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Читает каталог. Отсутствие/битый файл → пустой каталог (не падаем)."""
    if not path:
        return empty_catalog()
    catalog_path = Path(path).expanduser()
    if not catalog_path.is_file():
        return empty_catalog()
    try:
        with catalog_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return empty_catalog()
    catalog = empty_catalog()
    if isinstance(data, dict):
        if isinstance(data.get("sources"), dict):
            catalog["sources"] = data["sources"]
        if isinstance(data.get("models"), dict):
            catalog["models"] = data["models"]
        if isinstance(data.get("generated_at"), str):
            catalog["generated_at"] = data["generated_at"]
        if isinstance(data.get("version"), int):
            catalog["version"] = data["version"]
    return catalog


def save_catalog(path: str | os.PathLike[str], catalog: dict[str, Any]) -> None:
    """Атомарно сохраняет каталог JSON."""
    catalog_path = Path(path).expanduser()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=False, allow_nan=False)
        + "\n"
    )
    tmp = catalog_path.with_name(catalog_path.name + f".tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, catalog_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def catalog_entry(catalog: dict[str, Any] | None, alias: str) -> dict[str, Any] | None:
    if not isinstance(catalog, dict):
        return None
    models = catalog.get("models")
    if not isinstance(models, dict):
        return None
    entry = models.get(alias)
    return entry if isinstance(entry, dict) else None


def entry_route(entry: dict[str, Any] | None) -> tuple[Any, Any]:
    if not isinstance(entry, dict):
        return (None, None)
    return (entry.get("upstream"), entry.get("upstream_model"))


def route_matches(entry: dict[str, Any] | None, upstream: Any, model: Any) -> bool:
    return entry_route(entry) == (upstream, model)


def _is_missing(value: Any) -> bool:
    return value is None or value == [] or value == {}


def merge_entry(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Field-level merge new-поверх-old с сохранением валидных старых полей.

    Старые значения удерживаются только если alias всё ещё роутится в тот же
    upstream/модель. Удержанные поля помечаются ``stale``/``retained_fields``.
    Отсутствующие в новых данных поля сохраняются, но реально изменившиеся
    наборы (например supported_efforts) принимаются как есть.
    """
    merged = copy.deepcopy(new)
    retained: list[str] = []
    if not isinstance(old, dict) or entry_route(old) != entry_route(new):
        merged["stale"] = False
        merged["retained_fields"] = []
        return merged

    for field in ("name", "package", "compatibility", "cost", "variants"):
        if _is_missing(merged.get(field)) and not _is_missing(old.get(field)):
            merged[field] = copy.deepcopy(old[field])
            retained.append(field)

    old_lim = old.get("limit") if isinstance(old.get("limit"), dict) else {}
    new_lim = merged.get("limit") if isinstance(merged.get("limit"), dict) else {}
    for key in ("context", "output", "context_declared"):
        if new_lim.get(key) is None and old_lim.get(key) is not None:
            new_lim[key] = old_lim[key]
            retained.append(f"limit.{key}")
    merged["limit"] = new_lim

    old_caps = old.get("capabilities") if isinstance(old.get("capabilities"), dict) else {}
    new_caps = merged.get("capabilities") if isinstance(merged.get("capabilities"), dict) else {}
    if old_caps:
        merged_caps = dict(new_caps)
        if merged_caps.get("tools") is None and old_caps.get("tools") is not None:
            merged_caps["tools"] = old_caps["tools"]
            retained.append("capabilities.tools")
        if not merged_caps.get("input") and old_caps.get("input"):
            merged_caps["input"] = list(old_caps["input"])
            retained.append("capabilities.input")
        if not merged_caps.get("output") and old_caps.get("output"):
            merged_caps["output"] = list(old_caps["output"])
            retained.append("capabilities.output")
        merged["capabilities"] = merged_caps or None

    old_reasoning = old.get("reasoning") if isinstance(old.get("reasoning"), dict) else None
    new_reasoning = merged.get("reasoning") if isinstance(merged.get("reasoning"), dict) else None
    if old_reasoning:
        if new_reasoning is None:
            merged["reasoning"] = copy.deepcopy(old_reasoning)
            retained.append("reasoning")
        else:
            merged_reasoning = dict(new_reasoning)
            if not merged_reasoning.get("supported_efforts") and old_reasoning.get(
                "supported_efforts"
            ):
                merged_reasoning["supported_efforts"] = list(
                    old_reasoning["supported_efforts"]
                )
                retained.append("reasoning.supported_efforts")
            if (
                merged_reasoning.get("default_effort") is None
                and old_reasoning.get("default_effort")
            ):
                merged_reasoning["default_effort"] = old_reasoning["default_effort"]
                retained.append("reasoning.default_effort")
            merged_reasoning["default_effort"] = _valid_default(
                merged_reasoning.get("default_effort"),
                merged_reasoning.get("supported_efforts") or [],
            )
            merged["reasoning"] = merged_reasoning

    merged["metadata_quality"] = _quality(
        merged.get("limit") or {}, merged.get("capabilities")
    )
    if merged.get("recommended_disabled"):
        merged["metadata_quality"] = "unknown"
    merged["stale"] = bool(retained)
    merged["retained_fields"] = retained
    return merged


def compute_budgets(limit: dict[str, Any] | None, *, output_cap: int = 131072) -> dict[str, Any]:
    """Ориентиры бюджета. Advisory only: шлюз/синк эти лимиты не применяют.

    ``limit.output`` — жёсткий максимум upstream, а не рекомендованный размер
    одного запроса; OpenCode сам решает, сколько вывода запрашивать.
    ``usable_input`` — контекст минус зарезервированный вывод.
    """
    limit = limit or {}
    context = positive_int(limit.get("context"))
    output = positive_int(limit.get("output"))
    if context is None:
        return {"usable_input": None, "recommended_output": None, "hard_output": output}
    reserved = min(output, output_cap) if output is not None else output_cap
    reserved = max(1, min(reserved, context - 1)) if context > 1 else 1
    return {
        "usable_input": context - reserved,
        "recommended_output": reserved,
        "hard_output": output,
    }
