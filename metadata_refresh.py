"""Явное обновление каталога метаданных.

Только metadata-only discovery: HTTP GET ``{base_url}/models`` по каждому
upstream из allowlist (он же список настроенных upstream-ов конфига шлюза).
Никаких chat/LLM-запросов, никаких «сходить в интернет на каждый чат».

* таймаут ограничен;
* при сбое обновления сохраняется last-known-good;
* битые/неполные метаданные не заменяют точные значения дефолтами.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from catalog import (
    build_alias_entries,
    empty_catalog,
    merge_entry,
    normalize_upstream_models,
    now_utc_iso,
    positive_int,
)

DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass
class RefreshReport:
    upstreams: dict[str, dict[str, Any]] = field(default_factory=dict)
    models_updated: list[str] = field(default_factory=list)
    models_missing: list[str] = field(default_factory=list)
    models_pruned: list[str] = field(default_factory=list)
    kept_last_known_good: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "upstreams": self.upstreams,
            "models_updated": sorted(self.models_updated),
            "models_missing": sorted(self.models_missing),
            "models_pruned": sorted(self.models_pruned),
            "kept_last_known_good": sorted(self.kept_last_known_good),
        }


def source_allowlist(config: dict[str, Any]) -> list[str]:
    """Allowlist источников = настроенные upstream-ы (без сторонних хостов).

    Upstream'ы с ``api_format: "anthropic"`` (например kraube serve) не
    отдают OpenAI-``GET /models`` — их каталог не опрашивает; их модели
    живут в конфиге, а не в живом каталоге.
    """
    upstreams = config.get("upstreams") or {}
    if not isinstance(upstreams, dict):
        return []
    names: list[str] = []
    for name, entry in upstreams.items():
        if not isinstance(name, str) or not name:
            continue
        api_format = ""
        if isinstance(entry, dict):
            api_format = str(entry.get("api_format") or "openai").strip().lower()
        if api_format == "anthropic":
            continue
        names.append(name)
    return names


def fetch_upstream_models(
    config: dict[str, Any],
    upstream_name: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> Any:
    """GET ``{base_url}/models`` с ключом upstream. Только метаданные."""
    upstreams = config.get("upstreams") or {}
    if upstream_name not in source_allowlist(config):
        raise ValueError(f"upstream '{upstream_name}' is not in the configured allowlist")
    upstream = upstreams.get(upstream_name) or {}
    base_url = str(upstream.get("base_url", "")).rstrip("/")
    if not base_url:
        raise ValueError(f"upstream '{upstream_name}' has no base_url")
    api_key_env = upstream.get("api_key_env")
    api_key = os.environ.get(str(api_key_env)) if api_key_env else None
    if not api_key:
        raise RuntimeError(
            f"upstream '{upstream_name}' key env '{api_key_env}' is empty or missing"
        )
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    extra = upstream.get("headers") or {}
    if isinstance(extra, dict):
        headers.update({str(k): str(v) for k, v in extra.items()})
    kwargs: dict[str, Any] = {"timeout": timeout, "follow_redirects": False}
    if transport is not None:
        kwargs["transport"] = transport
    with httpx.Client(**kwargs) as client:
        response = client.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        return response.json()


def refresh_catalog(
    config: dict[str, Any],
    existing: dict[str, Any] | None,
    *,
    fetch: Callable[[str], Any] | None = None,
    aliases: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
    native_models: list[dict[str, Any]] | None = None,
    source_kind: str = "live",
    source_timestamps: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], RefreshReport]:
    """Обновляет каталог с сохранением last-known-good.

    ``fetch(upstream_name) -> payload`` — точка подмены для тестов. По умолчанию
    используется :func:`fetch_upstream_models`. ``aliases`` — карта alias→модель;
    если не задана, берётся из ``config["models"]``. ``source_kind`` —
    ``"live"`` или ``"fixture"``; ``source_timestamps`` — исходные ``fetched_at``
    для фикстур (чтобы не выдавать их за свежий live-запрос).
    """
    previous = existing if isinstance(existing, dict) else empty_catalog()
    catalog = empty_catalog()
    catalog["generated_at"] = previous.get("generated_at")
    catalog["sources"] = copy.deepcopy(previous.get("sources") or {})
    catalog["models"] = copy.deepcopy(previous.get("models") or {})
    report = RefreshReport()

    if aliases is None:
        raw_aliases = config.get("models") or {}
        aliases = raw_aliases if isinstance(raw_aliases, dict) else {}

    if fetch is None:

        def fetch(name: str) -> Any:
            return fetch_upstream_models(config, name, timeout=timeout, transport=transport)

    normalized: dict[str, list[dict[str, Any]]] = {}
    for name in source_allowlist(config):
        try:
            payload = fetch(name)
        except Exception as exc:  # noqa: BLE001 - любой сбой = LKG
            report.upstreams[name] = {
                "status": "error",
                "error": type(exc).__name__,
                "kept_last_known_good": True,
            }
            continue
        try:
            entries = normalize_upstream_models(name, payload, native_models)
        except Exception as exc:  # noqa: BLE001
            report.upstreams[name] = {
                "status": "error",
                "error": f"normalize:{type(exc).__name__}",
                "kept_last_known_good": True,
            }
            continue
        if not entries:
            report.upstreams[name] = {
                "status": "error",
                "error": "empty_metadata",
                "kept_last_known_good": True,
            }
            continue
        normalized[name] = entries
        processed_at = now_utc_iso()
        original_ts = (source_timestamps or {}).get(name) or processed_at
        catalog["sources"][name] = {
            "kind": source_kind,
            "type": source_kind,
            "url": str((config.get("upstreams") or {}).get(name, {}).get("base_url", "")),
            "fetched_at": original_ts,
            "processed_at": processed_at,
            "models": len(entries),
        }
        report.upstreams[name] = {"status": "ok", "models": len(entries)}

    fresh = build_alias_entries(aliases, normalized)
    previous_models = previous.get("models") if isinstance(previous.get("models"), dict) else {}
    for alias, spec in aliases.items():
        if not isinstance(spec, dict):
            continue
        upstream = spec.get("upstream")
        if not isinstance(upstream, str):
            continue
        if upstream not in normalized:
            if alias in catalog["models"]:
                report.kept_last_known_good.append(alias)
                kept = catalog["models"][alias]
                if isinstance(kept, dict):
                    kept["stale"] = True
            continue
        record = fresh.get(alias)
        if record is None:
            report.models_missing.append(alias)
            report.kept_last_known_good.append(alias)
            continue
        old = previous_models.get(alias)
        merged = merge_entry(old, record)
        merged["alias"] = alias
        merged["gateway_name"] = spec.get("name")
        merged["source"] = upstream
        merged["source_type"] = source_kind
        merged["fetched_at"] = catalog["sources"].get(upstream, {}).get("fetched_at")
        merged["processed_at"] = catalog["sources"].get(upstream, {}).get("processed_at")
        if merged.get("stale") and isinstance(old, dict):
            merged["stale_fields_from"] = old.get("fetched_at")
        catalog["models"][alias] = merged
        report.models_updated.append(alias)

    # Prune aliases, которых больше нет в конфиге (роут удалён/переименован).
    if aliases:
        for alias in list(catalog["models"]):
            if alias not in aliases:
                del catalog["models"][alias]
                report.models_pruned.append(alias)

    if report.models_updated:
        catalog["generated_at"] = now_utc_iso()
    return catalog, report


def has_precise_metadata(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    limit = entry.get("limit") or {}
    return (
        positive_int(limit.get("context")) is not None
        or positive_int(limit.get("output")) is not None
        or bool(entry.get("cost"))
    )
