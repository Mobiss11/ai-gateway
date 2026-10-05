"""CLI метаданных AI Gateway.

Примеры::

    # Офлайн-сид из sanitized-фикстур (без сети)
    uv run python catalog_cli.py refresh --config ~/.config/ai-gateway/config.json \
        --catalog ~/.config/ai-gateway/catalog.json \
        --from-fixture audit/upstream-models.json \
        --native-models audit/opencode-models-before.json

    # Явное metadata-only обновление (GET /models по allowlist upstream-ов)
    uv run python catalog_cli.py refresh --config ... --catalog ... --live

    # Dry-run синхронизации OpenCode (по умолчанию ничего не пишет)
    uv run python catalog_cli.py sync --config ... --catalog ... \
        --opencode /path/opencode.json

    # Применение с backup 0600 + атомарной записью
    uv run python catalog_cli.py sync ... --apply

Секреты не печатаются: конфиг шлюза читается только для alias/upstream/url.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import catalog as catalog_mod
import metadata_refresh
import opencode_sync


def _load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_gateway_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return _load_json(path)


def _load_native_models(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    data = _load_json(path)
    if isinstance(data, dict):
        models = data.get("data")
        if isinstance(models, list):
            return [m for m in models if isinstance(m, dict)]
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    return []


def _aliases(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.aliases:
        data = _load_json(args.aliases)
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            return data["models"]
        if isinstance(data, dict):
            return data
    aliases = config.get("models")
    return aliases if isinstance(aliases, dict) else {}


def _cmd_refresh(args: argparse.Namespace) -> int:
    config = _load_gateway_config(args.config)
    existing = catalog_mod.load_catalog(args.catalog)
    aliases = _aliases(args, config)
    native = _load_native_models(args.native_models)
    if args.from_fixture:
        fixture = _load_json(args.from_fixture)
        upstreams = (
            fixture.get("upstreams")
            if isinstance(fixture, dict) and isinstance(fixture.get("upstreams"), dict)
            else (fixture if isinstance(fixture, dict) else {})
        )
        fixture_ts = fixture.get("fetched_at") if isinstance(fixture, dict) else None
        # В фикстурном режиме allowlist — сами фикстуры (плюс уже настроенные).
        config["upstreams"] = {
            **{n: {"base_url": ""} for n in upstreams},
            **(config.get("upstreams") or {}),
        }

        def fetch(name: str) -> Any:
            if name not in upstreams:
                raise KeyError(f"no fixture for upstream '{name}'")
            return upstreams[name]

        new_catalog, report = metadata_refresh.refresh_catalog(
            config,
            existing,
            fetch=fetch,
            aliases=aliases,
            native_models=native,
            source_kind="fixture",
            source_timestamps={name: fixture_ts for name in upstreams},
        )
    elif args.live:
        env_file = args.env_file
        loaded_env = False
        try:
            import gateway  # локальный импорт: тянет web-стек только для live

            if not env_file:
                env_file = gateway.DEFAULT_ENV_FILE
            gateway.load_env_file(env_file)
            loaded_env = True
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not load env file {env_file}: {exc}", file=sys.stderr)
        new_catalog, report = metadata_refresh.refresh_catalog(
            config, existing, aliases=aliases, timeout=args.timeout, native_models=native
        )
        if loaded_env:
            print(f"loaded upstream keys from env file: {env_file}")
    else:
        print(
            "error: укажите --from-fixture FILE или явный --live (metadata-only discovery)",
            file=sys.stderr,
        )
        return 2

    report_data = report.as_dict()
    if not report.models_updated and existing.get("models"):
        print(
            "refresh failed: все источники недоступны; last-known-good сохранён, файл не изменён"
        )
        print(json.dumps(report_data, ensure_ascii=False, indent=2))
        return 1

    catalog_mod.save_catalog(args.catalog, new_catalog)
    print(
        f"catalog updated: {len(new_catalog.get('models') or {})} models, "
        f"generated_at={new_catalog.get('generated_at')}"
    )
    print(json.dumps(report_data, ensure_ascii=False, indent=2))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    data = catalog_mod.load_catalog(args.catalog)
    models = data.get("models") or {}
    print(f"generated_at={data.get('generated_at')} models={len(models)}")
    for alias, entry in models.items():
        if not isinstance(entry, dict):
            continue
        limit = entry.get("limit") or {}
        print(
            f"- {alias}: upstream={entry.get('upstream')} model={entry.get('upstream_model')} "
            f"quality={entry.get('metadata_quality')} "
            f"context={limit.get('context')} output={limit.get('output')} "
            f"efforts={(entry.get('reasoning') or {}).get('supported_efforts')} "
            f"package={entry.get('package')} recommended_disabled={entry.get('recommended_disabled')}"
        )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    data = catalog_mod.load_catalog(args.catalog)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"exported {len(data.get('models') or {})} models to {out}")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    catalog = catalog_mod.load_catalog(args.catalog)
    config = opencode_sync.load_opencode_config(args.opencode)
    gateway_aliases = None
    if args.config:
        gateway_config = _load_gateway_config(args.config)
        aliases = gateway_config.get("models")
        gateway_aliases = aliases if isinstance(aliases, dict) else {}
    plan = opencode_sync.plan_sync(
        catalog,
        config,
        provider=args.provider,
        add_missing=args.add_missing,
        disable_recommended=args.disable_recommended,
        gateway_aliases=gateway_aliases,
        request_output_budget=args.request_output_budget,
    )
    summary = plan.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not plan.changed:
        print("no changes; nothing to do")
        return 0
    if not args.apply:
        print("dry-run: nothing written (pass --apply to write a protected backup + atomic update)")
        return 0
    backup = opencode_sync.apply_sync(args.opencode, plan.config or {})
    print(f"applied; backup={backup}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Gateway metadata catalog CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--catalog", required=True, help="path to catalog.json")

    refresh = sub.add_parser("refresh", help="refresh metadata catalog")
    add_common(refresh)
    refresh.add_argument("--config", help="gateway config (aliases/upstreams allowlist)")
    refresh.add_argument("--aliases", help="alias map JSON (falls back to config.models)")
    refresh.add_argument("--native-models", help="resolved OpenCode models dump for cost/package")
    refresh.add_argument("--from-fixture", help="sanitized upstream-models fixture (offline)")
    refresh.add_argument("--live", action="store_true", help="explicit metadata-only discovery")
    refresh.add_argument("--timeout", type=float, default=metadata_refresh.DEFAULT_TIMEOUT_SECONDS)
    refresh.add_argument(
        "--env-file",
        default=None,
        help="env file with upstream keys for --live (default: gateway env file)",
    )
    refresh.set_defaults(func=_cmd_refresh)

    show = sub.add_parser("show", help="print catalog summary")
    add_common(show)
    show.set_defaults(func=_cmd_show)

    export = sub.add_parser("export", help="export catalog JSON")
    add_common(export)
    export.add_argument("--out", required=True)
    export.set_defaults(func=_cmd_export)

    sync = sub.add_parser("sync", help="sync catalog into OpenCode V2 config")
    add_common(sync)
    sync.add_argument("--opencode", required=True, help="path to OpenCode opencode.json")
    sync.add_argument(
        "--config",
        help="gateway config; enables route-match verification against config.models",
    )
    sync.add_argument("--provider", default=opencode_sync.MANAGED_PROVIDER_DEFAULT)
    sync.add_argument("--add-missing", action="store_true", help="create models absent from config")
    sync.add_argument(
        "--disable-recommended",
        action="store_true",
        help="set disabled=true for catalog models flagged recommended_disabled",
    )
    sync.add_argument("--apply", action="store_true", help="perform apply (default: dry-run)")
    sync.add_argument(
        "--request-output-budget",
        type=int,
        default=None,
        help=(
            "EXPLICIT opt-in: cap model limit.output (verified field) to N tokens; "
            "default: unchanged"
        ),
    )
    sync.set_defaults(func=_cmd_sync)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
