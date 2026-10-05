"""AI Gateway — OpenAI-совместимый шлюз к нескольким LLM-провайдерам.

Прозрачный прокси: держит API-ключи провайдеров (DeepSeek, OpenRouter, ...),
отдаёт единый OpenAI-совместимый endpoint для OpenCode и других инструментов.

Запуск: ``python gateway.py`` (uvicorn на host/port из конфига).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:  # каталог метаданных опционален: без него шлюз работает как раньше
    import catalog as catalog_mod
except ImportError:  # pragma: no cover - защита от одиночного запуска
    catalog_mod = None  # type: ignore[assignment]

try:  # мост к Anthropic Messages API нужен только api_format=anthropic
    import anthropic_bridge
except ImportError:  # pragma: no cover - защита от одиночного запуска
    anthropic_bridge = None  # type: ignore[assignment]

try:  # мост к локальному Claude Code CLI нужен только api_format=claude-cli
    import claude_cli_bridge
except ImportError:  # pragma: no cover - защита от одиночного запуска
    claude_cli_bridge = None  # type: ignore[assignment]

try:  # отчёт usage-метрик в журнал Борта; включается bort_usage_url
    import usage_report
except ImportError:  # pragma: no cover - защита от одиночного запуска
    usage_report = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = "~/.config/ai-gateway/config.json"
DEFAULT_ENV_FILE = "~/.config/ai-gateway/env"

# Anthropic классифицирует запросы сторонних агентных клиентов как
# «third-party app»: они тарифицируются через extra-usage кредиты, а не
# лимиты плана, и без кредитов детерминированно 400-ят («Third-party apps
# now draw from your extra usage…», 04.10.2026). Триггер — не агентная
# идентичность, а env-секция в системном промпте (Today's date + интро +
# <env>…</env>); флаг upstream'а move_env_to_user переносит её в первое
# user-сообщение (см. anthropic_bridge). Это не флейк — но на случай
# genuinely транзиентных 400 оставлен короткий повтор (2×~1 c).
_ANTHROPIC_FLAKE_BACKOFF = (0.5, 1.0)
_ANTHROPIC_FLAKE_MARKER = b"extra usage"

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}

# httpx сам распаковывает gzip/br/deflate, поэтому отдавать content-encoding
# upstream клиенту нельзя: тело уже декодировано.
_UNSAFE_PASSTHROUGH = {"content-encoding"}

# Поля usage, которые можно логировать (числа, без секретов). cache write не
# синтезируем: логируем только то, что реально вернул upstream.
_USAGE_NUMERIC_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

_TOKEN_IN_TEXT_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_KEY_IN_TEXT_RE = re.compile(r"\b(sk-[A-Za-z0-9._-]{6,})")

log = logging.getLogger("ai_gateway")


# --------------------------------------------------------------------------- #
# Конфигурация и переменные окружения
# --------------------------------------------------------------------------- #
def load_env_file(path: str | os.PathLike[str]) -> None:
    """Простой парсер KEY=VALUE. Строки с # игнорируются.

    Существующие переменные окружения НЕ перезаписываются.
    """
    env_path = Path(path).expanduser()
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Читает JSON-конфиг. Путь берётся из аргумента или env AI_GATEWAY_CONFIG."""
    raw_path = path or os.environ.get("AI_GATEWAY_CONFIG", DEFAULT_CONFIG_PATH)
    config_path = Path(raw_path).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Config file is not valid JSON: {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"Config root must be a JSON object: {config_path}")
    return config


def resolve_token(config: dict[str, Any]) -> str | None:
    """Токен клиента: auth.token, иначе значение переменной auth.token_env."""
    auth = config.get("auth") or {}
    token = auth.get("token")
    if token:
        return str(token)
    env_name = auth.get("token_env")
    if env_name:
        value = os.environ.get(str(env_name))
        if value:
            return value
    return None


def resolve_upstream_model(
    models: dict[str, Any],
    upstreams: dict[str, Any],
    requested: Any,
) -> tuple[str, str] | None:
    """(upstream, provider_model) для запрошенного id или None."""
    if not isinstance(requested, str) or not requested:
        return None
    entry = models.get(requested)
    if isinstance(entry, dict):
        upstream = entry.get("upstream")
        model = entry.get("model", requested)
        if upstream in upstreams:
            return str(upstream), str(model)
    # passthrough: "<upstream>/<provider-model>"
    prefix, sep, rest = requested.partition("/")
    if sep and rest and prefix in upstreams:
        return prefix, rest
    return None


# --------------------------------------------------------------------------- #
# Приложение
# --------------------------------------------------------------------------- #
def _openai_error(
    message: str,
    *,
    status_code: int,
    err_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


def _passthrough_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _HOP_BY_HOP
        and key.lower() not in _UNSAFE_PASSTHROUGH
    }


def sanitize_error_text(text: str) -> str:
    """Убирает возможные токены/ключи из текста исключения перед отдачей."""
    text = _TOKEN_IN_TEXT_RE.sub(r"\1<redacted>", text)
    text = _KEY_IN_TEXT_RE.sub("<redacted>", text)
    return text


def _route_matches_config(
    alias: str, catalog_entry: dict[str, Any], entry: dict[str, Any]
) -> bool:
    """Каталог применим только если alias всё ещё роутится в ту же модель."""
    if catalog_mod is None:
        return False
    expected_model = entry.get("model") or alias
    return catalog_mod.route_matches(catalog_entry, entry.get("upstream"), expected_model)


def _enrich_model_entry(
    alias: str,
    entry: dict[str, Any],
    catalog_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Базовая OpenAI-форма + метаданные каталога, если они известны.

    Метаданные применяются только при совпадении маршрута alias (upstream +
    upstream-модель); иначе alias считается перенаправленным и запись каталога
    игнорируется. Без каталога форма остаётся прежней.
    """
    data: dict[str, Any] = {
        "id": alias,
        "object": "model",
        "owned_by": entry.get("upstream"),
        "name": entry.get("name", alias),
    }
    if not catalog_entry:
        return data
    if not _route_matches_config(alias, catalog_entry, entry):
        data["metadata"] = {
            "source": catalog_entry.get("upstream"),
            "source_type": catalog_entry.get("source_type"),
            "quality": catalog_entry.get("metadata_quality"),
            "route_mismatch": True,
        }
        return data

    if isinstance(catalog_entry.get("name"), str) and not entry.get("name"):
        data["name"] = catalog_entry["name"]

    limit = catalog_entry.get("limit") if isinstance(catalog_entry.get("limit"), dict) else {}
    context = limit.get("context")
    output = limit.get("output")
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        data["context_window"] = context
    if isinstance(output, int) and not isinstance(output, bool) and output > 0:
        data["max_output_tokens"] = output
    if "context_window" in data and "max_output_tokens" in data:
        data["limit"] = {
            "context": data["context_window"],
            "output": data["max_output_tokens"],
        }

    capabilities = catalog_entry.get("capabilities")
    if isinstance(capabilities, dict):
        data["capabilities"] = capabilities
    reasoning = catalog_entry.get("reasoning")
    if isinstance(reasoning, dict):
        data["reasoning"] = reasoning
    if isinstance(catalog_entry.get("package"), str):
        data["package"] = catalog_entry["package"]
    cost = catalog_entry.get("cost")
    if isinstance(cost, list) and cost:
        data["cost"] = cost
    data["metadata"] = {
        "source": catalog_entry.get("source") or catalog_entry.get("upstream"),
        "source_type": catalog_entry.get("source_type"),
        "fetched_at": catalog_entry.get("fetched_at"),
        "processed_at": catalog_entry.get("processed_at"),
        "quality": catalog_entry.get("metadata_quality"),
        "stale": bool(catalog_entry.get("stale")),
        "retained_fields": list(catalog_entry.get("retained_fields") or []),
    }
    if catalog_entry.get("recommended_disabled"):
        data["recommended_disabled"] = True
    return data


def create_app(
    config: dict[str, Any],
    *,
    transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    client_factory: Callable[[dict[str, Any]], httpx.AsyncClient] | None = None,
    catalog: dict[str, Any] | None = None,
) -> FastAPI:
    """Собирает FastAPI-приложение из уже загруженного конфига.

    ``transport``, ``client_factory`` и ``catalog`` — точки подмены для тестов.
    """
    token = resolve_token(config)
    if not token:
        token_env = (config.get("auth") or {}).get("token_env")
        raise RuntimeError(
            "AI Gateway token is not configured. Set auth.token in the config "
            f"or define the environment variable named by auth.token_env"
            + (f" ({token_env})" if token_env else "")
            + "."
        )

    upstreams: dict[str, Any] = config.get("upstreams") or {}
    models: dict[str, Any] = config.get("models") or {}
    timeout = float(config.get("request_timeout_seconds", 300))

    # claude-cli: ограничение параллельных процессов CLI на каждый upstream.
    claude_semaphores: dict[str, asyncio.Semaphore] = {
        name: asyncio.Semaphore(
            max(1, int((cfg or {}).get("max_concurrent", 2)))
        )
        for name, cfg in upstreams.items()
        if isinstance(cfg, dict)
        and str(cfg.get("api_format") or "").strip().lower() == "claude-cli"
    }
    catalog_path = config.get("catalog_path")
    catalog_state: dict[str, Any] = {
        "path": catalog_path,
        "mtime_ns": None,
        "size": None,
        "data": {},
    }
    if catalog is not None:
        catalog_state["data"] = catalog
    elif catalog_mod is not None:
        catalog_state["data"] = catalog_mod.load_catalog(catalog_path)
        if catalog_path:
            try:
                st = os.stat(Path(str(catalog_path)).expanduser())
                catalog_state["mtime_ns"] = st.st_mtime_ns
                catalog_state["size"] = st.st_size
            except OSError:
                pass

    def current_catalog() -> dict[str, Any]:
        """Перечитывает каталог только при смене mtime/size (без сети)."""
        if catalog is not None or catalog_mod is None or not catalog_state["path"]:
            return catalog_state["data"]
        try:
            st = os.stat(Path(str(catalog_state["path"])).expanduser())
        except OSError:
            return catalog_state["data"]
        if (
            st.st_mtime_ns != catalog_state["mtime_ns"]
            or st.st_size != catalog_state["size"]
        ):
            catalog_state["data"] = catalog_mod.load_catalog(catalog_state["path"])
            catalog_state["mtime_ns"] = st.st_mtime_ns
            catalog_state["size"] = st.st_size
        return catalog_state["data"]

    if client_factory is None:

        def client_factory(_cfg: dict[str, Any]) -> httpx.AsyncClient:
            kwargs: dict[str, Any] = {"timeout": timeout, "follow_redirects": False}
            if transport is not None:
                kwargs["transport"] = transport
            return httpx.AsyncClient(**kwargs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = client_factory(config)
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="AI Gateway", lifespan=lifespan)
    app.state.config = config

    # Журнал Борта: отчёт usage после каждого хода (None — выключено).
    bort_url = config.get("bort_usage_url")
    if bort_url and usage_report is None:  # pragma: no cover - одиночный запуск
        log.warning("bort_usage_url set but usage_report module unavailable")
        bort_url = None

    async def _send_bort_report(
        client: httpx.AsyncClient,
        *,
        upstream_name: str,
        provider_model: str,
        started: float,
        usage: dict[str, Any] | None = None,
        request_id: str | None = None,
        status: str = "success",
        error_code: str | None = None,
    ) -> None:
        if not bort_url:
            return
        payload = usage_report.build_payload(
            provider=upstream_name,
            model=provider_model,
            usage=usage,
            request_id=request_id,
            status=status,
            error_code=error_code,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        await usage_report.report(client, bort_url, payload)

    def _authorized(request: Request) -> bool:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return False
        presented = header[7:].strip()
        return bool(presented) and secrets.compare_digest(presented, token)

    def _unauthorized() -> JSONResponse:
        response = _openai_error(
            "Unauthorized: missing or invalid bearer token", status_code=401
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    # ------------------------------------------------------------------ #
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "upstreams": sorted(upstreams.keys())}

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()
        data = [
            _enrich_model_entry(
                alias,
                entry if isinstance(entry, dict) else {},
                catalog_mod.catalog_entry(current_catalog(), alias)
                if catalog_mod is not None
                else None,
            )
            for alias, entry in models.items()
        ]
        return JSONResponse({"object": "list", "data": data})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if not _authorized(request):
            return _unauthorized()

        started = time.perf_counter()
        try:
            body = await request.json()
        except Exception:
            return _openai_error("Request body must be valid JSON", status_code=400)
        if not isinstance(body, dict):
            return _openai_error("Request body must be a JSON object", status_code=400)

        requested_model = body.get("model")
        resolved = resolve_upstream_model(models, upstreams, requested_model)
        if resolved is None:
            available = ", ".join(models.keys()) or "(none configured)"
            return _openai_error(
                f"Model '{requested_model}' is not configured. Available aliases: "
                f"{available}. You can also use '<upstream>/<model>' passthrough.",
                status_code=404,
                code="model_not_found",
            )

        upstream_name, provider_model = resolved
        upstream = upstreams.get(upstream_name) or {}

        api_format = str(upstream.get("api_format") or "openai").strip().lower()
        if api_format not in ("openai", "anthropic", "claude-cli"):
            return _openai_error(
                f"Upstream '{upstream_name}' has unsupported api_format "
                f"'{api_format}' (expected 'openai', 'anthropic' or 'claude-cli').",
                status_code=500,
                err_type="upstream_error",
            )
        if api_format == "claude-cli":
            # Локальный CLI: base_url/api_key не нужны, вместо них — процесс.
            if claude_cli_bridge is None:  # pragma: no cover - одиночный запуск
                return _openai_error(
                    "claude_cli_bridge module is unavailable",
                    status_code=500,
                    err_type="upstream_error",
                )
            return await claude_cli_bridge.handle_chat(
                request.app.state.client,
                body,
                upstream=upstream,
                upstream_name=upstream_name,
                provider_model=provider_model,
                requested_model=requested_model,
                started=started,
                report_bort=_send_bort_report,
                timeout_seconds=timeout,
                semaphore=claude_semaphores.get(upstream_name),
            )

        base_url = str(upstream.get("base_url", "")).rstrip("/")
        if not base_url:
            return _openai_error(
                f"Upstream '{upstream_name}' has no base_url configured.",
                status_code=502,
                err_type="upstream_error",
            )

        api_key_env = upstream.get("api_key_env")
        api_key = os.environ.get(str(api_key_env)) if api_key_env else None
        if not api_key:
            log.warning(
                "upstream key missing upstream=%s env=%s", upstream_name, api_key_env
            )
            return _openai_error(
                f"Upstream '{upstream_name}' is not configured: environment variable "
                f"'{api_key_env}' is empty or missing.",
                status_code=502,
                err_type="upstream_error",
            )

        if api_format == "anthropic":
            return await _proxy_anthropic_chat(
                request.app.state.client,
                body,
                upstream=upstream,
                upstream_name=upstream_name,
                base_url=base_url,
                api_key=api_key,
                requested_model=requested_model,
                provider_model=provider_model,
                started=started,
                report_bort=_send_bort_report,
            )

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        extra_headers = upstream.get("headers") or {}
        if isinstance(extra_headers, dict):
            headers.update({str(k): str(v) for k, v in extra_headers.items()})

        body["model"] = provider_model
        is_stream = bool(body.get("stream"))
        url = f"{base_url}/chat/completions"
        client: httpx.AsyncClient = request.app.state.client

        outgoing = client.build_request("POST", url, json=body, headers=headers)
        try:
            upstream_response = await client.send(outgoing, stream=True)
        except httpx.TimeoutException as exc:
            _log_call(requested_model, provider_model, 502, started)
            await _send_bort_report(
                client, upstream_name=upstream_name, provider_model=provider_model,
                started=started, status="error", error_code="timeout",
            )
            return _openai_error(
                f"Upstream '{upstream_name}' timed out: {sanitize_error_text(str(exc))}",
                status_code=502,
                err_type="upstream_error",
            )
        except httpx.HTTPError as exc:
            _log_call(requested_model, provider_model, 502, started)
            await _send_bort_report(
                client, upstream_name=upstream_name, provider_model=provider_model,
                started=started, status="error", error_code="network",
            )
            return _openai_error(
                f"Upstream '{upstream_name}' request failed: {sanitize_error_text(str(exc))}",
                status_code=502,
                err_type="upstream_error",
            )

        _log_call(
            requested_model,
            provider_model,
            upstream_response.status_code,
            started,
        )

        if is_stream and upstream_response.status_code < 400:
            sniffer = usage_report.OpenAIUsageSniffer() if bort_url else None

            async def stream_body() -> Iterator[bytes]:
                try:
                    async for chunk in upstream_response.aiter_bytes():
                        if sniffer is not None:
                            sniffer.feed(chunk)
                        yield chunk
                finally:
                    await upstream_response.aclose()
                    if sniffer is not None:
                        await _send_bort_report(
                            client,
                            upstream_name=upstream_name,
                            provider_model=provider_model,
                            started=started,
                            usage=sniffer.usage,
                            request_id=sniffer.request_id,
                        )

            return StreamingResponse(
                stream_body(),
                status_code=upstream_response.status_code,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            content = await upstream_response.aread()
        finally:
            await upstream_response.aclose()
        if upstream_response.status_code < 400:
            _log_usage(upstream_name, content)
            await _send_bort_report(
                client,
                upstream_name=upstream_name,
                provider_model=provider_model,
                started=started,
                usage=_openai_response_usage(content),
                request_id=_openai_response_id(content),
            )
        else:
            await _send_bort_report(
                client,
                upstream_name=upstream_name,
                provider_model=provider_model,
                started=started,
                status="error",
                error_code=f"http_{upstream_response.status_code}",
            )
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=_passthrough_headers(upstream_response),
        )

    return app


# --------------------------------------------------------------------------- #
# Anthropic Messages upstream (api_format: "anthropic", например kraube serve)
# --------------------------------------------------------------------------- #
async def _noop_bort_report(client: httpx.AsyncClient, **_kwargs: Any) -> None:
    """Заглушка: отчёт в Борт выключен."""


async def _proxy_anthropic_chat(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    *,
    upstream: dict[str, Any],
    upstream_name: str,
    base_url: str,
    api_key: str,
    requested_model: Any,
    provider_model: str,
    started: float,
    report_bort: Callable[..., Awaitable[None]] = _noop_bort_report,
) -> Response:
    """Переводит chat/completions → /v1/messages, ответ — обратно.

    Формат перевода — в :mod:`anthropic_bridge`; здесь только HTTP-часть:
    те же таймауты, логи вызовов/usage и санитизация ошибок, что и в
    OpenAI-пути.
    """
    if anthropic_bridge is None:  # pragma: no cover - одиночный запуск
        return _openai_error(
            "anthropic_bridge module is unavailable",
            status_code=500,
            err_type="upstream_error",
        )
    env_move = upstream.get("move_env_to_user")
    try:
        anthropic_request = anthropic_bridge.openai_to_anthropic(
            body,
            provider_model,
            move_env_to_user=isinstance(env_move, bool) and env_move,
        )
    except anthropic_bridge.TranslationError as exc:
        return _openai_error(
            f"Request cannot be translated to Anthropic Messages format: {exc}",
            status_code=400,
        )

    # Отладка «extra usage»-400: AI_GATEWAY_DUMP_REQUESTS=/tmp/dir пишет тело
    # запроса до отправки (по env-флагу, только локальные файлы).
    dump_dir = os.environ.get("AI_GATEWAY_DUMP_REQUESTS")
    if dump_dir:
        try:
            dump_path = Path(dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)
            (dump_path / f"req-{int(time.time() * 1000)}.json").write_text(
                json.dumps(anthropic_request, ensure_ascii=False)[:262144],
                encoding="utf-8",
            )
        except OSError:
            pass

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    extra_headers = upstream.get("headers") or {}
    if isinstance(extra_headers, dict):
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    is_stream = bool(body.get("stream"))
    stream_options = body.get("stream_options")
    include_usage = bool(
        isinstance(stream_options, dict) and stream_options.get("include_usage")
    )
    url = f"{base_url}/v1/messages"

    async def _send() -> httpx.Response:
        outgoing = client.build_request("POST", url, json=anthropic_request, headers=headers)
        return await client.send(outgoing, stream=True)

    async def _timeout(exc: httpx.TimeoutException) -> Response:
        _log_call(requested_model, provider_model, 502, started)
        await report_bort(
            client, upstream_name=upstream_name, provider_model=provider_model,
            started=started, status="error", error_code="timeout",
        )
        return _openai_error(
            f"Upstream '{upstream_name}' timed out: {sanitize_error_text(str(exc))}",
            status_code=502,
            err_type="upstream_error",
        )

    async def _network(exc: httpx.HTTPError) -> Response:
        _log_call(requested_model, provider_model, 502, started)
        await report_bort(
            client, upstream_name=upstream_name, provider_model=provider_model,
            started=started, status="error", error_code="network",
        )
        return _openai_error(
            f"Upstream '{upstream_name}' request failed: {sanitize_error_text(str(exc))}",
            status_code=502,
            err_type="upstream_error",
        )

    try:
        upstream_response = await _send()
    except httpx.TimeoutException as exc:
        return await _timeout(exc)
    except httpx.HTTPError as exc:
        return await _network(exc)

    _log_call(requested_model, provider_model, upstream_response.status_code, started)

    # Спорадический «extra usage» 400: повторяем, клиент флейка не видит.
    for attempt, delay in enumerate(_ANTHROPIC_FLAKE_BACKOFF):
        if upstream_response.status_code != 400:
            break
        try:
            probe = await upstream_response.aread()
        finally:
            await upstream_response.aclose()
        if _ANTHROPIC_FLAKE_MARKER not in probe.lower():
            # чужой 400 — без ретрая: вернуть уже прочитанное тело
            try:
                parsed = json.loads(probe)
            except (ValueError, TypeError):
                parsed = None
            error_payload = anthropic_bridge.anthropic_error_to_openai(parsed)
            if parsed is None and probe:
                error_payload["error"]["message"] = sanitize_error_text(
                    probe[:500].decode("utf-8", "replace")
                )
            await report_bort(
                client, upstream_name=upstream_name, provider_model=provider_model,
                started=started, status="error", error_code="http_400",
            )
            return JSONResponse(status_code=400, content=error_payload)
        log.warning(
            "anthropic extra-usage flake upstream=%s model=%s attempt=%s delay=%.1fs — retrying",
            upstream_name, provider_model, attempt + 1, delay,
        )
        await asyncio.sleep(delay)
        try:
            upstream_response = await _send()
        except httpx.TimeoutException as exc:
            return await _timeout(exc)
        except httpx.HTTPError as exc:
            return await _network(exc)
        _log_call(requested_model, provider_model, upstream_response.status_code, started)

    if is_stream and upstream_response.status_code < 400:
        translator = anthropic_bridge.AnthropicStreamTranslator(
            str(requested_model), include_usage=include_usage
        )

        async def stream_body() -> Iterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_bytes():
                    for frame in translator.feed(chunk):
                        yield frame
                for frame in translator.finish():
                    yield frame
            finally:
                await upstream_response.aclose()
                await report_bort(
                    client,
                    upstream_name=upstream_name,
                    provider_model=provider_model,
                    started=started,
                    usage=translator.last_usage(),
                    request_id=translator.last_id(),
                )

        return StreamingResponse(
            stream_body(),
            status_code=200,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        content = await upstream_response.aread()
    finally:
        await upstream_response.aclose()

    if upstream_response.status_code >= 400:
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            parsed = None
        error_payload = anthropic_bridge.anthropic_error_to_openai(parsed)
        if parsed is None and content:
            error_payload["error"]["message"] = sanitize_error_text(
                content[:500].decode("utf-8", "replace")
            )
        await report_bort(
            client,
            upstream_name=upstream_name,
            provider_model=provider_model,
            started=started,
            status="error",
            error_code=f"http_{upstream_response.status_code}",
        )
        return JSONResponse(
            status_code=upstream_response.status_code, content=error_payload
        )

    try:
        payload = json.loads(content)
        openai_response = anthropic_bridge.anthropic_to_openai(
            payload, str(requested_model)
        )
    except (ValueError, TypeError) as exc:
        return _openai_error(
            f"Upstream '{upstream_name}' returned an invalid response: "
            f"{sanitize_error_text(str(exc))}",
            status_code=502,
            err_type="upstream_error",
        )
    _log_usage(upstream_name, json.dumps(openai_response).encode("utf-8"))
    await report_bort(
        client,
        upstream_name=upstream_name,
        provider_model=provider_model,
        started=started,
        usage=openai_response.get("usage"),
        request_id=openai_response.get("id"),
    )
    return JSONResponse(status_code=200, content=openai_response)


def _openai_response_usage(content: bytes) -> dict[str, Any] | None:
    """usage из тела OpenAI-ответа (для отчёта в журнал Борта)."""
    if not content:
        return None
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _openai_response_id(content: bytes) -> str | None:
    """id из тела OpenAI-ответа — ключ дедупа событий в журнале."""
    if not content:
        return None
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("id"), str):
        return payload["id"]
    return None


def _log_call(
    requested_model: Any, provider_model: str, status_code: int, started: float
) -> None:
    duration_ms = (time.perf_counter() - started) * 1000
    log.info(
        "chat_completions requested=%s routed=%s status=%s duration_ms=%.1f",
        requested_model,
        provider_model,
        status_code,
        duration_ms,
    )


def _log_usage(upstream_name: str, content: bytes) -> None:
    """Числовая диагностика usage без содержимого ответа.

    Ничего не нормализует и не досчитывает: логирует только фактически
    вернувшиеся числовые поля (включая prompt_cache_hit/miss_tokens).
    """
    if not content:
        return
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return
    if not isinstance(payload, dict):
        return
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    parts = [
        f"{field}={usage[field]}"
        for field in _USAGE_NUMERIC_FIELDS
        if isinstance(usage.get(field), (int, float)) and not isinstance(usage.get(field), bool)
    ]
    if parts:
        log.info("usage upstream=%s %s", upstream_name, " ".join(parts))


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    env_file = os.environ.get("AI_GATEWAY_ENV_FILE", DEFAULT_ENV_FILE)
    load_env_file(env_file)

    config = load_config()
    app = create_app(config)

    listen = config.get("listen") or {}
    host = str(listen.get("host", "127.0.0.1"))
    port = int(listen.get("port", 8130))
    log.info("starting ai-gateway on http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()